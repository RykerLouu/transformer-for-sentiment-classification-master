''' Define the sublayers in encoder/decoder layer '''

import numpy as np
import torch
import torch.nn as nn
from .Modules import ScaledDotProductAttention

from global_random_seed import RANDOM_SEED
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, diagonal_positional_attention=False, position_dpa_dim=50,
                 dropout=0.1, scaled_dropout=0.1, use_batch_norm=True, residual_bool=False, temper_value=0.5):

        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.use_batch_norm = use_batch_norm
        self.residual_bool = residual_bool

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        # TODO: try # , nonlinearity='relu'
        nn.init.kaiming_normal_(self.w_qs.weight)  # xavier_normal used originally
        nn.init.kaiming_normal_(self.w_ks.weight)  # xavier_normal
        nn.init.kaiming_normal_(self.w_vs.weight)  # xavier_normal

        # new weight initialization as per (doesn't seem to improve performance):
        # https://github.com/jadore801120/attention-is-all-you-need-pytorch/commit/2077515a8ab24f4abdda9089c502fa14f32fc5d9
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.diagonal_positional_attention = diagonal_positional_attention
        self.position_dpa_dim = position_dpa_dim
        # for relative positional encodings
        if diagonal_positional_attention:
            self.w_qs_position = nn.Linear(d_model, n_head * position_dpa_dim)
            nn.init.kaiming_normal_(self.w_qs_position.weight)  # xavier_normal used originally

        # self.position_dpa2 = nn.Parameter(torch.FloatTensor(n_head, (96 * 2) - 1, d_k).cuda())

        # for dpa, fill with ones
        # self.dpa_qs = nn.Parameter(torch.FloatTensor(n_head, d_model*2, d_k).cuda())
        # init.constant(self.dpa_qs, 1)

        # TODO: test this, initially dropout was always set to 0.1!
        # TODO: higher makes the model stable, but Recall is now much lower!

        # this is from the original paper, where temper is calculated based on d_k
        self.attention = ScaledDotProductAttention(d_k, scaled_dropout, temper_value)
        # in our case, d_model works better
        # self.attention = ScaledDotProductAttention(d_model, scaled_dropout, temper_value)

        if self.use_batch_norm:  # batch norm
            self.layer_norm = nn.BatchNorm1d(d_model)
            # self.layer_norm = nn.GroupNorm(d_model, 42)
        else:  # layer norm
            self.layer_norm = nn.LayerNorm(d_model)

        # TODO: try with , bias=False.
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, output_mask=None, position_dpa_vector=None, sentence_words=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        sz_b, len_q, d_model = q.size()
        sz_b, len_k, d_model = k.size()
        sz_b, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # treat the result as a (n_head * mb_size) size batch
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # for relative positional embeddings
        if position_dpa_vector is not None:
            # map q to position_dpa_dim dimension
            sz_b1, len_q1, d_position = position_dpa_vector.size()
            # same relative positions for all self-attention heads
            position_dpa_vector = position_dpa_vector.repeat(n_head, 1, 1)  # (n*b x len_q1 x position_dpa_dim)

            q_position = self.w_qs_position(residual).view(sz_b, len_q, n_head, self.position_dpa_dim)
            q_position = q_position.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.position_dpa_dim)  # (n*b) x lq x position_dpa_dim
        else:
            q_position = None

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attns = self.attention(q, k, v,
                                       attn_mask=attn_mask,
                                       position_dpa=position_dpa_vector,
                                       q_position=q_position,
                                       sentence_words=sentence_words
                                       )                 # output size: (n_head*b, lq, dv)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        # project back to residual size, d_model
        # TODO: some people suggest to use bias=False when projecting!
        outputs = self.dropout(self.fc(output))
        # outputs = self.fc(output)   # try no dropout before BN

        if self.use_batch_norm:  # use batch norm
            # batch_norm expects (batch_size, h_units, seq_len), we have (batch_s, seq_len, h_units)
            outputs = outputs.permute(0, 2, 1)

            # have to make everything contiguous to make it run on CUDA
            if self.residual_bool:  # if new residual, add it only in PFF later
                outputs = self.layer_norm(outputs.contiguous())
            else:  # use typical self-attention implementation
                # TODO: make sure this actually works as it should
                outputs = self.layer_norm(outputs.contiguous() + residual.permute(0, 2, 1).contiguous())

            # move columns back
            outputs = outputs.permute(0, 2, 1)

        else:  # use layer norm
            if self.residual_bool:  # if new residual, add it only in PFF later
                outputs = self.layer_norm(outputs)
            else:
                outputs = self.layer_norm(outputs + residual)

        if output_mask is not None:
            # hidden output mask out for padding tokens
            outputs = outputs.masked_fill_(output_mask, 0)

        return outputs, attns


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1, use_batch_norm=True):
        super().__init__()

        self.use_batch_norm = use_batch_norm

        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)  # position-wise

        if self.use_batch_norm:
            self.layer_norm = nn.BatchNorm1d(d_hid)  # BatchNorm1d(d_hid)

            # other options here
            # self.layer_norm = nn.GroupNorm(d_hid, d_hid)
        else:
            self.layer_norm = nn.LayerNorm(d_hid)

        self.dropout = nn.Dropout(dropout)

        # instead of relu also tried: ELU, LeakyReLU, PReLU, ReLU6, RReLU, SELU
        # self.relu = nn.RReLU()  # nn.ReLU() used originally
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):

        # redirect the residual from the MultiHeadAttention directly to the end of FFN if given one
        if residual is None:
            residual = x

        output = x.transpose(1, 2)
        output = self.w_2(self.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        # try no dropout before BN

        if self.use_batch_norm:
            # batch_norm expects (batch_size, h_units, seq_len), we have (batch_s, seq_len, h_units)
            outputs = output.permute(0, 2, 1)
            residual_permuted = residual.permute(0, 2, 1)

            # have to make everything contiguous to make it run on CUDA

            outputs = outputs.contiguous() + residual_permuted.contiguous()
            outputs = self.layer_norm(outputs)
            # move columns back
            return outputs.permute(0, 2, 1)
        else:
            output = output + residual
            return self.layer_norm(output)
