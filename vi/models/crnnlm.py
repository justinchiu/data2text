from .base import Lm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class CrnnLm(Lm):
    def __init__(
        self,
        Ve = None,
        Vt = None,
        Vv = None,
        Vx = None,
        r_emb_sz = 256,
        x_emb_sz = 256,
        rnn_sz = 256,
        nlayers = 2,
        dropout = 0.3,
        tieweights = True,
    ):
        super(CrnnLm, self).__init__()

        if tieweights:
            assert(x_emb_sz == rnn_sz)

        self._N = 0

        self.Ve = Ve
        self.Vt = Vt
        self.Vv = Vv
        self.Vx = Vx
        self.r_emb_sz = r_emb_sz
        self.x_emb_sz = x_emb_sz
        self.rnn_sz = rnn_sz
        self.nlayers = nlayers
        self.dropout = dropout

        self.lute = nn.Embedding(
            num_embeddings = len(Ve),
            embedding_dim = r_emb_sz,
            padding_idx = Ve.stoi[self.PAD],
        )
        self.lutt = nn.Embedding(
            num_embeddings = len(Vt),
            embedding_dim = r_emb_sz,
            padding_idx = Vt.stoi[self.PAD],
        )
        self.lutv = nn.Embedding(
            num_embeddings = len(Vv),
            embedding_dim = r_emb_sz,
            padding_idx = Vv.stoi[self.PAD],
        )
        self.lutx = nn.Embedding(
            num_embeddings = len(Vx),
            embedding_dim = x_emb_sz,
            padding_idx = Vx.stoi[self.PAD],
        )
        self.rnn = nn.LSTM(
            input_size = x_emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = False,
            dropout = 0.3,
            bidirectional = False,
        )
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(
            in_features = rnn_sz,
            out_features = len(Vx),
            bias = False,
        )

        # attn projection
        self.Wa = nn.Linear(
            in_features = r_emb_sz,
            out_features = rnn_sz,
            bias = False,
        )

        # context projection
        self.Wc = nn.Linear(
            in_features = r_emb_sz + rnn_sz,
            out_features = rnn_sz,
            bias = False,
        )

        # Tie weights
        if tieweights:
            self.proj.weight = self.lutx.weight


    def forward(self, x, s, lenx, r, lenr):
        emb = self.lutx(x)
        p_emb = pack(emb, lenx)
        # no input feeding for now
        x, s = self.rnn(p_emb, s)
        # x: T x N x H
        x, idk = unpack(x)

        e = self.lute(r[0])
        t = self.lutt(r[1])
        v = self.lutv(r[2])

        # r: R x N x Er, Wa r: R x N x H
        r = self.Wa(e + t + v)

        # dot-prod attn
        # attn1 = (x.permute(1, 0, 2) @ r.permute(1, 2, 0)).permute(1, 0, 2)
        # vs
        logits = torch.einsum("tnh,rnh->tnr", [x, r])
        # mask attn
        idxs = torch.arange(0, max(lenr)).to(lenr.device)
        mask = idxs.repeat(len(lenr), 1) >= lenr.unsqueeze(1)
        logits.masked_fill_(mask.unsqueeze(0), -float("inf"))
        attn = F.softmax(logits, dim=-1)

        context = torch.einsum("tnr,rnh->tnh", [attn, r])

        x = torch.tanh(self.Wc(torch.cat([x, context], dim=-1)))

        return self.proj(self.drop(x)), s


    def init_state(self, N):
        if self._N != N:
            self._N = N
            self._state = (
                torch.zeros(self.nlayers, N, self.rnn_sz).to(self.lutx.weight.device),
                torch.zeros(self.nlayers, N, self.rnn_sz).to(self.lutx.weight.device),
            )
        return self._state
