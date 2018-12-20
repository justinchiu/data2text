from .base import Lm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class CrnnLm(Lm):
    def __init__(
        self,
        Vr = None,
        Vx = None,
        r_emb_sz = 256,
        x_emb_sz = 256,
        rnn_sz = 256,
        nlayers = 2,
        dropout = 0.3,
        tie_weights = True,
    ):
        super(CrnnLm, self).__init__()

        if tie_weights:
            assert(emb_sz == rnn_sz)

        self._N = 0

        self.Vr = Vr
        self.Vx = Vx
        self.r_emb_sz = r_emb_sz
        self.x_emb_sz = x_emb_sz
        self.rnn_sz = rnn_sz
        self.nlayers = nlayers
        self.dropout = dropout

        self.lutr = nn.Embedding(
            num_embeddings = len(Vr),
            embedding_dim = r_emb_sz,
            padding_idx = Vr.stoi[self.PAD],
        )
        self.lutx = nn.Embedding(
            num_embeddings = len(Vx),
            embedding_dim = x_emb_sz,
            padding_idx = Vx.stoi[self.PAD],
        )
        self.rnn = nn.LSTM(
            input_size = emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = False,
            dropout = 0.3,
            bidirectional = False,
        )
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(
            in_features = rnn_sz,
            out_features = len(V),
            bias = False,
        )

        # Tie weights
        self.proj.weight = self.lut.weight


    def forward(self, x, s, lens, r):
        emb = self.lutx(x)
        p_emb = pack(emb, lens)
        x, s = self.rnn(p_emb, s)

        records = self.lutr(r)
        # attn
        import pdb; pdb.set_trace()
        return self.proj(self.drop(unpack(x)[0])), s


    def init_state(self, N):
        if self._N != N:
            self._N = N
            self._state = (
                torch.zeros(self.nlayers, N, self.rnn_sz).to(self.lut.weight.device),
                torch.zeros(self.nlayers, N, self.rnn_sz).to(self.lut.weight.device),
            )
        return self._state
