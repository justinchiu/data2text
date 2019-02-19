from .base import Lm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from .fns import attn

class CrnnMlm(Lm):
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
        inputfeed = True,
    ):
        super(CrnnMlm, self).__init__()

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
        self.inputfeed = inputfeed

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
            input_size = x_emb_sz
                if not self.inputfeed
                else x_emb_sz + r_emb_sz,
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
            in_features = 3 * r_emb_sz,
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


        # Inference network


    def query(self):
        pass

    """
    emb_x: T x N x Hx
    r: R x N x Hr
    ctxt: T x N x Hc = f(emb_x, r)
    prior
    a: T x N x R ~ Cat(r^T W ctxt)
    unnormalized likelihood
    y: T x N x V = g(ctxt, attn(a, r))
    """

    # unnormalized likelihood
    def sya(self, past, ctxt, ec, a, lenx, lenr):
        # past: T x N x H?
        # index in with z?
        # r: R x N x H
        out = torch.tanh(self.Wc(torch.cat([past, ctxt], dim=-1)))
        return self.proj(self.drop(out))

    def pa0(self, emb_x, s, r, lenx, lenr):
        T, N, _ = emb_x.shape
        R = r.shape[0]
        ea, ec, output = None, None, None
        if not self.inputfeed:
            p_emb = pack(emb_x, lenx)
            rnn_o, s = self.rnn(p_emb, s)
            # rnn_o: T x N x H
            rnn_o, idk = unpack(rnn_o)
            # ea: T x N x R
            # ec: T x N x H
            log_ea, ea, ec = attn(rnn_o, r, lenr)
            output = rnn_o
        else:
            log_ea = []
            ea = []
            ec = []
            output = []
            ect = torch.zeros(N, self.r_emb_sz).to(emb_x.device)
            for t in range(T):
                inp = torch.cat([emb_x[t], ect], dim=-1)
                rnn_o, s = self.rnn(inp.unsqueeze(0), s)
                rnn_o = rnn_o.squeeze(0)
                log_eat, eat, ect = attn(rnn_o, r, lenr)
                log_ea.append(log_eat)
                ea.append(eat)
                ec.append(ect)
                output.append(rnn_o)
            log_ea = torch.stack(log_ea, 0)
            ea = torch.stack(ea, 0)
            ec = torch.stack(ec, 0)
            output = torch.stack(output, 0)
        return log_ea, ea, ec, output

    # posterior
    def pay(self, emb_x, s, r, y, lenx, lenr):
        pass

    def forward(self, x, s, lenx, r, lenr):
        e = self.lute(r[0])
        t = self.lutt(r[1])
        v = self.lutv(r[2])

        # r: R x N x Er
        # Wa r: R x N x H
        r = self.Wa(torch.tanh(torch.cat([e, t, v], dim=-1)))

        emb_x = self.lutx(x)

        log_pa, pa, ec, rnn_o = self.pa0(emb_x, s, r, lenx, lenr)

        # what should we do with pa, ec, and rnn_o?
        # ec is only used for a baseline
        R, N, H = r.shape
        T = x.shape[0]
        K = 1

        # pya later
        dist = torch.distributions.categorical.Categorical(probs=pa)
        # First dimension should be number of samples
        # K x T x N
        a_samples = dist.sample_n(K)
        a_log_p = log_pa.gather(-1, a_samples.permute(1, 2, 0))

        # add baseline
        rnn_o = rnn_o.unsqueeze(0).repeat(K+1, 1, 1, 1)
        ctxt = (r
            .unsqueeze(1).expand(R, T, N, H)
            .gather(
                0,
                a_samples.unsqueeze(-1).expand(K, T, N, H),
            )
        )
        # add baseline
        ctxt = torch.cat(
            [ctxt, ec.unsqueeze(0)],
            0,
        )

        sy = self.sya(rnn_o, ctxt, ec, pa, lenx, lenr)
        return sy

    def _old_forward(self, x, s, lenx, r, lenr):
        emb = self.lutx(x)
        T, N, H = emb.shape

        e = self.lute(r[0])
        t = self.lutt(r[1])
        v = self.lutv(r[2])

        # r: R x N x Er
        # Wa r: R x N x H
        r = self.Wa(torch.tanh(torch.cat([e, t, v], dim=-1)))
        R = r.shape[0]

        if not self.inputfeed:
            p_emb = pack(emb, lenx)
            rnn_o, s = self.rnn(p_emb, s)
            # rnn_o: T x N x H
            rnn_o, idk = unpack(rnn_o)
            # rnn_o: 1->R x T x N x H
            rnn_o = rnn_o.unsqueeze(0).repeat(R, 1, 1, 1)
            # ea: T x N x R
            ea, ec = attn(rnn_o, r, lenr)
            # ctxt: R x 1->T x N x H
            ctxt = r.unsqueeze(1).repeat(1, T, 1, 1)
            out = torch.tanh(self.Wc(torch.cat([rnn_o, ctxt], dim=-1)))
            # too large, use partial enumeration only + reinforce and baseline...?
            # or just reinforce + baseline
            import pdb; pdb.set_trace()
        else:
            outs = []
            ect = torch.zeros(N, self.r_emb_sz).to(emb.device)
            for t in range(T):
                inp = torch.cat([emb[t], ect], dim=-1)
                rnn_o, s = self.rnn(inp.unsqueeze(0), s)
                rnn_o = rnn_o.squeeze(0)
                eat, ect = attn(rnn_o, r, lenr)
                outs.append(torch.cat([rnn_o, ect], dim=-1))
            out = torch.tanh(self.Wc(torch.stack(outs, dim=0)))

        # return unnormalized vocab
        return self.proj(self.drop(out)), s


    def init_state(self, N):
        if self._N != N:
            self._N = N
            self._state = (
                torch.zeros(self.nlayers, N, self.rnn_sz).to(self.lutx.weight.device),
                torch.zeros(self.nlayers, N, self.rnn_sz).to(self.lutx.weight.device),
            )
        return self._state
