
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn.utils import clip_grad_norm_ as clip_
from torch.distributions import kl_divergence
from torch.distributions.categorical import Categorical

from .base import Lm

class Lvm(Lm):
    def _loop(self, iter, optimizer=None, clip=0, learn=False, re=None):
        context = torch.enable_grad if learn else torch.no_grad

        cum_loss = 0
        cum_ntokens = 0
        cum_rx = 0
        cum_kl = 0
        batch_loss = 0
        batch_ntokens = 0
        states = None
        with context():
            titer = tqdm(iter) if learn else iter
            for i, batch in enumerate(titer):
                if learn:
                    optimizer.zero_grad()
                text, lens = batch.text
                x = text[:-1]
                y = text[1:]
                lens = lens - 1

                e, lene = batch.entities
                t, lent = batch.types
                v, lenv = batch.values
                #rlen, N = e.shape
                #r = torch.stack([e, t, v], dim=-1)
                r = [e, t, v]
                assert (lene == lent).all()
                lenr = lene

                # should i include <eos> in ppl? no, should not.
                mask = y.ne(1) #* y.ne(3)
                nwords = mask.sum()
                # assert nwords == lens.sum()
                T, N = y.shape
                R = e.shape[0]
                #if states is None:
                states = self.init_state(N)
                logits, _, sampled_log_pa, log_pa, log_qay = self(x, states, lens, r, lenr, y)

                nll = self.loss(logits, y)
                B = nll[-1]
                reward = (nll[:-1] - B.unsqueeze(0)).detach() * sampled_log_pa
                reward = reward[mask.unsqueeze(0).expand_as(reward)].sum()
                nll = nll[:-1].sum(0)[mask].sum()


                # giving nans because of masking, sigh
                #"""
                qa = log_qa.exp()
                qa[log_qay == float("-inf")] = 0
                kl0 = (log_qay.exp() * (log_qay - log_pa))
                kl0 = log_qay - log_pa
                kl0[log_qay == float("-inf")] = 0
                kl = kl0.sum()
                #import pdb; pdb.set_trace()
                #"""
                """
                kl = 0
                for i, l in enumerate(lenr.tolist()):
                    p = Categorical(logits=log_pa[:,i,:l])
                    q = Categorical(logits=log_qay[:,i,:l])
                    kl0 = kl_divergence(q, p).sum()
                    kl += kl0
                    """


                #p = Categorical(logits=log_pa[mask])
                #q = Categorical(logits=log_qay[mask])
                #kl = kl_divergence(q, p).sum()

                nelbo = nll + kl
                if learn:
                    (nelbo - reward).div(nwords.item()).backward()
                    if clip > 0:
                        gnorm = clip_(self.parameters(), clip)
                        #for param in self.rnn_parameters():
                            #gnorm = clip_(param, clip)
                    optimizer.step()
                cum_loss += nelbo.item()
                cum_ntokens += nwords.item()
                batch_loss += nelbo.item()
                batch_ntokens += nwords.item()
                if re is not None and i % re == -1 % re:
                    titer.set_postfix(loss = batch_loss / batch_ntokens, gnorm = gnorm)
                    batch_loss = 0
                    batch_ntokens = 0
        return cum_loss, cum_ntokens


    def loss(self, logits, y):
        K = logits.shape[0]
        T, N = y.shape
        y = y.unsqueeze(0).repeat(K, 1, 1)
        yflat = y.view(-1, 1)
        return -(F.log_softmax(logits, dim=-1)
            .view(K*T*N, -1)
            .gather(-1, yflat)#[(yflat != 1)] # don't include 3 as well?
            .view(K,T,N)
        )

