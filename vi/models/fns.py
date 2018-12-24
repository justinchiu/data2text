import torch
import torch.nn as nn
import torch.nn.functional as F


def attn(x, r, lenr):
    logits = torch.einsum("...nh,rnh->...nr", [x, r])
    # mask
    idxs = torch.arange(0, max(lenr)).to(lenr.device)
    # mask: N x R
    mask = idxs.repeat(len(lenr), 1) >= lenr.unsqueeze(-1)
    if logits.dim() > 2:
        mask = mask.unsqueeze(0)
    logits.masked_fill_(mask, -float("inf"))
    a = F.softmax(logits, dim=-1)
    c = torch.einsum("...nr,rnh->...nh", [a, r])
    return a, c


if __name__ == "__main__":
    # Ensure the broadcasting is correct
    T, N, H, R = 5, 3, 13, 7
    x, r = torch.randn(T, N, H), torch.randn(R, N, H)
    lenr = torch.LongTensor(N).fill_(R)
    z = attn(x, r, lenr)
    zs = torch.stack([attn(x[t], r, lenr) for t in range(T)], dim=0)
    print((z - zs).abs().max())

