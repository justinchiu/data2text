
import argparse

import torch
import torch.optim as optim

from torchtext.data import BucketIterator

import data
from models.rnnlm import RnnLm

import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        default="../boxscore-data/rotowire",
        type=str
    )

    parser.add_argument("--devid", default=-1, type=int)

    parser.add_argument("--bsz", default=32, type=int)
    parser.add_argument("--epochs", default=32, type=int)

    parser.add_argument("--clip", default=5, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--lrd", default=0.1, type=float)
    parser.add_argument("--dp", default=0, type=float)
    parser.add_argument("--wdp", default=0, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)

    parser.add_argument("--optim", choices=["Adam", "SGD"])

    # Adam
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # SGD
    parser.add_argument("--mom", type=float, default=0)
    parser.add_argument("--dm", type=float, default=0)
    parser.add_argument("--nonag", action="store_true", default=False)

    # Model
    parser.add_argument(
        "--model",
        choices=["LM", "CLM", "HMM", "HSMM"],
        default="LM"
    )

    parser.add_argument("--attn", choices=["dot", "bilinear"], default="dot")

    parser.add_argument("--nlayers", default=2, type=int)
    parser.add_argument("--emb-sz", default=256, type=int)
    parser.add_argument("--rnn-sz", default=256, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--tieweights", action="store_true")

    parser.add_argument("--brnn", action="store_true")
    parser.add_argument("--inputfeed", action="store_true")

    parser.add_argument("--re", default=100, type=int)

    parser.add_argument("--seed", default=1111, type=int)
    return parser.parse_args()


args = get_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device(f"cuda:{args.devid}" if args.devid >= 0 else "cpu")

# Data
ENT, TYPE, VALUE, TEXT = data.make_fields()
train, valid, test = data.RotoDataset.splits(ENT, TYPE, VALUE, TEXT, path=args.filepath)

data.build_vocab(ENT, TYPE, VALUE, TEXT, train)

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train, valid, test),
    batch_size = args.bsz,
    device = device,
    repeat = False,
    sort_within_batch = True,
    #sort_key = already given in dataset?
)

# Model
model = RnnLm(
    V       = TEXT.vocab,
    emb_sz  = args.emb_sz,
    rnn_sz  = args.rnn_sz,
    nlayers = args.nlayers,
    dropout = args.dropout,
).to(device)
print(model)

params = list(model.parameters())

optimizer = optim.Adam(
    params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))


for e in range(args.epochs):
    # Train
    model.train_epoch(
        iter      = train_iter,
        clip      = args.clip,
        re        = args.re,
        optimizer = optimizer,
    )

    # Validate
    model.validate(valid_iter)

import pdb; pdb.set_trace()

