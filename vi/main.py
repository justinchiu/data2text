
import argparse
import torch

from torchtext.data import BucketIterator

import data
from models import lm

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

    parser.add_argument("--model", choices=["Seq2seq", "Attn"], default="Seq2seq")
    parser.add_argument("--attn", choices=["dot", "bilinear"], default="dot")
    parser.add_argument("--nlayers", default=3, type=int)
    parser.add_argument("--nhid", default=512, type=int)
    parser.add_argument("--whid", default=512, type=int)
    parser.add_argument("--tieweights", action="store_true")
    parser.add_argument("--brnn", action="store_true")
    parser.add_argument("--inputfeed", action="store_true")

    parser.add_argument("--re", default=100, type=int)

    parser.add_argument("--seed", default=1111, type=int)
    return parser.parse_args()


args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device(f"cuda:{args.devid}" if args.devid > 0 else "cpu")

# Data
ENT, TYPE, VALUE, TEXT = data.make_fields()
train, valid, test = data.RotoDataset.splits(ENT, TYPE, VALUE, TEXT, path=args.filepath)

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train, valid, test),
    batch_size = args.bsz,
    device = device,
    repeat = False,
    sort_within_batch = True,
    #sort_key = already given in dataset?
)

data.build_vocab(ENT, TYPE, VALUE, TEXT, train)


import pdb; pdb.set_trace()

