
import argparse
import torch

import data
from models import models

import json

def get_args():
    parser = argparse.ArgumentParser()
    args.add_argument(
        "--filepath",
        default=-"/n/rush_lab/jc/code/data2text/boxscore-data/rotowire",
        type=str
    )

    args.add_argument("--devid", default=-1, type=int)

    args.add_argument("--bsz", default=32, type=int)
    args.add_argument("--epochs", default=32, type=int)

    args.add_argument("--clip", default=5, type=float)
    args.add_argument("--lr", default=0.01, type=float)
    args.add_argument("--lrd", default=0.1, type=float)
    args.add_argument("--dp", default=0, type=float)
    args.add_argument("--wdp", default=0, type=float)
    args.add_argument("--wd", default=1e-4, type=float)

    args.add_argument("--optim", choices=["Adam", "SGD"])

    # Adam
    args.add_argument("--b1", type=float, default=0.9)
    args.add_argument("--b2", type=float, default=0.999)
    args.add_argument("--eps", type=float, default=1e-8)

    # SGD
    args.add_argument("--mom", type=float, default=0)
    args.add_argument("--dm", type=float, default=0)
    args.add_argument("--nonag", action="store_true", default=False)

    args.add_argument("--model", choices=["Seq2seq", "Attn"], default="Seq2seq")
    args.add_argument("--attn", choices=["dot", "bilinear"], default="dot")
    args.add_argument("--nlayers", default=3, type=int)
    args.add_argument("--nhid", default=512, type=int)
    args.add_argument("--whid", default=512, type=int)
    args.add_argument("--tieweights", action="store_true")
    args.add_argument("--brnn", action="store_true")
    args.add_argument("--inputfeed", action="store_true")

    args.add_argument("--re", default=100, type=int)

    args.add_argument("--seed", default=1111, type=int)
    return parser.parse_args()


args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device(f"cuda:{args.devid}" if args.devid > 0 else "cpu")

# Data
ENT, TYPE, VALUE, TEXT = data.make_fields()
data.RotoDataset.splits(ENT, TYPE, VALUE, TEXT, path=args.filepath)
