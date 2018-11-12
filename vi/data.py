import torchtext
from torchtext.data import Dataset, Field, Example, Iterator, TabularDataset

import json

def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)

class IeDataset(Dataset):
    def __init__(self, path, text_field, value_field, entity_field, **kwargs):
        pass


    @classmethod
    def splits(
        cls, text_field, label_field, parse_field=None,
        extra_fields={}, root='.data', train='train.json',
        validation='vald.json', test='test.json'
     ):
        pass


    @classmethod
    def iters(cls, batch_size=32, device=0, root=".data", vectors=None, **kwargs):
        pass



if __name__ == "__main__":
    filepath = "/n/rush_lab/jc/code/data2text/boxscore-data/rotowire/train.json"
    json = load_json(filepath)
    TEXT = Field(lower=True, include_lengths=True)
    train = TabularDataset(filepath, "jsonlist", {"summary": ("text", TEXT)})
    valid = train
    TEXT.build_vocab(train.text)
    train_iter, valid_iter = torchtext.data.BucketIterator.splits((train, valid), batch_size=3)
    import pdb; pdb.set_trace()
