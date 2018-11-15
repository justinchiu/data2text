import torchtext
from torchtext.data import Dataset, Field, Example, Iterator, TabularDataset

import io
import os
import json


def nested_items(name, x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from nested_items(f"{name}_{k}", v)
    else:
        yield (name, x)


class IeExample(Example):
    @classmethod
    def fromJson(cls, data, fields):
        exs = []
        for x in json.load(data):
            ex = cls()
            box_score = x["box_score"]
            id2name = box_score["PLAYER_NAME"]

            for k, v in nested_items("", x):
                print(k)
                import pdb; pdb.set_trace()
            for key in ["home_name", "home_city", "vis_name", "vis_city", "day"]:
                setattr(ex, name, field.preprocess(x[key]))
                pass
            import pdb; pdb.set_trace()
            # lines
            # boxscore
            # summary
            for key, vals in fields.items():
                #import pdb; pdb.set_trace()
                if key not in x:
                    raise ValueError("Specified key {} was not found in "
                                     "the input data".format(key))
                if vals is not None:
                    if not isinstance(vals, list):
                        vals = [vals]
                    for val in vals:
                        name, field = val
                        setattr(ex, name, field.preprocess(x[key]))
            exs.append(ex)
        return exs

class IeDataset(Dataset):
    def __init__(
        self, path, text_field, entity_field, type_field, value_field,
        skip_header=False, **kwargs):

        fields = {
            "summary": ("text", TEXT),
        }

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            examples = IeExample.fromJson(f, fields)
            import pdb; pdb.set_trace()

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(IeDataset, self).__init__(examples, fields, **kwargs)


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
    ENT = Field(lower=True)
    TYPE = Field(lower=True)
    VALUE = Field(lower=True)
    TEXT = Field(lower=True, include_lengths=True)

    #train = TabularDataset(filepath, "jsonlist", {"summary": ("text", TEXT)})
    #valid = train # hack
    #TEXT.build_vocab(train.text)
    #train_iter, valid_iter = torchtext.data.BucketIterator.splits((train, valid), batch_size=3)

    train = IeDataset(filepath, TEXT, ENT, TYPE, VALUE)
    import pdb; pdb.set_trace()
