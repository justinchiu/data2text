from torchtext import data

text = data.Field()
train = data.TabularDataset(
    path = "boxscore-data/rotowire/train.json",
    format = "jsonlist",
    fields = {
        "summary": ("summary", text), # [STR]
        "home_name": ("home_name", data.Field()), # STR
        "home_city": ("home_city", data.Field()), # STR
        "home_line": ("home_line", data.Field()), # JSON
        "vis_name": ("vis_name", data.Field()), # STR
        "vis_city": ("vis_city", data.Field()), # STR
        "vis_line": ("vis_line", data.Field()), # JSON
        "day": ("day", data.Field()), # STR
        "box_score": ("box_score", data.Field()), # JSON
    },
)
text.build_vocab(
    train,
    #max_size = 50k,
    #min_freq = 3,
)
print(len(text.vocab))
import pdb; pdb.set_trace()
