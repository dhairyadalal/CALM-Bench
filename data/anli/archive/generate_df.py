#%%
import json
import pandas as pd 

train_rows = []
with open('train.jsonl', 'r') as json_file:
    json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        train_rows.append(result)

with open("train-labels.lst", "r") as f:
    train_labels = [l.strip() for l in f.readlines()]
    
train_df = pd.DataFrame(train_rows)
train_df["answer"] = train_labels
train_df["split"] = "train"

dev_rows = []
with open('dev.jsonl', 'r') as json_file:
    json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        dev_rows.append(result)

with open("dev-labels.lst", "r") as f:
    dev_labels = [l.strip() for l in f.readlines()]
    
dev_df = pd.DataFrame(dev_rows)
dev_df["answer"] = dev_labels
dev_df["split"] = "val"

#%%
train_df["label"] = train_df["answer"].apply(lambda x: int(x)-1)
dev_df["label"] = dev_df["answer"].apply(lambda x: int(x)-1)

#%%
test_rows = []
with open('alphanli-test/anli.jsonl', 'r') as json_file:
    json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        test_rows.append(result)

    
test_df = pd.DataFrame(test_rows)
test_df["label"] = None
test_df["split"] = "test"

#%%
final = pd.concat([train_df, dev_df, test_df])
final.to_csv("anli_dataset.csv")
#%%
from compress_pickle import dump, load
with open("../anli-data.lzma", "wb") as f:
    dump(final, f, compression="lzma")

# %%
test = load(open("../anli-data.lzma", "rb"), compression="lzma")

# %%
