from compress_pickle import load
import torch
import pandas as pd 
from typing import List, Tuple
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class aNLIMCDataset(Dataset):
    
    def __init__(self, encodings: dict):  
        self.encodings = encodings
        
    def __len__(self) -> int:
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx: int) -> dict:
        e = {k: v[idx] for k,v in self.encodings.items()}
        return e 


class aNLIMCreader():
       
    def __init__(self, 
            model_alias: str, 
            cache_loc: str = "data/anli/anli-data.lzma"):
        self.data = load(open(cache_loc, "rb"), compression="lzma")

        self.tokenizer = AutoTokenizer.from_pretrained(model_alias)
        self.model_alias = model_alias
        
    def generate_pairs(self, df) -> Tuple[List[str], List[str]]:
        first_sents = []
        second_sents = []
        
        for _, row in df.iterrows():
            first_sents.append(row.hyp1)
            first_sents.append(row.hyp2)
            
            second_sents.append(row.obs1 + " " + row.obs2)
            second_sents.append(row.obs1 + " " + row.obs2)

        return first_sents, second_sents
    
    def collate_mc(self, batch: List[dict]) -> dict:
        has_labels = True if "labels" in batch[0] else False
        
        if has_labels:
            labels = [i.pop("labels") for i in batch]           
        
        batch_size = len(batch)
        num_choices = 2
        
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in batch]
        flattened_features = sum(flattened_features, [])
            
        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        
        # Add back labels
        if has_labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
        
    
    def get_dataset(self, split: str) -> aNLIMCDataset:
        
        df = self.data.query("split == @split")
        f,s = self.generate_pairs(df)
        tokenized_pairs = self.tokenizer(f,s, truncation=True, max_length=80)
        tokenized_grouped = {k: [v[i:i+2] for i in range(0, len(v), 2)] 
                                for k, v in tokenized_pairs.items()}
        
        if split != "test":
            tokenized_grouped.update({"labels": df["label"].tolist()}
                                     )
        ds = aNLIMCDataset(tokenized_grouped)
        return ds
    
    def get_dataloader(self, split: str, batch_size=24) -> DataLoader:
        
        ds = self.get_dataset(split)
        shuffle = True if split == "train" else False
        
        dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=self.collate_mc,
                num_workers=1
        )
        
        return dl
    