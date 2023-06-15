import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import List, Tuple
import torch 

from src.readers.causal_dataset import CausalDataSet

class ECAREdatset(Dataset):
    
    def __init__(self, encodings: BatchEncoding):
        self.encodings = encodings
    
    def __getitem__(self, index: int):
        return {key: val[index] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings["input_ids"])

class ECAREreader():
    
    def __init__(self, model_alias: str, data_loc: str="data/e-care/e-care.csv"):
        
        self.data = pd.read_csv(data_loc)
        self.tokenizer = AutoTokenizer.from_pretrained(model_alias)
        self.model_alias = model_alias
        
    def generate_pairs(self, df) -> Tuple[List[str], List[str]]:
        first_sents = []
        second_sents = []
        
        for _, row in df.iterrows():
            if row["ask-for"] == "cause":
                first_sents.append(row["hypothesis1"])
                first_sents.append(row["hypothesis2"])
                
                second_sents.append(row.premise)
                second_sents.append(row.premise)
            
            else:
                first_sents.append(row.premise)
                first_sents.append(row.premise)
                
                second_sents.append(row["hypothesis1"])
                second_sents.append(row["hypothesis2"])
        
        
        return first_sents, second_sents
    
    
    def get_dataset(self, split: str) -> Dataset:
        
        df = self.data.query("split == @split")        
        f,s = self.generate_pairs(df)
        tokenized_pairs = self.tokenizer(f, s, truncation=True, max_length=35)        
        tokenized_grouped = {k: [v[i:i+2] for i in range(0, len(v), 2)] 
                                for k, v in tokenized_pairs.items()}
        
        labels = torch.tensor(df["label"].tolist())
        tokenized_grouped.update({"labels": labels})
        ds = ECAREdatset(tokenized_grouped)
        return  ds 
    
    
    def get_dataloader(self, split: str, batch_size=64) -> DataLoader:
        
        ds = self.get_dataset(split)
        shuffle = True if split == "train" else False
        dl = DataLoader(
                    ds, 
                    batch_size=batch_size, 
                    shuffle=shuffle, 
                    collate_fn=self.collate_copa_mc,
                    num_workers=4
                )  
        return dl 
    
    def collate_copa_mc(self, batch):
    
        # Extract out labels
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
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
    