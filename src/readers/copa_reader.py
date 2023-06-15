import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from transformers import AutoTokenizer
from typing import List, Tuple
import torch 

from .causal_dataset import CausalDataSet

# class COPAReader():
    
#     def __init__(
#             self, 
#             data: pd.DataFrame,
#             model_alias: str = "distilbert-base-uncased"
#         ):
#         super().__init__()
#         self.data = data
#         self.tokenizer = AutoTokenizer.from_pretrained(model_alias)
#         self.sep = self.tokenizer.sep_token
#         self.model_alias = model_alias
        
#         # Generate inputs
#         self.data["input"] = self.data.apply(lambda x: self.create_input(x), axis=1) 
            
#     def create_input(self, row):
#         cause_q = "What was the cause of this?"
#         effect_q = "What happened as a result?"
#         question_str = cause_q if row.question == "cause" else effect_q
#         return row.premise + self.sep + question_str + self.sep + row.choice1 + self.sep + row.choice2 
     
        
#     def get_dataset(self, split: str) -> Dataset:
#         df = self.data.query("split == @split")
#         encoded_input = self.tokenizer(df["input"].tolist(), 
#                                        truncation=True,
#                                        padding=True,
#                                        return_tensors = "pt")
        
#         labels = df["label"].tolist()
        
#         return CausalDataSet(encodings=encoded_input, labels=labels)
    
#     def get_dataloader(
#                     self, 
#                     split: str, 
#                     batch_size:int = 32, 
#                     seed: int = 1988
#                 ) -> DataLoader:
#         shuffle_flag = True if split == "train" else False
#         return DataLoader(
#                     self.get_dataset(split), 
#                     shuffle=shuffle_flag, 
#                     batch_size=batch_size,
#                     num_workers=4
#                 )

class CopaMCdatset(Dataset):
    
    def __init__(self, encodings: dict, labels: list):
        
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> dict:        
        e = {k: v[idx] for k, v in self.encodings.items()}
        e["labels"] = self.labels[idx]
        return e

class CopaMCreader():
    
    def __init__(self, model_alias: str, data_loc: str="data/copa/copa-data.csv"):
        
        self.data = pd.read_csv(data_loc)
        self.tokenizer = AutoTokenizer.from_pretrained(model_alias)
        self.model_alias = model_alias
        
    def generate_pairs(self, df) -> Tuple[List[str], List[str]]:
        first_sents = []
        second_sents = []
        
        for _, row in df.iterrows():
            if row.question == "cause":
                first_sents.append(row.choice1)
                first_sents.append(row.choice2)
                
                second_sents.append(row.premise)
                second_sents.append(row.premise)
            
            else:
                first_sents.append(row.premise)
                first_sents.append(row.premise)
                
                second_sents.append(row.choice1)
                second_sents.append(row.choice2)
        
        
        return first_sents, second_sents
    
    
    def get_dataset(self, split: str) -> Dataset:
        
        df = self.data.query("split == @split")        
        f,s = self.generate_pairs(df)
        tokenized_pairs = self.tokenizer(f, s, truncation=True)        
        tokenized_grouped = {k: [v[i:i+2] for i in range(0, len(v), 2)] 
                                for k, v in tokenized_pairs.items()}
        ds = CopaMCdatset(tokenized_grouped, df["label"].tolist())
        return  ds 
    
    
    def get_dataloader(self, 
                split: str, 
                batch_size=24,
                num_workers=4) -> DataLoader:
                
        ds = self.get_dataset(split)
        shuffle = True if split == "train" else False

        dl = DataLoader(ds, shuffle=shuffle, batch_size=batch_size, collate_fn=self.collate_copa_mc, num_workers=1) 
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
            max_length=25,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
    