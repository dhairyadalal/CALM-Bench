from compress_pickle import load
import pandas as pd 
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from transformers.tokenization_utils_base import BatchEncoding
from typing import List, Tuple 

from .causal_dataset import CausalDataSet


class WIQAReader():
    
    def __init__(self, model_alias: str, cache_loc: str = "data/wiqa/wiqa-data.lzma"):        
        super().__init__()
        self.data = load(open(cache_loc, "rb"), compression="lzma")

        self.tokenizer = AutoTokenizer.from_pretrained(model_alias)
        self.model_alias = model_alias
    
    def generate_inputs(self, data: pd.DataFrame) -> List[str]:
        
        inputs = []
        sep = self.tokenizer.sep_token
        for i,r in data.iterrows():
            input_ = r["context"] + sep + r["question_stem"] + sep + "more" + \
                     sep + "less" + sep + "no effect" 
            inputs.append(input_)
            
        return inputs
    
    def get_dataset(self, split: str) -> CausalDataSet:
        
        df = self.data.query("split == @split")
        inputs = self.generate_inputs(df)
        
        encoded_inputs = self.tokenizer(inputs, 
                                        truncation=True,
                                        padding=True,
                                        return_tensors = "pt")
        
        labels = torch.tensor(df["label"].tolist())
        
        encoded_inputs.update({"labels": labels})
        
        ds = CausalDataSet(encodings=encoded_inputs)
        return ds
      
    def get_dataloader(
                self, 
                split: str, 
                batch_size:int = 32, 
                seed: int = 1988
            ) -> DataLoader:
        
        shuffle_flag = True if split == "train" else False
        return DataLoader(
                    self.get_dataset(split), 
                    shuffle=shuffle_flag, 
                    batch_size=batch_size
                )
        
# class WIQAMCDataset(Dataset):
    
#     def __init__(self, encodings: BatchEncoding):
#         self.encodings = encodings
    
#     def __getitem__(self, index: int):
#         return {key: val[index] for key, val in self.encodings.items()}
    
#     def __len__(self):
#         return len(self.encodings["input_ids"])
  


# class WIQAMCReader():
    
#     def __init__(self, 
#             model_alias: str, 
#             cache_loc: str = "data/wiqa/wiqa-data.lzma"):        
#         super().__init__()
#         self.data = load(open(cache_loc, "rb"), compression="lzma")

#         self.tokenizer = AutoTokenizer.from_pretrained(model_alias)
#         self.model_alias = model_alias
    
#     def generate_mc_options(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        
#         first_sents, second_sents = [], []
#         sep = self.tokenizer.sep_token
#         for _, row in df.iterrows():
            
#             question = row["context"] + sep + row["question_stem"] 
#             first_sents.extend([question] * 3)
            
#             second_sents.append("more")
#             second_sents.append("less")
#             second_sents.append("no effect")        
    
#         return first_sents, second_sents
    
    
#     def collate_mc(self, batch):
#         labels = [i.pop("labels") for i in batch]
            
#         batch_size = len(batch)
#         num_choices = 3
        
#         flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in batch]
#         flattened_features = sum(flattened_features, [])
            
#         batch = self.tokenizer.pad(
#             flattened_features,
#             padding=True,
#             pad_to_multiple_of=None,
#             return_tensors="pt",
#         )
        
#         # Un-flatten
#         batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        
#         # Add back labels
#         batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        
#         return batch   
    
#     def get_dataset(self, split: str) -> WIQAMCDataset:
        
#         df = self.data.query("split == @split")
#         f,s = self.generate_mc_options(df)
        
#         tokenized_pairs = self.tokenizer(f,s, truncation=True, padding=True)
#         tokenized_grouped = {k: [v[i:i+3] for i in range(0, len(v), 3)] 
#                                 for k, v in tokenized_pairs.items()}
        
#         labels = torch.tensor(df["label"].tolist())
#         tokenized_grouped.update({"labels": labels})
    
#         ds = WIQAMCDataset(tokenized_grouped)
#         return ds
      
#     def get_dataloader(self, 
#                 split: str, 
#                 batch_size:int = 32, 
#             ) -> DataLoader:
        
#         ds = self.get_dataset(split)
#         shuffle = True if split == "train" else False
        
#         dl = DataLoader(ds,
#                 batch_size=batch_size,
#                 shuffle=shuffle,
#                 collate_fn=self.collate_mc,
#                 num_workers=4, pin_memory=True
#             )
        
#         return dl