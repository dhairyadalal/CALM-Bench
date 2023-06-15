from compress_pickle import load
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import List, Tuple


class CosmosQaDataset(Dataset):
    
    def __init__(self, encodings: BatchEncoding):
        self.encodings = encodings
    
    def __getitem__(self, index: int):
        return {key: val[index] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings["input_ids"])
  

class CosmosQaMCReader():
    
    def __init__(self,
            cache_loc: str = "data/cosmosqa/cosmosqa-data.lzma",
            model_alias: str = "bert-base-uncased"
        ):
        self.data = load(open(cache_loc, "rb"), compression="lzma")
        self.tokenizer = AutoTokenizer.from_pretrained(model_alias)
    

    def generate_mc_options(self, 
                        df: pd.DataFrame
                    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        
        first_sents = []
        second_sents = []
        
        for _, row in df.iterrows():
            question = row.question + " " + row.context
            first_sents.extend([question] * 4)
            
            second_sents.append(row["answer0"])
            second_sents.append(row["answer1"])
            second_sents.append(row["answer2"])
            second_sents.append(row["answer3"])
            
        return first_sents, second_sents
    
    def collate_mc(self, batch):
        has_labels = True if "labels" in batch[0] else False   
        
        if has_labels:     
            labels = [i.pop("labels") for i in batch]
        
        batch_size = len(batch)
        num_choices = 4
        
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
    
    
    def get_dataset(self, split: str) -> CosmosQaDataset:
        
        df = self.data.query("split == @split")
        f,s = self.generate_mc_options(df)
        
        tokenized_pairs = self.tokenizer(f,s,truncation=True, max_length=155)
        tokenized_grouped = {k: [v[i:i+4] for i in range(0, len(v), 4)] 
                                for k, v in tokenized_pairs.items()}
        
        if split != "test":
            labels = torch.tensor(df["label"].tolist())
            tokenized_grouped.update({"labels": labels})
        
        ds = CosmosQaDataset(tokenized_grouped)
        return ds
    
    def get_dataloader(self, split: str, batch_size=24) -> DataLoader:
        
        ds = self.get_dataset(split)
        shuffle = True if split == "train" else False
        
        dl = DataLoader(ds,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=self.collate_mc,
                num_workers=1
            )
        
        return dl
    