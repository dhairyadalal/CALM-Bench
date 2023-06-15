import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np  
import torch 
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import BatchEncoding
from torch.utils.data import Dataset, DataLoader

from typing import List

class CounterFactualIdentificationDataSet(Dataset):
       
    def __init__(self, encodings: BatchEncoding) -> None:
        super().__init__()
        
        self.encodings = encodings
    
    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) 
                    for key, val in self.encodings.items()}
        return item
    
    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


class CounterFactualIdentificationReader():
        
    def __init__(self, 
            data_loc: str = "data/counterfactual-relation/counterfactual-identification.csv", 
            model_alias: str = "bert-base-uncased"):                
        self.data = pd.read_csv(data_loc)
        self.tokenizer = AutoTokenizer.from_pretrained(model_alias)
        self.model_alias = model_alias
        
    def get_dataset(self, split: str) -> CounterFactualIdentificationDataSet:
        
        df = self.data.query("split == @split")
        
        encoded_inputs = self.tokenizer(df["sentence"].tolist(), 
                                        truncation=True,
                                        max_length = 128,
                                        padding=True,
                                        return_tensors = "pt")
        
        labels = torch.tensor(df["gold_label"].tolist())
        encoded_inputs.update({"labels": labels})
        ds = CounterFactualIdentificationDataSet(encodings=encoded_inputs)
        return ds
       
    def get_dataloader(self, 
                split: str, 
                batch_size:int = 32
            ) -> DataLoader:
        
        shuffle_flag = True if split == "train" else False
        return DataLoader(self.get_dataset(split), 
                    shuffle=shuffle_flag, 
                    batch_size=batch_size
                )
        

class CounterFactualRelationDataset(Dataset):
    
    def __init__(self, 
            data: pd.DataFrame, 
            tokenizer: AutoTokenizer, 
            label2id: dict,
            max_len: int=128
        ):
        
        self.len = len(data)
        self.data = data 
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id    
                    
    def __getitem__(self, index):
        sent = word_tokenize(self.data.iloc[index]["sentence"].strip())
        tagged_sent = self.data.iloc[index]["tagged_sent"].split()
        
        encoding = self.tokenizer(sent, 
                            is_split_into_words=True,
                            return_offsets_mapping=True,
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_len
                        )

        labels = [self.label2id[tag] for tag in tagged_sent]
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        i = 0 
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = labels[i]
                i += 1
        
        item = {k:torch.as_tensor(v) for k,v in encoding.items()}
        item["labels"] = torch.as_tensor(encoded_labels)
        
        return item

    def __len__(self):
        return self.len


class CounterFactualRelationReader():
    
    def __init__(self, 
            data_loc: str = "data/counterfactual-relation/counterfactual-extraction.csv", 
            model_alias: str = "bert-base-uncased"
        ):
        
        super().__init__()
        self.data = pd.read_csv(data_loc)
        
        bad_sents = {
            "roberta" : [204392, 204958, 205197, 205292, 205490],
            "deberta" : [203966, 204009, 204051, 204235, 204302, 204309, 204361, 204392, 204432, 204497, 204623, 204655, 204837, 204958, 205197, 205292, 205490]
        }
        
        # filter out bad sents 
        if "roberta" in model_alias:
            ids = bad_sents["roberta"]
            self.data = self.data.query("sentenceID in @ids").reset_index(drop=True)
        
        if "deberta" in model_alias:
            ids = bad_sents["deberta"]
            self.data = self.data.query("sentenceID in @ids").reset_index(drop=True)

        if "deberta" in model_alias or "roberta" in model_alias:
            self.tokenizer = AutoTokenizer.from_pretrained(model_alias, use_fast=True, add_prefix_space=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_alias, use_fast=True)        
        
        self.model_alias = model_alias      
        self.label2id = {
                            'O': 0,
                            'B-Antecedent' : 1, 
                            'I-Antecedent' : 2,
                            'B-Consequent' : 3,  
                            'I-Consequent' : 4, 
                        }
        
        self.id2label = {v:k for k,v in self.label2id.items()} 
            
    def get_dataset(self, split: str) -> CounterFactualRelationDataset:
        df = self.data.query("split == @split")
        ds = CounterFactualRelationDataset(df, self.tokenizer, self.label2id)
        return ds
    
    def get_dataloader(self, split: str, batch_size:int = 32) -> DataLoader:
        shuffle_flag = True if split == "train" else False
        return DataLoader(self.get_dataset(split), 
                    shuffle=shuffle_flag, 
                    batch_size=batch_size
                )