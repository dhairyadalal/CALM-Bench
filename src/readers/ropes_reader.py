from compress_pickle import load
import pandas as pd 
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
import re 

class RopesDataset(Dataset):
    
    def __init__(self, encodings: BatchEncoding):
        self.encodings = encodings
    
    def __getitem__(self, index: int):
        return {key: val[index] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)
    

class RopesReader():
    
    def __init__(self, 
                 cache_loc: str = "data/ropes/ropes-data.lzma",
                 model_alias: str = "bert-base-uncased"):
    
        self.data = load(open(cache_loc, "rb"), compression="lzma")
        self.tokenizer = AutoTokenizer.from_pretrained(model_alias)
    
    
    def _add_position_annotation(self,
                            encodings: BatchEncoding,
                            data: pd.DataFrame,
                            inputs: List[str]
                        ) -> BatchEncoding:
        
        start_positions = []
        end_positions = []
          
        for i in range(len(encodings["input_ids"])):
            
            try:
                context = inputs[i].lower() if self.tokenizer.do_lower_case else inputs[i]
                answer = data.iloc[i].answer_label.lower() if self.tokenizer.do_lower_case else data.iloc[i].answer_label
                
            except:
                context = inputs[i]
                answer = data.iloc[i].answer_label
    
            es = 0
            ee = 0
    
            pos = re.search(r"\b"+answer+r"\b", context)
     
            if pos:
                e = pos.end() 
            
                es = encodings.char_to_token(i, pos.start())
                ee = encodings.char_to_token(i, pos.end())
                
                if ee is None:
                    for ix in range(3):
                        ee = encodings.char_to_token(i, e + ix)
                        if self.tokenizer.decode(encodings["input_ids"][i][es:ee]) == answer:
                            break
                        
            else: 
                sos = re.search(answer, context)
                if sos: 
                    es = encodings.char_to_token(i, sos.start())
                    ee = es
                    
                    for ix in range(3):
                        ts = self.tokenizer.decode(encodings["input_ids"][i][es:ee+ix])
                        if answer in ts:
                            break            
            ts = self.tokenizer.decode(encodings["input_ids"][i][es: ee])
            
            start_positions.append(es if es is not None else 0)
            end_positions.append(ee if ee is not None else 0)
            
        encodings.update(
                {
                    "start_positions": torch.tensor(start_positions),
                    "end_positions": torch.tensor(end_positions)
                })
        
        return encodings
        
    def generate_encodings(self, 
                           data: pd, 
                           use_background: bool,
                           add_positions: bool
                        ) -> BatchEncoding:
        """ Method generate tokenized input and sub-token aligned positions 
            annotations for model training. 
        """
        
        inputs = []     
        for _, r in data.iterrows():
            
            if use_background:
                inputs.append(
                    r.question + " " + self.tokenizer.sep_token + " " + \
                    r.situation + " " + r.background
                )
            else:
                inputs.append(
                    r.question + " " + self.tokenizer.sep_token + " " + \
                    r.situation
                )
        
        encodings = self.tokenizer(inputs, 
                                truncation=True, 
                                padding=True,
                                return_tensors = "pt",
                                max_length=235
                            )
        
        if add_positions:
            encodings = self._add_position_annotation(encodings, data, inputs)
        return encodings
               
    
    def get_dataset(self, split: str, use_background: bool=True) -> RopesDataset:
        
        data = self.data.query("split == @split")
    
        if split != "test":
            encodings = self.generate_encodings(data, use_background=use_background, add_positions=True)
        else:
            encodings = self.generate_encodings(data, use_background=use_background, add_positions=False)

        ds = RopesDataset(encodings)
        
        return ds 
    
    def get_dataloader(self, 
                    split: str, 
                    use_background: bool=True,
                    batch_size: int = 24
                ) -> DataLoader:
        
        ds = self.get_dataset(split=split, use_background=use_background)
        shuffle = True if split == "train" else False
        
        dataloader = DataLoader(
                        dataset=ds, 
                        shuffle=shuffle, 
                        batch_size=batch_size,
                        num_workers=4
                    )
        return dataloader
    