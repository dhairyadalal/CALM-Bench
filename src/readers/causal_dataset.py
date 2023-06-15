import torch 
from torch.utils.data import Dataset
from transformers.tokenization_utils import BatchEncoding
from typing import List

class CausalDataSet(Dataset):
    
    def __init__(self, encodings: BatchEncoding) -> None:
        super().__init__()
        
        self.encodings = encodings
    
    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) 
                    for key, val in self.encodings.items()}
        return item
    
    def __len__(self) -> int:
        return len(self.encodings["input_ids"])
