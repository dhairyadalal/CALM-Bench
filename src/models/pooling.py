import torch 
import torch.nn as nn


class BertPooler(nn.Module):
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):        
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class DeBertaPooler(nn.Module):
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=.1)
        self.activation = nn.GELU()
        
    def forward(self, hidden_states):        
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
 
