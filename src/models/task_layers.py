import torch
import torch.nn as nn 
from .pooling import DeBertaPooler


class aNLITaskLayer(nn.Module):
    
    def __init__(self, hidden_size: int, dropout_prob: float = .1):
        super().__init__()
        
        self.pooler = DeBertaPooler(hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        pooled_state = self.dropout(self.pooler(hidden_states))
        logits = self.classifier(pooled_state)
        return {"logits": logits}
    

class CausalIdentificationTaskLayer(nn.Module):
  
    def __init__(self, hidden_size: int, dropout_prob: float = .1):
        super().__init__()
        
        self.pooler = DeBertaPooler(hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, 2)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        out = self.dropout(self.pooler(hidden_states))
        logits = self.classifier(out)       
        return {"logits": logits}
    

class CausalRelationTaskLayer(nn.Module):
    
    def __init__(self, num_tags: int, hidden_size: int, dropout_prob: float = .1):
        super().__init__()    
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_tags)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        out = self.dropout(hidden_states)
        logits = self.classifier(out)       
        return {"logits": logits}
    

class CounterFactualRelationTaskLayer(nn.Module):
    
    def __init__(self, num_tags: int, hidden_size: int, dropout_prob: float = .1):
        super().__init__()    
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_tags)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        out = self.dropout(hidden_states)
        logits = self.classifier(out)       
        return {"logits": logits}
    

class CFIdentificationTaskLayer(nn.Module):
  
    def __init__(self, hidden_size: int, dropout_prob: float = .1):
        super().__init__()
        
        self.pooler = DeBertaPooler(hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, 2)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        out = self.dropout(self.pooler(hidden_states))
        logits = self.classifier(out)       
        return {"logits": logits}
    

class CopaTaskLayer(nn.Module):
    
    def __init__(self, hidden_size: int, dropout_prob: float = .1):
        super().__init__()
        
        self.pooler = DeBertaPooler(hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        pooled_state = self.dropout(self.pooler(hidden_states))
        logits = self.classifier(pooled_state)
        return {"logits": logits}


class ECARETaskLayer(nn.Module):
    
    def __init__(self, hidden_size: int, dropout_prob: float = .1):
        super().__init__()
        
        self.pooler = DeBertaPooler(hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        pooled_state = self.dropout(self.pooler(hidden_states))
        logits = self.classifier(pooled_state)
        return {"logits": logits}


class CosmosQaTaskLayer(nn.Module):
    
    def __init__(self, hidden_size: int, dropout_prob: float = .1):
        super().__init__()
        
        self.pooler = DeBertaPooler(hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        pooled_state = self.dropout(self.pooler(hidden_states))
        logits = self.classifier(pooled_state)
        return {"logits": logits}

class GeneralMCTaskLayer(nn.Module):
    
    def __init__(self, hidden_size: int, dropout_prob: float = .1):
        super().__init__()
        
        self.pooler = DeBertaPooler(hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        pooled_state = self.dropout(self.pooler(hidden_states))
        logits = self.classifier(pooled_state)
        return {"logits": logits}

class RopeSimpleTaskLayer(nn.Module):
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.qa_outputs = nn.Linear(hidden_size, 2)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        return {"start_logits": start_logits, "end_logits": end_logits}


class WIQAMCTaskLayer(nn.Module):
    
    def __init__(self, hidden_size: int, dropout_prob: float = .1):
        super().__init__()
        
        self.pooler = DeBertaPooler(hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        pooled_state = self.dropout(self.pooler(hidden_states))
        logits = self.classifier(pooled_state)
        return {"logits": logits}


class WiqaTaskLayer(nn.Module):
  
    def __init__(self, hidden_size: int, dropout_prob: float = .1):
        super().__init__()
        
        self.pooler = DeBertaPooler(hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size, 3)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:       
        out = self.dropout(self.pooler(hidden_states))
        logits = self.classifier(out)       
        return {"logits": logits}