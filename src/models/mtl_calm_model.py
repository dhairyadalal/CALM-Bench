import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import AutoModel
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from sklearn.metrics import accuracy_score
from seqeval.metrics import accuracy_score as seqeval_accuracy

from .task_layers import aNLITaskLayer, \
                        CausalIdentificationTaskLayer, \
                        CausalRelationTaskLayer, \
                        CounterFactualRelationTaskLayer, \
                        CFIdentificationTaskLayer, \
                        CopaTaskLayer, \
                        CosmosQaTaskLayer, \
                        ECARETaskLayer, \
                        RopeSimpleTaskLayer, \
                        WiqaTaskLayer, \
                        GeneralMCTaskLayer
    
    
class MTLCausalityModel(pl.LightningModule):
    
    def __init__(self, config: dict):
        
        super().__init__()
        self.config = config 
        self.learning_rate = config["learning_rate"]
        self.predict_task = None 
        self.loss_strategy = config["loss_strategy"]
        self.update_seed(config["seed"])
        
        self.train_dl_map = {}
        
        # ======================== Model Details ==============================
        self.lm = AutoModel.from_pretrained(config["model_alias"])
        hidden_size = self.lm.config.hidden_size
        dropout_prob = self.lm.config.hidden_dropout_prob
        
        self.task_heads = nn.ModuleDict({
                                "anli": aNLITaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob),
                                "causal_identification": CausalIdentificationTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob),
                                "causal_relation": CausalRelationTaskLayer(num_tags=5, hidden_size=hidden_size, dropout_prob=dropout_prob)   ,
                                "counterfactual_identification": CFIdentificationTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob),
                                "counterfactual_relation": CounterFactualRelationTaskLayer(num_tags=5, hidden_size=hidden_size, dropout_prob=dropout_prob),
                                "copa": CopaTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob),
                                "ecare": ECARETaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob),
                                "cosmosqa": CosmosQaTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob),
                                "ropes": RopeSimpleTaskLayer(hidden_size=hidden_size),
                                "wiqa": WiqaTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob)
                            })
            
    def set_train_dataloaders(self, dl_map: dict):
        self.train_dl_map = dl_map
    
    def update_seed(self, seed: int):
        pl.seed_everything(seed, workers=True)
    
    def update_lr(self, lr: float):
        self.learning_rate = lr
                                    
    def set_predict_task(self, task: str):
        self.predict_task = task 
    
    def forward(self, task, input_ids, attention_mask):
        
        if task in ["copa", "anli", "cosmosqa", "ecare"]:         
            num_choices = input_ids.shape[1]
            input_ids = input_ids.view(-1, input_ids.size(-1)) 
            mask = attention_mask.view(-1, attention_mask.size(-1))
            
            hidden_states = self.lm.forward(input_ids, attention_mask=mask)["last_hidden_state"]
            
            logits = self.task_heads[task](hidden_states)["logits"]
            logits = logits.view(-1, num_choices)
            logits = {"logits": logits}    
            
        else:
            hidden_states = self.lm(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
            logits = self.task_heads[task](hidden_states)
      
        return logits
    
    def training_step(self, batch, batch_idx):
        
        train_sum_loss = 0
        task_loss_list = []
        for task, b in batch.items():     
               
            if task == "ropes":
                start_positions = b["start_positions"]
                end_positions = b["end_positions"]
            
                logits_dict = self.forward(task=task, input_ids = b["input_ids"], attention_mask = b["attention_mask"])
                
                start_logits = logits_dict["start_logits"]
                end_logits = logits_dict["end_logits"]
                
                task_loss = None            
                if start_positions is not None and end_positions is not None:
                    
                    if len(start_positions.size()) > 1:
                        start_positions = start_positions.squeeze(-1)
                    
                    if len(end_positions.size()) > 1:
                        end_positions = end_positions.squeeze(-1)
                    
                    ignored_index = start_logits.size(1)
                    start_positions = start_positions.clamp(0, ignored_index)
                    end_positions = end_positions.clamp(0, ignored_index)
                    
                    loss_func = nn.CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_func(start_logits, start_positions)
                    end_loss = loss_func(end_logits, end_positions)
                    task_loss = (start_loss + end_loss) / 2  
            
            elif task in ["causal_relation", "counterfactual_relation"]:
                logits = self.forward(task = task, input_ids = b["input_ids"], attention_mask = b["attention_mask"])["logits"]   
                task_loss = F.cross_entropy(logits.view(-1, 5), b["labels"].view(-1))         
            
            else:
                logits = self.forward(task=task, input_ids=b["input_ids"], attention_mask=b["attention_mask"])["logits"]
                task_loss = F.cross_entropy(logits, b["labels"])
            
            # Add losses together and also add it to loss tracker
            train_sum_loss += task_loss
            task_loss_list.append(task_loss)
        
        if self.loss_strategy == "mean":
            train_mean_loss = torch.mean(torch.stack(task_loss_list))
            self.log("train_loss", train_mean_loss, prog_bar=True, on_epoch=False, on_step=True)
            return train_mean_loss      
        else:
            self.log("train_loss", train_sum_loss, prog_bar=True, on_epoch=False, on_step=True)
            return train_sum_loss
    
    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        out = self(task= self.predict_task, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        
        if self.predict_task == "ropes":
            start_logits = out["start_logits"]
            end_logits = out["end_logits"]
            return {"start_logits": start_logits, "end_logits": end_logits}
        
        if self.predict_task in ["causal_relation", "counterfactual_relation"]:
            return out["logits"]
        return torch.argmax(out["logits"], dim=1).detach().tolist()  
    
    def train_dataloader(self):
        combined_loader = CombinedLoader(self.train_dl_map, mode="max_size_cycle") 
        #combined_loader = CombinedLoader(self.train_dl_map, mode="min_size") 
        return combined_loader
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
class MTLCausalityModel2(pl.LightningModule):
    
    def __init__(self, config: dict):
        
        super().__init__()
        self.config = config 
        self.learning_rate = config["learning_rate"]
        self.predict_task = None 
        self.loss_strategy = config["loss_strategy"]
        self.update_seed(config["seed"])
        
        self.train_dl_map = {}
        
        # ======================== Model Details ==============================
        self.lm = AutoModel.from_pretrained(config["model_alias"])
        hidden_size = self.lm.config.hidden_size
        dropout_prob = self.lm.config.hidden_dropout_prob
        
        self.mc_head = GeneralMCTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob)
        
        self.task_heads = nn.ModuleDict({
                                "anli": self.mc_head,
                                "copa": self.mc_head,
                                "ecare": self.mc_head,
                                "cosmosqa": self.mc_head,
                                "ropes": RopeSimpleTaskLayer(hidden_size=hidden_size),
                                "wiqa": WiqaTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob)
                            })
            
    def set_train_dataloaders(self, dl_map: dict):
        self.train_dl_map = dl_map
    
    def update_seed(self, seed: int):
        pl.seed_everything(seed, workers=True)
    
    def update_lr(self, lr: float):
        self.learning_rate = lr
                                    
    def set_predict_task(self, task: str):
        self.predict_task = task 
    
    def forward(self, task, input_ids, attention_mask):
        
        if task in ["copa", "anli", "cosmosqa", "ecare"]:         
            num_choices = input_ids.shape[1]
            input_ids = input_ids.view(-1, input_ids.size(-1)) 
            mask = attention_mask.view(-1, attention_mask.size(-1))
            
            hidden_states = self.lm.forward(input_ids, attention_mask=mask)["last_hidden_state"]
            
            logits = self.task_heads[task](hidden_states)["logits"]
            logits = logits.view(-1, num_choices)
            logits = {"logits": logits}    
            
        else:
            hidden_states = self.lm(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
            logits = self.task_heads[task](hidden_states)
      
        return logits
    
    def training_step(self, batch, batch_idx):
        
        train_sum_loss = 0
        task_loss_list = []
        for task, b in batch.items():     
               
            if task == "ropes":
                start_positions = b["start_positions"]
                end_positions = b["end_positions"]
            
                logits_dict = self.forward(task=task, input_ids = b["input_ids"], attention_mask = b["attention_mask"])
                
                start_logits = logits_dict["start_logits"]
                end_logits = logits_dict["end_logits"]
                
                task_loss = None            
                if start_positions is not None and end_positions is not None:
                    
                    if len(start_positions.size()) > 1:
                        start_positions = start_positions.squeeze(-1)
                    
                    if len(end_positions.size()) > 1:
                        end_positions = end_positions.squeeze(-1)
                    
                    ignored_index = start_logits.size(1)
                    start_positions = start_positions.clamp(0, ignored_index)
                    end_positions = end_positions.clamp(0, ignored_index)
                    
                    loss_func = nn.CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_func(start_logits, start_positions)
                    end_loss = loss_func(end_logits, end_positions)
                    task_loss = (start_loss + end_loss) / 2  
            
            elif task in ["causal_relation", "counterfactual_relation"]:
                logits = self.forward(task = task, input_ids = b["input_ids"], attention_mask = b["attention_mask"])["logits"]   
                task_loss = F.cross_entropy(logits.view(-1, 5), b["labels"].view(-1))         
            
            else:
                logits = self.forward(task=task, input_ids=b["input_ids"], attention_mask=b["attention_mask"])["logits"]
                task_loss = F.cross_entropy(logits, b["labels"])
            
            # Add losses together and also add it to loss tracker
            train_sum_loss += task_loss
            task_loss_list.append(task_loss)
        
        if self.loss_strategy == "mean":
            train_mean_loss = torch.mean(torch.stack(task_loss_list))
            self.log("train_loss", train_mean_loss, prog_bar=True, on_epoch=False, on_step=True)
            return train_mean_loss      
        else:
            self.log("train_loss", train_sum_loss, prog_bar=True, on_epoch=False, on_step=True)
            return train_sum_loss
    
    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        out = self(task= self.predict_task, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        
        if self.predict_task == "ropes":
            start_logits = out["start_logits"]
            end_logits = out["end_logits"]
            return {"start_logits": start_logits, "end_logits": end_logits}
        
        if self.predict_task in ["causal_relation", "counterfactual_relation"]:
            return out["logits"]
        return torch.argmax(out["logits"], dim=1).detach().tolist()  
    
    def train_dataloader(self):
        combined_loader = CombinedLoader(self.train_dl_map, mode="max_size_cycle") 
        #combined_loader = CombinedLoader(self.train_dl_map, mode="min_size") 
        return combined_loader
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer