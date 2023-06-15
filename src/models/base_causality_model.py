import torch
import torch.nn as nn 
from transformers import AutoModel
import pytorch_lightning as pl
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
                        WiqaTaskLayer 
      
class CausalityModel(pl.LightningModule):
    
    def __init__(self, config: dict):
        
        super().__init__()
          
        self.task = config["task"]
        self.learning_rate = config["learning_rate"]
        
        self.lm = AutoModel.from_pretrained(config["model_alias"])
        hidden_size = self.lm.config.hidden_size
        dropout_prob = self.lm.config.hidden_dropout_prob
        
        self.anli_layer = aNLITaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob)      
        self.causal_identification_layer = CausalIdentificationTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob)   
        self.casual_relation_layer = CausalRelationTaskLayer(num_tags=5, hidden_size=hidden_size, dropout_prob=dropout_prob)   
        self.cf_identification_layer = CFIdentificationTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob)
        self.cf_relation_layer = CounterFactualRelationTaskLayer(num_tags=5, hidden_size=hidden_size, dropout_prob=dropout_prob)
        self.copa_layer = CopaTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob)
        self.cosmos_layer = CosmosQaTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob)
        self.ecare_layer = ECARETaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob)
        self.ropes_layer = RopeSimpleTaskLayer(hidden_size=hidden_size)
        self.wiqa_layer = WiqaTaskLayer(hidden_size=hidden_size, dropout_prob=dropout_prob)
        
        self.task_map = {
                "anli": self.anli_layer,
                "causal_identification": self.causal_identification_layer,
                "causal_relation": self.casual_relation_layer,
                "counterfactual_identification": self.cf_identification_layer,
                "counterfactual_relation": self.cf_relation_layer,
                "copa": self.copa_layer,
                "ecare": self.ecare_layer,
                "cosmosqa": self.cosmos_layer,
                "ropes": self.ropes_layer,
                "wiqa": self.wiqa_layer
            }

    
    def update_seed(self, seed: int):
        pl.seed_everything(seed, workers=True)
    
    def update_lr(self, lr: float):
        self.learning_rate = lr
    
    def update_task(self, task: str):
        self.task = task 
                                
    def forward(self, input_ids, attention_mask):
        
        if self.task in ["copa", "anli", "cosmosqa", "ecare"]:
            
            num_choices = input_ids.shape[1]
            input_ids = input_ids.view(-1, input_ids.size(-1)) 
            mask = attention_mask.view(-1, attention_mask.size(-1))
            
            hidden_states = self.lm.forward(input_ids, attention_mask=mask)["last_hidden_state"]
            
            logits = self.task_map[self.task](hidden_states)["logits"]
            logits = logits.view(-1, num_choices)
            logits = {"logits": logits}
        
        else:
            hidden_states = self.lm(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
            logits = self.task_map[self.task](hidden_states)
      
        return logits

    
    def training_step(self, batch, batch_idx):
        
        if self.task == "ropes":
            start_positions = batch["start_positions"]
            end_positions = batch["end_positions"]
            
            logits_dict = self.forward(input_ids = batch["input_ids"], 
                                attention_mask = batch["attention_mask"])
            
            start_logits = logits_dict["start_logits"]
            end_logits = logits_dict["end_logits"]
            
            loss = None            
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
                loss = (start_loss + end_loss) / 2        
        
        elif self.task in ["causal_relation", "counterfactual_relation"]:
            loss_func = nn.CrossEntropyLoss()
        
            logits = self.forward(input_ids = batch["input_ids"], 
                        attention_mask = batch["attention_mask"])["logits"]
            
            loss = loss_func(logits.view(-1, 5), batch["labels"].view(-1))
    
        else:
            loss_func = nn.CrossEntropyLoss()
        
            logits = self.forward(input_ids = batch["input_ids"], 
                        attention_mask = batch["attention_mask"])["logits"]
            loss = loss_func(logits, batch["labels"])       
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        
        if self.task == "ropes": 
            start_positions = batch["start_positions"]
            end_positions = batch["end_positions"]
            
            logits_dict = self.forward(input_ids = batch["input_ids"], 
                                attention_mask = batch["attention_mask"])
            
            start_logits = logits_dict["start_logits"]
            end_logits = logits_dict["end_logits"]
            
            loss = None            
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
                loss = (start_loss + end_loss) / 2
                
                self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        
        elif self.task in ["causal_relation", "counterfactual_relation"]:
            labels = batch["labels"]
            
            loss_func = nn.CrossEntropyLoss()
            logits = self.forward(input_ids = batch["input_ids"],
                            attention_mask = batch["attention_mask"])["logits"]
            
            loss = loss_func(logits.view(-1, 5), labels.view(-1))
            
            # compute accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = logits.view(-1, 5) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)

            active_accuracy = labels.view(-1) != -100 
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            acc = seqeval_accuracy(labels.tolist(), predictions.tolist())
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            self.log("val_acc", acc, prog_bar=True, on_epoch=True)            
           
        else:       
            loss_func = nn.CrossEntropyLoss()
            logits = self.forward(input_ids = batch["input_ids"],
                            attention_mask = batch["attention_mask"])["logits"]
            
            loss = loss_func(logits, batch["labels"])
            _, yhat = torch.max(logits, dim=1)
            acc = accuracy_score(batch["labels"].tolist(), yhat.tolist())
            
        
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        
        return loss

    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        out = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        
        if self.task == "ropes":
            start_logits = out["start_logits"]
            end_logits = out["end_logits"]
            return {"start_logits": start_logits, "end_logits": end_logits}
        
        if self.task in ["causal_relation", "counterfactual_relation"]:
            return out["logits"]
        return torch.argmax(out["logits"], dim=1).detach().tolist()

       
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer