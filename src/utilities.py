import torch 
import numpy as np
import pickle 
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1_score 

from typing import List

from src.models import CausalityModel
from src.readers import aNLIMCreader, \
                        CausalIdentificationReader, \
                        CausalRelationReader, \
                        CounterFactualIdentificationReader, \
                        CounterFactualRelationReader, \
                        CopaMCreader, \
                        CosmosQaMCReader, \
                        RopesReader, \
                        WIQAReader

import warnings
import os 
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "true"

TASK_LIST = [
                "causal_identification", 
                "causal_relation",
                "counterfactual_identification",
                "counterfactual_relation",
                "copa",
                "cosmosqa",
                "wiqa",
                "ropes",
                "anli",
            ]


READER_MAP = {
                "causal_identification": CausalIdentificationReader, 
                "causal_relation": CausalRelationReader, 
                "counterfactual_identification": CounterFactualIdentificationReader,
                "counterfactual_relation": CounterFactualRelationReader,
                "copa": CopaMCreader,
                "cosmosqa": CosmosQaMCReader, 
                "wiqa": WIQAReader, 
                "ropes": RopesReader,
                "anli": aNLIMCreader
            }


def finetune_task(model: CausalityModel, 
            train_dl: DataLoader, 
            val_dl: DataLoader,
            epochs: int, 
        ) -> CausalityModel:    
    
    trainer =  pl.Trainer(
                max_epochs=epochs, 
                enable_model_summary=False,
                enable_checkpointing=False,
                gpus=[0],
                accelerator="gpu", 
                precision=16,
            )
       
    trainer.fit(model, train_dl, val_dl)
    return model 

def evaluate_relation_task(batch_pred_logits, reader, dl: DataLoader):

    preds, gold_labels = [], []
    for i, batch in enumerate(dl):
        labels = batch["labels"]
        logits = batch_pred_logits[i]
        
        flattened_targets = labels.view(-1)
        active_logits = logits.view(-1, 5)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        
        active_accuracy = labels.view(-1) != -100
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        gold_labels.append([reader.id2label[i.item()] for i in labels])
        preds.append([reader.id2label[i.item()] for i in predictions])

    report = seqeval_report(gold_labels, preds)
    f1 = seqeval_f1_score(gold_labels, preds)
    return {"report": report, "preds": preds, "f1": f1}      
        
        
def eval_task_with_val(batch_preds, dl):
    
    gold = []
    preds = []
    
    for i, batch in enumerate(dl):
        preds.extend(batch_preds[i])
        gold.extend(batch["labels"])
    
    acc = accuracy_score(gold, preds)
    return {"accuracy": acc, "preds": preds} 
    
            
def eval_ropes_legacy(reader, batch_preds, val_dl: DataLoader):
           
    gold = reader.data.query("split=='val'")["answer_label"].tolist()
    gold = [text.lower().strip() for text in gold]
    preds = []
    
    for i, batch in enumerate(val_dl):
        start_idx = batch_preds[i][0]
        end_idx = batch_preds[i][1]

        for i in range(len(batch["input_ids"])):
            span_pred = reader.tokenizer.decode(batch["input_ids"][i][start_idx[i]: end_idx[i]])
            preds.append(span_pred.lower().strip())
    
    acc = accuracy_score(gold, preds)
    
    return {"accuracy": acc, "preds": preds}


def compute_f1(prediction, truth):
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def compute_average_f1(preds, gold):
    scores = [compute_f1(p, gold[i]) for i,p in enumerate(preds)]
    return np.mean(scores)

def eval_ropes(reader, batch_pred_logits, val_dl: DataLoader, fix_or: bool=False):
    
    preds = []
    
    for i, batch in enumerate(val_dl):
        batch_logits = batch_pred_logits[i]
        batch_input_ids = batch["input_ids"]
    
        for ix in range(len(batch["input_ids"])):
            
            n_best_size = 20 
            max_answer_length = 10
            
            start_logits = batch_logits["start_logits"][ix].numpy()
            end_logits = batch_logits["end_logits"][ix].numpy()
            
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                     
            answer_dict = {}
            valid_answers = []
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    if start_index <= end_index: # We need to refine that test to check the answer is inside the context
                        ix_input_ids = batch_input_ids[ix][start_index: end_index]
                        text = reader.tokenizer.decode(ix_input_ids)
                        
                        d = {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_index": start_index,
                                "end_index": end_index,
                                "text": text
                            }
                        valid_answers.append(d)
                        answer_dict[text] = d        
                        
            valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)
            best_answer = valid_answers[0]["text"]
                
            if " or " in best_answer:
                option_scores = []
                for option in best_answer.split(" or "):
                    if option in answer_dict:
                        option_scores.append((option, answer_dict[option]["score"]))
                    else:
                        option_scores.append((option, 0))
                
                options = sorted(option_scores, key=lambda x: x[1], reverse=True)
                best_answer = options[0][0]
            
            preds.append(best_answer)
    
    gold = reader.data.query("split=='val'")["answer_label"].tolist()
    gold = [text.lower().strip() for text in gold]
    preds = [text.lower().strip() for text in preds]
              
    acc = accuracy_score(gold, preds)
    f1 = compute_average_f1(preds, gold)
    return {"accuracy": acc, "f1": f1, "preds": preds}

def eval_task_with_test(batch_preds, test_dl: DataLoader):
           
    gold = []
    preds = []
    
    for i, batch in enumerate(test_dl):
        preds.extend(batch_preds[i])
        gold.extend(batch["labels"])
   
    acc = accuracy_score(gold, preds)
    return {"accuracy": acc, "preds": preds}
     
           
def initialize_readers(model_alias: str):
    
    if model_alias == "bert-base-uncased":
        file_name = "reader_cache/bert-base-uncased-cached-readers.pkl"
    
    initialized_readers = pickle.load(open(file_name, "rb"))
    return initialized_readers
    
