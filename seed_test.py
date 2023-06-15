from numpy import sort
import torch
import torch.nn as nn 
from transformers import AutoModel
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score
from seqeval.metrics import accuracy_score as seqeval_accuracy

import pandas as pd 
import pickle
from src.readers import RopesReader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

from src.models import CausalityModel
from src.readers import ECAREreader
from src.utilities import eval_ropes, evaluate_relation_task, eval_task_with_val
import os 

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def initialize_readers(model_alias: str):
    
    if model_alias == "bert-base-uncased":
        file_name = "reader_cache/bert-base-uncased-readers.pkl"
        
    if model_alias == "bert-base-cased":
        file_name = "reader_cache/bert-base-cased-readers.pkl"
    
    if model_alias == "nghuyong/ernie-2.0-en":
        file_name = "reader_cache/erine-base-cached-readers.pkl"
    
    if model_alias ==  "nghuyong/ernie-2.0-large-en":
        file_name = "reader_cache/erine-large-cached-readers.pkl"
    
    if model_alias == "bert-large-uncased":
        file_name = "reader_cache/bert-large-cached-readers.pkl" 
    
    if model_alias == "nghuyong/ernie-2.0-large-en":
        file_name = "reader_cache/erine-large-cached-readers.pkl"
    
    if model_alias == "nghuyong/ernie-2.0-base-en":
        file_name = "reader_cache/ernie-2.0-base-en.pkl"
     
   
    initialized_readers = pickle.load(open(file_name, "rb"))
    return initialized_readers
    

alias = "bert-base-uncased"
reader_map = initialize_readers(alias)

TASK_LIST = [
        #"copa", 
        #"ecare", 
        #"wiqa", 
        #"ropes", 
        #"anli", 
        "cosmosqa" 
    ]

def train_model(model, train_dl, val_dl):
    
    if "relation" in model.task:
        num_steps = 200
    else:
        num_steps = 50
    
    trainer =  pl.Trainer(
                max_steps= num_steps, 
                enable_model_summary=False,
                enable_checkpointing=False,
                gpus=[0],
                accelerator="gpu", 
                precision=16,
            )
    trainer.fit(model, train_dl, val_dl)
    return model 

def eval_model(model, lr, seed, task, reader_map, file_name):
    
        trainer =  pl.Trainer(
            max_steps= 100, 
            enable_model_summary=False,
            enable_checkpointing=False,
            gpus=[0],
            accelerator="gpu", 
            precision=16,
        )
    
        if task == "ropes":
            val_dl = reader_map[task]["val"]
            batch_preds = trainer.predict(model, val_dl)
            acc = eval_ropes(reader, batch_preds, val_dl)["accuracy"]
            
            with open(file_name, "a") as f:
                f.write(f"\n{task} | seed: {seed} | learning_rate: {lr} | acc: {acc}")
            return acc 
                    
        elif task in ["anli", "cosmosqa", "ecare"]:
            val_dl = reader_map[task]["val"]
            batch_preds = trainer.predict(model, val_dl)
            acc = eval_task_with_val(batch_preds, val_dl)["accuracy"]

            with open(file_name, "a") as f:
                f.write(f"\n{task}| seed: {seed} | learning_rate: {lr} | acc: {acc}")
                
            return acc 
                
        elif task in ["causal_relation", "counterfactual_relation"]:
            test_dl = reader_map[task]["test"]
            batch_preds = trainer.predict(model, test_dl)
            res = evaluate_relation_task(batch_preds, reader, test_dl)
            
            with open(file_name, "a") as f:
                f.write(f"\n{task}| seed: {seed} | learning_rate: {lr} \n")                 
                f.write(res["report"])
            return res["f1"]
        
        else:
            test_dl = reader_map[task]["test"]
            batch_preds = trainer.predict(model, test_dl)
            acc = eval_task_with_val(batch_preds, test_dl)["accuracy"]
            
            with open(file_name, "a") as f:
                f.write(f"\n{task}| seed: {seed} | learning_rate: {lr} | acc: {acc}")      
            
            return acc            

for task in TASK_LIST:
    print(f"evaluating for model: {alias}")
       
    reader = reader_map[task]["reader"]
    train_dl = reader_map[task]["train"]
    val_dl = reader_map[task]["val"]
    
    config = {"model_alias": alias, "task": task}
    
    path = f"experiments/seed_tests/{alias}/"
    file_name = path + f"{task}.txt"
    all_results = path + "final_seeds.txt"
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    # # #1. Find best lr 
    lr_vals = [] 
    
    for lr in [1e-5, 3e-5, 2e-5, 5e-5]:
        config["learning_rate"] = lr
        model = CausalityModel(config)
        model.update_seed(0)
        model = train_model(model, train_dl, val_dl)
        
        acc = eval_model(model, lr, 0, task, reader_map, file_name)
        lr_vals.append((lr, acc))
    
    best_lr = sorted(lr_vals, key=lambda x: x[1], reverse=True)[0][0]
    print(f"best lr: {best_lr}")    
    
    config["learning_rate"] = best_lr
    
    set_results = []
    # 2. Record seed results             
    for seed in [0, 1, 42, 1988, 2022, 3023]:
        model = CausalityModel(config)
        model.update_seed(seed)    
        model = train_model(model, train_dl, val_dl)
        acc = eval_model(model, best_lr, seed, task, reader_map, file_name)        
        
        set_results.append((seed, best_lr, acc))
    best_set = sorted(set_results, key=lambda x: x[2], reverse=True)[0]

    with open(all_results, "a") as f:
        f.write(f"\nTask: {task} | Best params ==> seed: {best_set[0]} | lr {best_set[1]} | acc {best_set[2]}")      
