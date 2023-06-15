#%%
import torch 
import numpy as np
import pickle 
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score

from seqeval.metrics import classification_report as seqeval_report
from typing import List

from src.models import CausalityModel
from src.readers import aNLIMCreader, \
                        CausalIdentificationReader, \
                        CausalRelationReader, \
                        CounterFactualIdentificationReader, \
                        CounterFactualRelationReader, \
                        CopaMCreader, \
                        ECAREreader,\
                        CosmosQaMCReader, \
                        RopesReader, \
                        WIQAReader
from src.utilities import eval_ropes                        

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
        "ecare",
        "wiqa",
        "ropes",
        "anli",
    ]


TASK_CONFIG = {
        "causal_identification": {"seed": 1988, "learning_rate": 1e-4, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "causal_identification"}, 
        "causal_relation": {"seed": 3023, "learning_rate": 1e-4, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "causal_relation"},
        "counterfactual_identification": {"seed": 0, "learning_rate": 5e-5, "training_epochs": 4, "model_alias": "bert-base-uncased", "task": "counterfactual_identification"},
        "counterfactual_relation": {"seed": 0, "learning_rate": 1e-4, "training_epochs": 4, "model_alias": "bert-base-uncased", "task": "counterfactual_relation"},
        "copa": {"seed": 42, "learning_rate": 2e-5, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "copa"},
        "cosmosqa": {"seed": 1988, "learning_rate": 5e-05, "training_epochs": 1, "model_alias": "bert-base-uncased", "task": "cosmosqa"} ,
        "wiqa": {"seed": 42, "learning_rate": 2e-05, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "wiqa"},
        "ropes": {"seed": 1988, "learning_rate": 3e-05, "training_epochs": 1, "task": "ropes"},
        "anli": {"seed": 0, "learning_rate": 3e-05, "training_epochs": 1, "model_alias": "bert-base-uncased", "task": "anli"},
        "ecare": {"seed": 2022, "learning_rate": 3e-05, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "ecare"}
}


def evaluate_relation_task(batch_pred_logits, test_dl, reader):
    
    preds, gold_labels = [], []
    for i, batch in enumerate(test_dl):
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
    return {"report": report, "preds": preds}
                   
            
def eval_task_with_test(batch_preds, test_dl):
       
    gold, preds = [], []
    for i, batch in enumerate(test_dl):
        preds.extend(batch_preds[i])
        gold.extend(batch["labels"])
   
    acc = accuracy_score(gold, preds)
    return {"accuracy": acc, "preds": preds}
            
def initialize_readers(model_alias: str):
    
    if model_alias == "bert-base-uncased":
        file_name = "reader_cache/bert-base-uncased-cached-readers.pkl"
        
    if model_alias == "nghuyong/ernie-2.0-base-en":
        file_name = "reader_cache/ernie-2.0-base-en.pkl"
               
    initialized_readers = pickle.load(open(file_name, "rb"))
    return initialized_readers
    
    
def main():
    
   alias = "nghuyong/ernie-2.0-base-en"
   
   print("initializing readers ...")
   initialized_reader_map = initialize_readers(alias)
   
   print("begining exp... ")
   for task in TASK_LIST:
        print(f"generating baselines for {task}")
        save_path = f"experiments/baselines2/{alias}/"
    
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        if not os.path.exists(save_path + "preds/"):
            os.makedirs(save_path + "preds/")
        
        # 1. Load data loader
        reader = initialized_reader_map[task]["reader"]
        train_dl = initialized_reader_map[task]["train"]
        val_dl = initialized_reader_map[task]["val"]
        
        # 2. Setup model
        config = TASK_CONFIG[task]
        config["model_alias"] = alias
        model = CausalityModel(config)
        model.update_seed(config["seed"])
        model.update_lr(config["learning_rate"])
        
        trainer =  pl.Trainer(
            max_epochs=config["training_epochs"], 
            enable_model_summary=False,
            enable_checkpointing=False,
            gpus=[0],
            accelerator="gpu", 
            precision=16,
        )        

        trainer.fit(model, train_dl, val_dl)
        
        if task in ["causal_relation", "counterfactual_relation"]:
            test_dl = initialized_reader_map[task]["test"]
            batch_preds = trainer.predict(model, test_dl)
            results = evaluate_relation_task(batch_preds, test_dl, reader)
            
            with open(save_path + "preds/" + model.task + "_eval_preds.txt", "w") as f:
                for pred in results["preds"]:
                    f.write(str(pred) + " \n")
            
            with open(save_path + "baseline.txt", "a") as f:
                f.write(f"\n task: {model.task}\n")
                f.write(results["report"])
                    
        elif task in ["anli", "cosmosqa", "ecare"]:
            batch_preds = trainer.predict(model, val_dl)
            results = eval_task_with_test(batch_preds, val_dl)
            
            with open(save_path + "baseline.txt", "a") as f:  
                f.write(f"\n{model.task} - accuracy: {round(results['accuracy'], 3)}")
    
            with open(save_path + "preds/" + model.task + "_eval_preds.txt", "w") as f:
                for pred in results["preds"]:
                    f.write(str(pred) + " \n")
            
            
        elif task == "ropes":
            batch_pred_logits = trainer.predict(model, val_dl)
        
            res = eval_ropes(reader, batch_pred_logits, val_dl, fix_or=True)
            acc = res["accuracy"]
            f1 = res["f1"]
            
            with open(save_path + "baseline.txt", "a") as f:  
                f.write(f"\n{model.task} - accuracy: {round(acc, 3)} - F1: {round(f1, 3)}")
    
            with open(save_path + "preds/" + model.task + "_eval_preds.txt", "w") as f:
                for pred in res["preds"]:
                    f.write(str(pred) + " \n")
        
        else:
            test_dl = initialized_reader_map[task]["test"]
            batch_preds = trainer.predict(model, test_dl)
            results = eval_task_with_test(batch_preds, test_dl)
            
            with open(save_path + "baseline.txt", "a") as f:  
                f.write(f"\n{model.task} - accuracy: {round(results['accuracy'], 3)}")
    
            with open(save_path + "preds/" + model.task + "_eval_preds.txt", "w") as f:
                for pred in results["preds"]:
                    f.write(str(pred) + " \n")
             
      
if __name__ == "__main__":
    main()
