#%%
import torch 
import pickle 
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from seqeval.metrics import classification_report as seqeval_report

from typing import List

from src.models import CausalityModel

from src.utilities import eval_ropes, initialize_readers

import warnings
import os 
warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "true"

TASK_CONFIG = {
        "causal_identification": {"seed": 0, "learning_rate": 2e-5, "training_epochs": 3, "task": "causal_identification"}, 
        "causal_relation": {"seed": 0, "learning_rate": 5e-5, "training_epochs": 3, "task": "causal_relation"},
        "counterfactual_identification": {"seed": 42, "learning_rate": 3e-05, "training_epochs": 3, "task": "counterfactual_identification"},
        "counterfactual_relation": {"seed": 2022, "learning_rate": 5e-5, "training_epochs": 3, "task": "counterfactual_relation"},
        "anli": {"seed": 0, "learning_rate": 2e-05, "training_epochs": 1,  "task": "anli"},
        "copa": {"seed": 42, "learning_rate": 2e-05, "training_epochs": 5, "task": "copa"},
        "cosmosqa": {"seed": 0, "learning_rate": 3e-05, "training_epochs": 1, "task": "cosmosqa"} ,
        "ecare": {"seed": 2022, "learning_rate": 3e-05, "training_epochs": 2, "task": "ecare"},
        "wiqa": {"seed": 1988, "learning_rate": 2e-05, "training_epochs": 3, "task": "wiqa"},
        "ropes": {"seed": 42, "learning_rate": 3e-05, "training_epochs": 1, "task": "ropes"},
        
}

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


def evaluate_relation_task(model: CausalityModel, 
                    reader,
                    train_dl: DataLoader, 
                    val_dl: DataLoader, 
                    test_dl: DataLoader,
                    save_loc: str):
    epochs = TASK_CONFIG[model.task]["training_epochs"]      
    trainer =  pl.Trainer(
                max_epochs=epochs, 
                enable_model_summary=False,
                enable_checkpointing=False,
                gpus=[0],
                accelerator="gpu", 
                precision=16,
            )   
    batch_pred_logits = trainer.predict(model, test_dl)

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
    
    with open(save_loc + "baseline.txt", "a") as f:
        f.write(f"\n task: {model.task}\n")
        f.write(report)
        
        
def eval_task_with_val(model, train_dl: DataLoader, val_dl: DataLoader, save_loc):
    epochs = TASK_CONFIG[model.task]["training_epochs"]      
    trainer =  pl.Trainer(
                max_epochs=epochs, 
                enable_model_summary=False,
                enable_checkpointing=False,
                gpus=[0],
                accelerator="gpu", 
                precision=16,
            )
    batch_preds = trainer.predict(model, val_dl)
    
    gold = []
    preds = []
    
    for i, batch in enumerate(val_dl):
        preds.extend(batch_preds[i])
        gold.extend(batch["labels"])
    
    acc = accuracy_score(gold, preds)
    
    with open(save_loc + "baseline.txt", "a") as f:  
        f.write(f"\n {model.task} - accuracy: " + str(round(acc, 3)))
        
    with open(save_loc + "preds/" + model.task + "_eval_preds.txt", "w") as f:
        for pred in preds:
            f.write(str(pred) + " \n")
            

def eval_task_with_test(model: CausalityModel, 
                    reader, 
                    train_dl: DataLoader, 
                    val_dl: DataLoader,
                    test_dl: DataLoader,
                    save_loc: str):
        
    epochs = TASK_CONFIG[model.task]["training_epochs"]      
    
    trainer =  pl.Trainer(
                max_epochs=epochs, 
                enable_model_summary=False,
                enable_checkpointing=False,
                accelerator="gpu", 
                gpus=[0],
                precision=16,
            )
    
    batch_preds = trainer.predict(model, test_dl)
    
    gold = []
    preds = []
    
    for i, batch in enumerate(test_dl):
        preds.extend(batch_preds[i])
        gold.extend(batch["labels"])
   
    acc = accuracy_score(gold, preds)
    with open(save_loc + "baseline.txt", "a") as f:
        acc = str(round(acc, 3))
        f.write(f"\n{model.task} - Accuracy: {acc}")
        
    with open(save_loc + "preds/" + model.task + "_test_preds.txt", "w") as f:
        for pred in preds:
            f.write(str(pred) + " \n")
            
def initialize_readers(model_alias: str):
    
    if model_alias == "bert-base-uncased":
        file_name = "reader_cache/bert-base-uncased-cached-readers.pkl"
    
    if model_alias == "bert-base-cased":
        file_name = "reader_cache/bert-base-cased-readers.pkl"
    
    if model_alias == "nghuyong/ernie-2.0-en":
        file_name = "reader_cache/erine-base-cached-readers.pkl"
    
    
    if model_alias == "nghuyong/ernie-2.0-base-en":
        file_name = "reader_cache/ernie-2.0-base-en.pkl"
    
    
    initialized_readers = pickle.load(open(file_name, "rb"))
    return initialized_readers
   

def main():
    
    alias = "nghuyong/ernie-2.0-base-en"
                
    print(f"initializing readers ... {alias}")
    initialized_reader_map = initialize_readers(alias)
    
    print("begining exp... ")
    for task in [
        "causal_identification",
        "causal_relation",
        "counterfactual_identification",
        "counterfactual_relation",
        "copa",
        "anli",
        "ecare",
        "wiqa",
        "ropes",
        "cosmosqa"  
    ]:
        
        print(f"generating baselines for {task}")
        save_path = f"experiments/transfer_baselines/{alias}/{task}/"
    
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        if not os.path.exists(save_path + "preds/"):
            os.makedirs(save_path + "preds/")
                
        # # 1. Fine-tune model 
        reader = initialized_reader_map[task]["reader"]
        train_dl = initialized_reader_map[task]["train"]
        val_dl = initialized_reader_map[task]["val"]
        
        # # 2. Setup model
        config = TASK_CONFIG[task]
        config["model_alias"] = alias
        model = CausalityModel(config)
        model.update_seed(config["seed"])
        
        trainer =  pl.Trainer(
            max_epochs=config["training_epochs"], 
            enable_model_summary=False,
            enable_checkpointing=False,
            gpus=[0],
            accelerator="gpu", 
            precision=16,
        )
    
        trainer.fit(model, train_dl, val_dl)
        
        model_name = f"model_cache/ernie-2.0-base-en/{task}.pkl"
        pickle.dump(model, open(model_name, "wb"))
        
        #model = pickle.load(open(model_name, "rb"))   
        
        # Evaluate transferrability of task on remaineder of tasks             
        for eval_task in TASK_LIST:
            
            print(f"evaluating task - {eval_task}")
            if eval_task == task:
                continue
            
            reader = initialized_reader_map[eval_task]["reader"]
            train_dl = initialized_reader_map[eval_task]["train"]
            val_dl = initialized_reader_map[eval_task]["val"]

            eval_model = pickle.load(open(model_name, "rb"))
            eval_model.task = eval_task 
            eval_model.update_lr(TASK_CONFIG[eval_task]["learning_rate"])
            eval_model.update_seed(TASK_CONFIG[eval_task]["seed"])           
                            
            trainer =  pl.Trainer(
                max_epochs=TASK_CONFIG[eval_task]["training_epochs"], 
                enable_model_summary=False,
                enable_checkpointing=False,
                gpus=[0],
                accelerator="gpu", 
                precision=16,
            )
            
            trainer.fit(eval_model, train_dl, val_dl)
            
            if eval_task in ["causal_relation", "counterfactual_relation"]:
                test_dl = initialized_reader_map[eval_task]["test"]
                evaluate_relation_task(eval_model, reader, train_dl, val_dl, test_dl, save_path)
            
            elif eval_task in ["anli", "cosmosqa", "ecare"]:
                eval_task_with_val(eval_model, train_dl, val_dl, save_path)
                
            elif eval_task == "ropes":
                batch_pred_logits = trainer.predict(eval_model, val_dl)
                
                res = eval_ropes(reader, batch_pred_logits, val_dl)
                acc = res["accuracy"]
                f1 = res["f1"]
                
                with open(save_path + "baseline.txt", "a") as f:  
                    f.write(f"\n{eval_model.task} - accuracy: {round(acc, 3)} - F1: {round(f1, 3)}")
                
                with open(save_path + "preds/" + eval_model.task + "_eval_preds.txt", "w") as f:
                        for pred in res["preds"]:
                            f.write(str(pred) + " \n")
            else:
                test_dl = initialized_reader_map[eval_task]["test"]
                eval_task_with_test(eval_model, reader, train_dl, val_dl, test_dl, save_path)
             
      
if __name__ == "__main__":
    main()
