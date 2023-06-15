import numpy as np 
import os 
import pickle 
import torch
import pytorch_lightning as pl 

from src.utilities import eval_task_with_test, evaluate_relation_task, eval_ropes
from src.models import MTLCausalityModel
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ------------------------ Task Config --------------------------------------- #

TASK_CONFIG = {
        "copa": {"seed": 1, "learning_rate": 1e-5, "training_epochs": 1},
        "cosmosqa": {"seed": 3023, "learning_rate": 2e-05, "training_epochs": 1} ,
        "wiqa": {"seed": 1988, "learning_rate": 2e-05, "training_epochs": 1},
        "ropes": {"seed": 0, "learning_rate": 5e-05, "training_epochs": 1},
        "anli": {"seed": 3023, "learning_rate": 2e-05, "training_epochs": 1},
        "ecare": {"seed": 42, "learning_rate": 2e-05, "training_epochs": 1}
}

eval_tasks =  [ "anli", "ecare", "cosmosqa", "wiqa" ]
train_tasks = [ "anli", "ecare", "cosmosqa", "wiqa", "ropes", "copa" ]


def initialize_readers(model_alias: str):
    
    if model_alias == "bert-base-uncased":
        file_name = "reader_cache/bert-base-uncased-readers.pkl"
    
    if model_alias == "nghuyong/ernie-2.0-en":
        file_name = "reader_cache/erine-base-cached-readers.pkl"
        
    if model_alias == "bert-base-cased":
        file_name = "reader_cache/bert-base-cased-readers.pkl"
    
    if model_alias == "nghuyong/ernie-2.0-base-en":
        file_name = "reader_cache/ernie-2.0-base-en.pkl"
    
    initialized_readers = pickle.load(open(file_name, "rb"))
    return initialized_readers


# ------------------ Setup MTL Model and Train ------------------------------- #


alias = "nghuyong/ernie-2.0-base-en"
print(f"Initializing readers for {alias}")
reader_map = initialize_readers(alias)

train_dl_map = {task: reader_map[task]["train"] for task in train_tasks}

config = {
    "model_alias": alias,
    "learning_rate": 3e-5,
    "tasks": train_tasks,
    "loss_strategy": "mean",
    "seed": 3023,
}
        
trainer =  pl.Trainer(
        max_steps = 8000,
        enable_model_summary=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        gpus=[0],
        accelerator="gpu", 
        precision=16
    )

weights_path = "model_cache/mtl/ernie-2.0-base-en/final-config/model-8000.pt"

model = MTLCausalityModel(config)
#model.load_state_dict(torch.load( "model_cache/mtl/bert-base-uncased/config2/config2-4000.pt" ))

model.set_train_dataloaders(train_dl_map)
trainer.fit(model)
torch.save(model.state_dict(), weights_path)

with open("experiments/mtl/ernie-2.0-base-en/results/final-config/all_results.txt", "a") as all_f:
    all_f.write("\n================= MTL2 Simplified No Finetuning - Finetune 8000 steps [Config: anli, cosmosqa, ecare, wiqa] ===================\n")
    all_f.write(f"seed: {config['seed']} | learning rate: {config['learning_rate']}  \n")
    
    acc_scores = []
    for task in eval_tasks:
        # Grab validation dataset as test is unavailable 
        if task in ["anli", "cosmosqa", "ecare", "ropes"]:
            dl = reader_map[task]["val"]
        else:
            dl = reader_map[task]["test"]
        
        dl_map = {task: reader_map[task]["train"]}
        task_config = TASK_CONFIG[task]

        trainer =  pl.Trainer(
                    max_epochs=task_config["training_epochs"], 
                    enable_model_summary=False,
                    enable_checkpointing=False,
                    num_sanity_val_steps=0,
                    limit_val_batches=0,
                    gpus=[0],
                    accelerator="gpu", 
                    precision=16
                )
        model = MTLCausalityModel(config)
        model.load_state_dict(torch.load(weights_path))

        model.set_train_dataloaders(dl_map)
        model.update_lr(task_config["learning_rate"])
        model.update_seed(task_config["seed"])
        trainer.fit(model)
        
        # 1. Set predict task
        print(f"setting eval task -> {task}")
        model.set_predict_task(task)
        
        # 2. Get batch preds
        model.set_predict_task(task)
        batch_preds = trainer.predict(model, dl)
        
        # 3. Results 
        results_map = {}
        if task == "ropes":
            try:
                reader = reader_map[task]["reader"]
                res = eval_ropes(reader, batch_preds, dl)
                all_f.write(f"{task} - EM: {res['accuracy']} | F1: {res['f1']}\n")
                results_map[task] = res
                acc_scores.append(res["accuracy"])
                continue
            except:
                all_f.write(f"{task} - EM: error | F1: error \n")
                continue
        
        elif task in ["causal_relation", "counterfactual_relation"]:
            reader = reader_map[task]["reader"]
            res = evaluate_relation_task(batch_preds, reader, dl)
            results_map[task] = res
            all_f.write(f"{task} \n {res['report']} \n")
            continue 
        else:
            res = eval_task_with_test(batch_preds, dl)    
            results_map[task] = res   
            all_f.write(f"{task} - acc: {res['accuracy']} \n")
            acc_scores.append(res["accuracy"])
            continue
    all_f.write(f"\n seed: {config['seed']} | learning rate: {config['learning_rate']} -> mean acc {np.mean(acc_scores)} scores: {acc_scores}")
    all_f.close()
                