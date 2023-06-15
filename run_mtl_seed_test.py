from itertools import accumulate
import numpy as np 
import os 
import pickle 
import pytorch_lightning as pl 

from src.utilities import eval_task_with_test, evaluate_relation_task, eval_ropes
from src.models import MTLCausalityModel
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from src.readers import aNLIMCreader, \
                        CausalIdentificationReader, \
                        CausalRelationReader, \
                        CounterFactualIdentificationReader, \
                        CounterFactualRelationReader, \
                        CopaMCreader, \
                        CosmosQaMCReader, \
                        ECAREreader, \
                        RopesReader, \
                        WIQAReader

READER_MAP = {
                "causal_identification": CausalIdentificationReader, 
                "causal_relation": CausalRelationReader, 
                "counterfactual_identification": CounterFactualIdentificationReader,
                "counterfactual_relation": CounterFactualRelationReader,
                "copa": CopaMCreader,
                "cosmosqa": CosmosQaMCReader, 
                "ecare": ECAREreader,
                "wiqa": WIQAReader, 
                "ropes": RopesReader,
                "anli": aNLIMCreader
            }
 
train_tasks =  [ 
          "anli",
          "ecare",
          "cosmosqa",
          "ropes",
          "wiqa" 
      ]

eval_tasks = [ 
          "anli",
          "copa",
          "ecare",
          "cosmosqa",
          "ropes",
          "wiqa" 
        ]

def initialize_readers(model_alias: str):
    
    initialized_reader_map = {}
    
    for task in [ #"causal_identification", "causal_relation",
                  #"counterfactual_identification", "counterfactual_relation",
                  "copa", "cosmosqa", 'ecare', "wiqa", "ropes", "anli"]:
        print(f"generating baseline for {model_alias} - {task}")
        reader = READER_MAP[task](model_alias = model_alias)
        initialized_reader_map[task] = reader        
    return initialized_reader_map

alias = "bert-base-uncased"
all_readers = initialize_readers(alias)


dl_map = {}
for task, reader in all_readers.items():
    print(f"getting dataloader for {task}")
    dl_map[task] = reader.get_dataloader("train", batch_size=24)


for seed in [0, 1, 42, 1988, 2022, 3023]:          
    lr = 5e-5
    config = {
        "model_alias":  alias,
        "learning_rate": lr,
        "tasks": train_tasks,
        "loss_strategy": "mean",
        "seed": seed,
    }
    
    trainer =  pl.Trainer(
            max_steps = 100,
            enable_model_summary=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
            gpus=[0],
            accelerator="gpu", 
            precision=16
        )
        
    model = MTLCausalityModel(config)
    model.set_train_dataloaders(dl_map)
    trainer.fit(model)
    
    with open("experiments/mtl/bert-base-uncased/seed_tests/all_results.txt", "a") as all_f, \
        open("experiments/mtl/bert-base-uncased/seed_tests/summary.txt", "a") as sum_f:
        all_f.write("\n==============================================\n")
        all_f.write(f"seed: {seed} | learning rate: {lr} \n")
        
        acc_scores = []
        for task in eval_tasks:
            # Grab validation dataset as test is unavailable 
            if task in ["anli", "cosmosqa", "ecare", "ropes"]:
                dl = all_readers[task].get_dataloader("val", batch_size=12)
            else:
                dl = all_readers[task].get_dataloader("test", batch_size=12)
            
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
                    reader = all_readers[task]
                    res = eval_ropes(reader, batch_preds, dl)
                    all_f.write(f"{task} - EM: {res['accuracy']} | F1: {res['f1']}\n")
                    results_map[task] = res
                    acc_scores.append(res["accuracy"])
                    continue
                except:
                    all_f.write(f"{task} - EM: error | F1: error \n")
                    continue
            
            elif task in ["causal_relation", "counterfactual_relation"]:
                reader = all_readers[task]
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
        print(f"len acc scores: {len(acc_scores)}, {acc_scores}, mean: {np.mean(acc_scores)}")
        sum_f.write(f"\n seed: {seed} | learning rate: {lr} -> mean acc {np.mean(acc_scores)} scores: {acc_scores}")
        all_f.close()
        sum_f.close()
        