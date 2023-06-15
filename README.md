# Overview
This repo contains the anonymized code for the EACL long paper submission `CALM-Bench: A Multi-task Benchmark for Evaluating Causality Aware Language Models`. 

The repo is organzized as follows:
- config/: contains the hyperparameters for the BERT and ERNIE single task models
- src/: source code for all experimentse 


Note the underlying data for the CALM-Bench tasks is not included. All benchmark tasks
are available publicly. See paper for further details. 


# Running Experiments 
The sections below contain the variou scripts to recreate the experiments in the paper. Please
note the scripts need to be manaually modified to specific which base language model is being
used. 

## Baseline Models 
To generate all the baseline models run `generate_baselines.py`. 

## Transfer Learning
To generate all the transfer learning experiment run `generate_transfer_baselines.py`

## MTL Training and evaluation 
To train and evaluate the MTL model run `run_mtl_finetune_exp.py`