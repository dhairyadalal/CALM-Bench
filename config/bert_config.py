TASK_CONFIG = {
        "causal_identification": {"seed": 42, "learning_rate": 5e-5, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "causal_identification"}, 
        "causal_relation": {"seed": 1, "learning_rate": 5e-5, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "causal_relation"},
        "counterfactual_identification": {"seed": 1, "learning_rate": 1e-5, "training_epochs": 4, "model_alias": "bert-base-uncased", "task": "counterfactual_identification"},
        "counterfactual_relation": {"seed": 3023, "learning_rate": 5e-05, "training_epochs": 4, "model_alias": "bert-base-uncased", "task": "counterfactual_relation"},
        "copa": {"seed": 1, "learning_rate": 1e-5, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "copa"},
        "cosmosqa": {"seed": 3023, "learning_rate": 2e-05, "training_epochs": 1, "model_alias": "bert-base-uncased", "task": "cosmosqa"} ,
        "wiqa": {"seed": 1988, "learning_rate": 2e-05, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "wiqa"},
        "ropes": {"seed": 0, "learning_rate": 5e-05, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "ropes"},
        "anli": {"seed": 3023, "learning_rate": 2e-05, "training_epochs": 1, "model_alias": "bert-base-uncased", "task": "anli"},
        "ecare": {"seed": 42, "learning_rate": 2e-05, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "ecare"}
}

# Bert base cased 
TASK_CONFIG = {
        "causal_identification": {"seed": 1988, "learning_rate": 1e-4, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "causal_identification"}, 
        "causal_relation": {"seed": 3023, "learning_rate": 1e-4, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "causal_relation"},
        "counterfactual_identification": {"seed": 0, "learning_rate": 5e-5, "training_epochs": 4, "model_alias": "bert-base-uncased", "task": "counterfactual_identification"},
        "counterfactual_relation": {"seed": 0, "learning_rate": 1e-4, "training_epochs": 4, "model_alias": "bert-base-uncased", "task": "counterfactual_relation"},
        "copa": {"seed": 42, "learning_rate": 2e-5, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "copa"},
        "cosmosqa": {"seed": 1988, "learning_rate": 5e-05, "training_epochs": 1, "model_alias": "bert-base-uncased", "task": "cosmosqa"} ,
        "wiqa": {"seed": 0, "learning_rate": 5e-05, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "wiqa"},
        "ropes": {"seed": 0, "learning_rate": 3e-4, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "ropes"},
        "anli": {"seed": 0, "learning_rate": 3e-05, "training_epochs": 1, "model_alias": "bert-base-uncased", "task": "anli"},
        "ecare": {"seed": 0, "learning_rate": 5e-05, "training_epochs": 3, "model_alias": "bert-base-uncased", "task": "ecare"}
}