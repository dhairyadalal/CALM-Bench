import pickle 
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

TASK_LIST = [
                "causal_identification", 
                "causal_relation",
                "counterfactual_identification",
                "counterfactual_relation",
                "copa",
                "cosmosqa",
                'ecare',
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
                "ecare": ECAREreader,
                "wiqa": WIQAReader, 
                "ropes": RopesReader,
                "anli": aNLIMCreader
            }


def initialize_readers(model_alias: str):
    
    initialized_reader_map = {}
    
    for task in TASK_LIST:
        
        batch_size = 24 
        
        print(f"generating baseline for {model_alias} - {task}")
        reader = READER_MAP[task](model_alias = model_alias)
        train_dl = reader.get_dataloader(split="train", batch_size=batch_size)
        val_dl = reader.get_dataloader(split="val", batch_size=batch_size) 
            
        initialized_reader_map[task] = {"reader": reader}
        initialized_reader_map[task]["train"] = train_dl
        initialized_reader_map[task]["val"] = val_dl
    
        if task in [
            "causal_identification", 
            "causal_relation", 
            "counterfactual_identification",
            "counterfactual_relation",
            "copa", 
            "wiqa"
        ]:         
            test_dl = reader.get_dataloader(split="test", batch_size=batch_size) 
            initialized_reader_map[task]["test"] = test_dl
    
    return initialized_reader_map



def main():
    
    model_aliases = [
                "bert-base-uncased", 
                "nghuyong/ernie-2.0-base-en"
            ]    
    
    for alias in model_aliases:
        initialized_reader = initialize_readers(alias)
        pickle.dump(initialized_reader, open(f"{alias}-readers.pkl", "wb"))
             
      
if __name__ == "__main__":
    main()


