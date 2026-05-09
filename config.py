from pathlib import Path

def get_config():
    return{
        "batch_size": 256,
        "num_epochs": 30,
        "lr": 1e-5,
        "seq_len": 80,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "si",
        "model_folder": "weights",
        "model_base_name": "transformer",
        "preload": 19,
        "tokenizer_file": "../tokenizers/tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config,epoch:str):
    model_folder = config['model_folder']
    model_base_name = config['model_base_name']
    model_file_name = f"{model_base_name}{epoch}.pt"

    return str(Path('.') / model_folder/model_file_name)