import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import yaml

def initialize_dataset():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        os.environ["KAGGLE_USERNAME"] = config["kaggle_api"]["username"]
        os.environ["KAGGLE_KEY"] = config["kaggle_api"]["key"]
    api = KaggleApi()
    api.authenticate()
    if not (Path.cwd() / Path("dataset/flowers")).exists():
        api.dataset_download_files("alxmamaev/flowers-recognition", path = "./dataset/", unzip = True)

    