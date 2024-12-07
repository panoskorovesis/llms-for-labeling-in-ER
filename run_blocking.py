import pandas as pd
from pyjedai.datamodel import Data

# Change the default .cache folder to one inside the project folder
import os
os.environ['HF_HOME'] = '/home/papadakis_korovesis/llms-for-labeling-in-ER/.cache'

import src.blocking.blocking as blocking
from src.utils.utils import load_config

import importlib

importlib.reload(blocking)

config = load_config("./src/config/config.json")

blocker = blocking.Blocker(
    config=config,
    dataset_1_path="er_datasets/D3/amazonclean.csv",
    dataset_2_path="er_datasets/D3/gpclean.csv",
    ground_truth_path="er_datasets/D3/gtclean.csv",
    csv_separator='#',
    verbose=True,
)

blocker.run_blocking_workflow(
    vectorizer="sminilm",
    similarity_search="faiss",
    top_k=10,
    similarity_distance="euclidean",
    with_entity_matching=True
)