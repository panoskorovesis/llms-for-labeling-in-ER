import os
from src.utils.utils import load_config

config = load_config("./src/config/config.json")

# Change the default .cache folder to one inside the project folder
os.environ["HF_HOME"] = config["huggingface"]["cache_path"]

# NOTE: This import has to be after the .cache change in order to properly work
from src.embeddings.embeddings import Embedder
from src.utils.enums import Embedding_Models


MODEL = Embedding_Models.EMBER_V1
D1_RECORDS_PATH = "er_datasets/D2/d1_records.json"
D2_RECORDS_PATH = "er_datasets/D2/d2_records.json"

embedder = Embedder(
    model=MODEL,
    d1_records_path=D1_RECORDS_PATH,
    d2_records_path=D2_RECORDS_PATH,
    verbose=True,
    use_gpu=True,
)

embedder.embed(restart=False)
