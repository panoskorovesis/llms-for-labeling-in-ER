import os
from src.utils.utils import load_config

config = load_config("./src/config/config.json")

# Change the default .cache folder to one inside the project folder
os.environ["HF_HOME"] = config["huggingface"]["cache_path"]

# NOTE: This import has to be after the .cache change in order to properly work
from src.embeddings.embeddings import Embedder
from src.utils.enums import Embedding_Models


MODEL = Embedding_Models.JINA_EMBEDDINGS_V5_OMNI_SMALL
# MODEL = "ALL"
D1_RECORDS_PATH = "er_datasets/D2/d1_records.json"
D2_RECORDS_PATH = "er_datasets/D2/d2_records.json"
PAIRS_PATH = "er_datasets/D2/pairs.json"
MAX_EMBEDDING_SIZE = 4096

# "ALL" runs all the available models
if MODEL == "ALL":
    models = [
        Embedding_Models.MINI_LM_L12_V2,
        Embedding_Models.MINI_LM_V6,
        Embedding_Models.EMBER_V1,
        Embedding_Models.STELLA_EN,
        Embedding_Models.BGE_M3,
        Embedding_Models.JASPER_TOKEN_COMPRESSION,
        Embedding_Models.OCTEN_EMBEDDING_4B,
        Embedding_Models.NEMOTRON_3_EMBED_1B,
        Embedding_Models.JINA_EMBEDDINGS_V5_OMNI_SMALL,
        Embedding_Models.KITEFISH_NANO_EM1_06B,
        Embedding_Models.QWEN_3_5_EMBEDDING_4B,
        Embedding_Models.F2LLM_4B,
    ]
else:
    models = [MODEL]

for model in models:
    print(f"RUNNING EMBEDDINGS FOR {model}")

    embedder = Embedder(
        model=model,
        d1_records_path=D1_RECORDS_PATH,
        d2_records_path=D2_RECORDS_PATH,
        pairs_path=PAIRS_PATH,
        verbose=True,
        use_gpu=True,
        max_word_embeddings_size=MAX_EMBEDDING_SIZE,
        use_last_token_pool=True,
        use_task=False,
    )

    embedder.embed(restart=False)

    # Delete the embedder to free up memory
    del embedder
    embedder = None

    print("--------------\n--------------\n--------------\n")

# TODO: https://huggingface.co/Octen/Octen-Embedding-4B
# TODO: https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16
# TODO: https://huggingface.co/KiteFishAI/Nano-Em1-0.6B-v2.1
# TODO: https://huggingface.co/jinaai/jina-embeddings-v5-omni-small
# TODO: https://huggingface.co/Qwen/Qwen3-Embedding-4B

# TODO: WILL THIS FIT?

# TODO: FOR JINA EVALUATE DIFFERENT TASKS !!!!
