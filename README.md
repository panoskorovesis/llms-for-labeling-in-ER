# Leveraging LLMs for Entiry Resolution

## Prompting

The first method relies _solely_ on prompting.

We used the prompts proposed at CoEM as a baseline and then finetunned them for faster inferencing.

__We contribute a benchmark with the CoEM and our prompts on several established Entity Resolution Datasets__, using different open source models.

The complete list is as follows:

1. Gemma2-9b
1. Phi3.5-3.8B
1. Phi4-14B
1. Qwen2.5-14B
1. Falcon3-10B

## Embeddings

The second method relies on embeddings.

We used:

1. different SOTA embeddings models, many of which can be found on the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
2. Some of the decoder only LLMs from the first method, in order to study their performance at the embedding generation task

__We contribute a embeddings benchmark on several established Entity Resolution Datasets__, using different open source models.

The complete list is as follows:

__Embedding Models__:

1. dunzhang/stella_en_1.5B_v5
1. llmrails/ember-v1
1. BAAI/bge-m3
1. BAAI/bge-multilingual-gemma2
1. BAAI/bge-en-icl
1. sentence-transformers/all-MiniLM-L6-v2
1. sentence-transformers/all-MiniLM-L12-v2
1. intfloat/e5-mistral-7b-instruct
1. Salesforce/SFR-Embedding-Mistral
1. google/Gemma-Embeddings-v1.0

__LLMs__:

1. Qwen/Qwen2.5-7B-Instruct
2. google/gemma-2-9b-it