import datetime
import json
import os

import duckdb
import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel, FlagICLModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.enums import Embedding_Models


class Embedder:
    def __init__(
        self,
        model: Embedding_Models,
        d1_records_path: str,
        d2_records_path: str,
        pairs_path: str,
        verbose: bool = False,
        max_word_embeddings_size: int = 256,
        use_gpu=False,
        use_last_token_pool=False,
        use_task=False,  # Setting this to true will generate embeddings using A task along with the d1 record id
    ):
        # depending on the model create
        # 1) Tokenizer, model
        # 2) model (in case of sentence transformer)

        self.verbose = verbose
        self.e_model = model
        self.max_word_embeddings_size = max_word_embeddings_size
        self.use_last_token_pool = use_last_token_pool
        self.use_task = use_task

        # Set the constant for the table name
        if not self.use_task:
            self.TABLE_NAME = "embeddings_optimal"
        else:
            self.TABLE_NAME = "task_embeddings"

        # Load the jsons
        with open(d1_records_path, "r") as fp:
            self.d1_records = json.load(fp)

        with open(d2_records_path, "r") as fp:
            self.d2_records = json.load(fp)

        with open(pairs_path, "r") as fp:
            self.pairs = json.load(fp)

        if not use_gpu:
            self.device = "cpu"
        # If the user requested a gpu use it if available
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set some examples for BAAI/bge-en-icl
        self.examples = [
            {
                "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
                "query": "what is a virtual interface",
                "response": "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes.",
            },
            {
                "instruct": "Given a web search query, retrieve relevant passages that answer the query.",
                "query": "causes of back pain in female for a week",
                "response": "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management.",
            },
        ]

        # Load the tokenizer and the model
        self.tokenizer, self.model = self.load_tokenizer_and_model(model)

        # initialize the db connection
        self.con = self.initialize_duckdb_connection(d1_records_path=d1_records_path)

        if self.verbose:
            print(f"Will run on {self.device}")
            print("Embedder initialized")

    def load_tokenizer_and_model(self, model: Embedding_Models) -> tuple:
        """Load the requried tokenizer and model

        For the traditional models we need the tokenizer and the model
        To keep our code simple we use the AutoTokenizer, AutoModel methods

        For the sentence transformers we dont need a tokenizer
        We set it to null so the following methods know what to do
        """
        traditional_models = {
            Embedding_Models.ROBERTA_LARGE,
            Embedding_Models.QWEN_2_5_7B,
            Embedding_Models.E5_MISTRAL_7B,
            Embedding_Models.PHI_3,
        }

        sentence_transformers = {
            Embedding_Models.EMBER_V1,
            Embedding_Models.MINI_LM_V6,
            Embedding_Models.STELLA_EN,
            Embedding_Models.MINI_LM_L12_V2,
            Embedding_Models.SFR_EMBEDDING_MISTRAL,
            Embedding_Models.GTE_QWEN2,
            Embedding_Models.OCTEN_EMBEDDING_4B,
        }

        jasper_models = {
            Embedding_Models.JASPER_TOKEN_COMPRESSION,
        }

        # Models from Beijing Academy of Artificial Intelligence require special handling (BAAI)
        # They use their own library, not sentence transformers
        baai_models = {Embedding_Models.BGE_M3}

        # One variation also requires the FlagICLModel import
        baai_icl_models = {Embedding_Models.BGE_EN_ICL}

        if model in traditional_models:
            tokenizer = AutoTokenizer.from_pretrained(str(model))
            emb_model = AutoModel.from_pretrained(str(model))

        elif model in sentence_transformers:
            tokenizer = None
            # NOTE: SentenceTransformer will go to gpu by default if it's available
            # Some of our models do not fit in gpu so we have to specify the device here
            # Later the model.to() will essentially do nothing but that's ok!
            emb_model = SentenceTransformer(
                str(model),
                device=self.device,
                trust_remote_code=True,
            )
        elif model in jasper_models:
            print(f"Loading Jasper model: {str(model)}")
            tokenizer = None
            emb_model = SentenceTransformer(
                str(model),
                device=self.device,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "attn_implementation": "sdpa",
                },
                trust_remote_code=True,
            )
        elif model in baai_models:
            tokenizer = None
            emb_model = BGEM3FlagModel(str(model), use_fp16=True)
        elif model in baai_icl_models:
            tokenizer = None
            emb_model = FlagICLModel(
                str(model),
                use_fp16=True,
                devices=["cpu"],
                query_instruction_for_retrieval="Retrieve semantically similar text.",
                examples_for_task=self.examples,
            )
        else:
            raise ValueError(f"Model: {str(model)} is not currently supported!")

        return tokenizer, emb_model

    def initialize_duckdb_connection(self, d1_records_path: str):
        """Initialize a duckdb connection

        The db will be saved in the same folder as the pair.json file
        The name will be results.duckdb

        This db will be used to keep track of time and the resuls of the process
        """

        # Keep everything up until the last slash
        db_path = d1_records_path.rsplit("/", maxsplit=1)[0]
        # Add the db name
        db_path += "/results.duckdb"

        con = duckdb.connect(db_path)

        if self.verbose:
            print(
                f"Duck db connection initialized! Tables: \n{con.sql('SHOW TABLES;')}"
            )

        return con

    def create_db_table_if_needed(self, force=False):
        """Create a table to store the embeddings in duckdb

        If the force is set to True we will delete the table and recreate it
        If the force is set to False, we will create the table only if it's not created
        """

        if force:
            self.con.sql(f"""DROP TABLE IF EXISTS {self.TABLE_NAME}""")

            if self.verbose:
                print(f"Table {self.TABLE_NAME} was deleted. It will be recreated.")

        # If we dont have a task we look at the embeddings table
        if not self.use_task:
            # This table will be empty
            # Columns will be added on the fly containing the name of the model that generated the embeddings
            self.con.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                record_id VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                embeddings FLOAT[] NOT NULL,
                created_at TIMESTAMP NOT NULL,
                model VARCHAR NOT NULL
            );
            """)
        # if we have a task we look at task_embeddings
        else:
            # Also create the table for the instruct embeddings
            # Field explanation:
            # d1_reference_id: The d1_id which can be found as a key in the pairs.json
            # record_id: The id the embeddings bellong to. THis can be from d1 or d2
            # is_d2:
            #   True --> the record_id reffers to d2
            #   False --> the record_id reffers to d1
            self.con.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                d1_reference_id VARCHAR NOT NULL,
                record_id VARCHAR NOT NULL,
                is_d2 BOOLEAN NOT NULL,
                embeddings FLOAT[] NOT NULL,
                created_at TIMESTAMP NOT NULL,
                model VARCHAR NOT NULL
            );
            """)

    def get_ids_and_texts(self, data: dict):
        """Separate the dict data into two lists
        One with the keys and another with the values
        """
        return list(data.keys()), list(data.values())

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Create the task description for the LLM

        This will be used in the embeddings generation method that requires
        1) The d1 record
        2) All d2 candidate records

        This applies only to the method that DOES NOT use sentence transformers
        """
        return f"Instruct: {task_description}\nQuery: {query}"

    def tokenize_data(self, texts: list) -> tuple:
        """Tokenize the data using the tokenizer

        Return the input ids and the masks
        """

        # Tokenize using batch_encode_plus method
        # This will return a json with input_ids and the attention masks
        # encoding = self.tokenizer.batch_encode_plus(
        #     texts,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_word_embeddings_size,
        #     return_tensors="pt",  # Return PyTorch tensors
        #     add_special_tokens=True,  # Add special tokens CLS and SEP
        # )

        # If we do not include a task we know that the texts list contains only
        # one element
        # Otherwise, it contains the task, query (which is the d1 record) and all relevant d2 records
        if not self.use_task:
            texts = texts[0]

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_word_embeddings_size,
            return_tensors="pt",  # Return PyTorch tensors
        )

        # input_ids are numerical representations of the tokenized input text.
        # Each token in the input text is mapped to a unique ID from the model's vocabulary

        # Move the input_ids and the mask to device
        # This makes things faster if gpu is available
        input_ids = encoding["input_ids"]
        input_ids = input_ids.to(self.device)

        attention_mask = encoding["attention_mask"]
        attention_mask = attention_mask.to(self.device)

        return input_ids, attention_mask

    def get_completed_embeddings(
        self, model: Embedding_Models, dataset_name: str
    ) -> set:
        """Get the processed records from duckdb

        We will get the pairs and return them as a set
        for better indexing.
        This will allow the code to continue the execution from where
        it stopped

        Returns
        -------
            set: The processed pairs
        """
        processed_pairs = set()

        if not self.use_task:
            records = self.con.sql(f"""
            SELECT DISTINCT record_id
            from {self.TABLE_NAME}
            where model = '{model}' and dataset = '{dataset_name}'
            """).fetchall()
        # If we are using a task we have to take the reference id
        else:
            records = self.con.sql(f"""
            SELECT DISTINCT d1_reference_id
            from {self.TABLE_NAME}
            where model = '{model}'
            """).fetchall()

        # NOTE: Fetchall will return these in a tuple.
        # For example (444, )
        # We want to keep only the ids and return them

        processed_pairs = {record[0] for record in records}

        return processed_pairs

    def save_embeddings_to_db(
        self,
        record_id: str,
        embeddings: np.array,
        model: Embedding_Models,
        dataset_name: str,
    ) -> None:
        """Perform one insertion to the db

        This method will insert a record_id along with it's embeddings and the model that produced them
        """
        self.con.execute(
            f"""
        INSERT INTO {self.TABLE_NAME} (record_id, dataset, embeddings, created_at, model) 
        VALUES(?, ?, ?, ?, ?)
        """,
            [
                record_id,
                dataset_name,
                embeddings.tolist(),
                datetime.datetime.now(),
                str(model),
            ],
        )

    def save_embeddings_to_db_bulk(
        self,
        embeddings: dict,
        model: Embedding_Models,
        dataset_name: str,
    ):
        # The first method is to create a dataframe and then insert it
        # The second is to perform a bulk insert

        # NOTE: Embeddings is a dict with the following structure
        # {
        #     "record_id": {
        #         "embeddings": np.array,
        #         "created_at": datetime.datetime.now(),
        #     }
        # }

        # Create a dataframe
        # Do not change the index since it's the record_id
        df = pd.DataFrame.from_dict(embeddings, orient="index")

        # Add the missing columns
        df["record_id"] = df.index
        df["dataset"] = dataset_name
        df["model"] = str(model)

        print(df.info())

        # Insert the dataframe into the db
        self.con.execute(
            f"""
            INSERT INTO {self.TABLE_NAME} SELECT record_id, dataset, embeddings, created_at, model FROM df
            """
        )

    def save_task_embeddings_to_db(
        self,
        record_id: str,
        d1_reference_id: str,
        embeddings: np.array,
        model: Embedding_Models,
        is_d2: bool,
    ):
        """Perform one insertion to the db

        This method will insert one task embedding to the db
        """
        self.con.execute(
            f"""
            INSERT INTO {self.TABLE_NAME} (d1_reference_id, record_id, is_d2, embeddings, created_at, model)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            [
                d1_reference_id,
                record_id,
                is_d2,
                embeddings.tolist(),
                datetime.datetime.now(),
                str(model),
            ],
        )

    def last_token_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def generate_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        """Generate the embeddings using the model

        To get the whole text embeddings we will use the mean() of the words
        """
        # give the data to the model as input
        with torch.no_grad():
            ouputs = self.model(input_ids, attention_mask=attention_mask)

        if not self.use_last_token_pool:
            # The embeddings are in the last_hidden_state with a shape of
            # torch.Size([1, x, y]), x: The number of tokens (words), y the size of the produced embeddings
            # to get a sentence representation we will do a mean()
            text_embeddings = ouputs.last_hidden_state
            text_embeddings = text_embeddings.mean(dim=1)
        # Another way is proposed here
        # https://huggingface.co/intfloat/e5-mistral-7b-instruct
        # This keeps only the last layer, removing padding
        else:
            text_embeddings = self.last_token_pool(
                ouputs.last_hidden_state, attention_mask=attention_mask
            )
        # then we will convert to numpy from tensors
        # if we are in cuda, copy the tensor to cpu first
        if str(self.device) == "cuda":
            text_embeddings = text_embeddings.cpu()
        text_embeddings = text_embeddings.detach().numpy()
        """
        GIVEN TASK = FALSE
        here the shape is (1, y)
        to make things easier for further calculations we will flatten
        the final shape will be (y,)

        GIVEN TASK = TRUE
        here the shape will be (x, y)
            x: The number of candidates + 1 (for the d1_id)
            y: The embeddings length

        In this case we DONT WANT TO FLATTEN
        """
        if not self.use_task:
            text_embeddings = text_embeddings.flatten()

        # return them
        return text_embeddings

    def embed_word_tokens(self, records: list, dataset_name: str):
        """Generate embeddings using word tokens

        This method will be used with models that are not! sentence transformers
        The final embeddings will be the mean of all the word embeddings
        """

        # First split the dict into ids and texts
        ids, texts = self.get_ids_and_texts(records)

        completed = self.get_completed_embeddings(
            model=self.e_model, dataset_name=dataset_name
        )

        # do a loop and generate the embeddings one by one
        for idx, text in tqdm(
            zip(ids, texts),
            total=len(texts),
            desc=f"Generating embeddings for {dataset_name}",
        ):
            # if the id is in the completed skip
            if idx in completed:
                if self.verbose:
                    print(
                        f"Embeddings are already generated for {idx} - {self.e_model}. Will continue"
                    )
                continue

            # tokenize the data
            input_ids, attention_mask = self.tokenize_data(texts=[text])

            # generate the embeddings
            embeddings = self.generate_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # save them to the db
            self.save_embeddings_to_db(
                idx,
                embeddings=embeddings,
                model=self.e_model,
                dataset_name=dataset_name,
            )

    def generate_sentence_embeddings(self, text: str):
        """Generate the embeddings for one sentence

        This will be done using the .encode functionality
        No checks for gpu are needed here since encode()
        returns a numpy array

        BAAI models need special handling for the embedding generation
        """

        # the shape will be (1, embeddings_size)
        # We want to get rid of the 1, so we flatten
        if self.e_model == Embedding_Models.BGE_M3:
            embeddings = self.model.encode(
                [text],
                max_length=8192,  # This is the proposed lengh. We can make it smaller if we want
            )["dense_vecs"]
        elif self.e_model == Embedding_Models.BGE_EN_ICL:
            # TODO: Add a case here for use_task and not use_task
            # Link: https://huggingface.co/BAAI/bge-en-icl
            embeddings = self.model.encode(
                [text],
                max_length=8192,  # This is the proposed lengh. We can make it smaller if we want
            )
        else:
            # NOTE: Here the input can either be str or a list
            # If it's str we want to add a list
            # str should come for use_task=False
            # otherwise it should be a list
            # if isinstance(text, str):
            # text = [text]

            embeddings = self.model.encode(text)
        # here the shape is (1, embeddings_size) for use_task = False
        # and (N, embeddings_size) otherwise
        # to make things easier for further calculations we will flatten
        # ONLY FOR use_task = False
        # the final shape will be (embeddings_size,)
        if not self.use_task and embeddings.shape[0] == 1:
            embeddings = embeddings.flatten()

        return embeddings

    def embed_sentences(self, records: list, dataset_name: str):
        """Generate embeddings for the sentences using setence transformers"""
        # First split the dict into ids and texts
        ids, texts = self.get_ids_and_texts(records)

        completed = self.get_completed_embeddings(
            model=self.e_model, dataset_name=dataset_name
        )

        record_embeddings = {}

        # do a loop and generate the embeddings one by one
        for i in tqdm(
            range(len(ids)),
            total=len(texts),
            desc=f"Generating embeddings for {dataset_name}",
        ):
            idx = ids[i]
            text = texts[i]

            # if the id is in the completed skip
            if idx in completed:
                if self.verbose:
                    print(
                        f"Embeddings are already generated for {idx} - {self.e_model}. Will continue"
                    )
                continue

            embeddings = self.generate_sentence_embeddings(text=text)

            # Initialize an empty dict for the record
            record_embeddings[idx] = {}
            # Fill the dict with the embeddings and creation timestamp
            record_embeddings[idx]["embeddings"] = embeddings
            record_embeddings[idx]["created_at"] = datetime.datetime.now()

        # If all the embeddings are already generated return
        if len(record_embeddings) == 0:
            if self.verbose:
                print(
                    f"All embeddings are already generated for {dataset_name} - {self.e_model}. Will continue"
                )
            return

        # Save the embeddings
        self.save_embeddings_to_db_bulk(
            embeddings=record_embeddings,
            model=self.e_model,
            dataset_name=dataset_name,
        )

        # # No go through the dict and save the embeddings
        # for idx, embeddings in tqdm(
        #     record_embeddings.items(),
        #     total=len(record_embeddings),
        #     desc=f"Saving embeddings for {dataset_name}",
        # ):
        #     # TODO: MAKE THIS FASTER && BATCH INSERT
        #     # save them to the db
        #     self.save_embeddings_to_db(
        #         idx,
        #         embeddings=embeddings,
        #         model=self.e_model,
        #         dataset_name=dataset_name,
        #     )

    def embed_word_tokens_with_task(
        self, d1_records: dict, d2_records: dict, pairs: dict
    ):
        """Generate embeddings using the AutoModel and an additional task

        This is the method proposed here: https://huggingface.co/intfloat/e5-mistral-7b-instruct

        We will do this in batches. Each batch contains the
        1. d1 record (which is the TASK)
        2. All relevant d2_records (which are the documents)

        Due to the fact that d2 documents can appear multiple times for

        The task description will be
        Retrieve semantically similar text.
        OR
        Instruct: Given a web search query, retrieve relevant passages that answer the query
        """

        TASK_DESCRIPTION = "Given a query, retrieve semantically similar text."

        # We dont need a dataset name here since each db has one of it's own
        completed_ids = self.get_completed_embeddings(
            model=self.e_model, dataset_name=None
        )

        # For each pair in the self.pairs dictionary
        for d1_id, d2_candidate_ids in tqdm(
            pairs.items(), total=len(pairs), desc="Generating Task Embeddings"
        ):
            if d1_id in completed_ids:
                if self.verbose:
                    print(
                        f"Task Embeddings already generated for {d1_id}, {self.e_model}"
                    )
                continue

            # Locate the text for d1
            # Also apply the instruction
            d1_text = self.get_detailed_instruct(
                task_description=TASK_DESCRIPTION, query=d1_records[d1_id]
            )
            # Locate the texts for d2
            d2_texts = [d2_records[d2_id] for d2_id in d2_candidate_ids]

            # At this point we want to merge the two keeping the d1_text at the top
            input_texts = [d1_text] + d2_texts

            # Tokenize the data using the max_word_embeddings_size defined at the constructor
            input_ids, attention_mask = self.tokenize_data(input_texts)

            # Now generate the embeddings
            embeddings = self.generate_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Finally we have to add the embeddings to the db table
            for idx, emb in enumerate(embeddings):
                # if idx == 0 then we are talking about the d1 record
                if idx == 0:
                    self.save_task_embeddings_to_db(
                        record_id=d1_id,
                        d1_reference_id=d1_id,
                        is_d2=False,
                        embeddings=embeddings[idx][:],
                        model=self.e_model,
                    )
                else:
                    self.save_task_embeddings_to_db(
                        # NOTE: We have to add -1 here since we are counting from 0
                        record_id=d2_candidate_ids[idx - 1],
                        d1_reference_id=d1_id,
                        is_d2=True,
                        embeddings=embeddings[idx][:],
                        model=self.e_model,
                    )

    def embed_sentences_with_task(
        self, d1_records: dict, d2_records: dict, pairs: dict
    ):
        """Generate embeddings using the AutoModel and an additional task

        This is the method proposed here: https://huggingface.co/intfloat/e5-mistral-7b-instruct

        We will do this in batches. Each batch contains the
        1. d1 record (which is the TASK)
        2. All relevant d2_records (which are the documents)

        Due to the fact that d2 documents can appear multiple times for

        The task description will be
        Retrieve semantically similar text.
        OR
        Instruct: Given a web search query, retrieve relevant passages that answer the query
        """

        TASK_DESCRIPTION = "Given a query, retrieve semantically similar text."

        # We dont need a dataset name here since each db has one of it's own
        completed_ids = self.get_completed_embeddings(
            model=self.e_model, dataset_name=None
        )

        # For each pair in the self.pairs dictionary
        for d1_id, d2_candidate_ids in tqdm(
            pairs.items(), total=len(pairs), desc="Generating Task Embeddings"
        ):
            if d1_id in completed_ids:
                if self.verbose:
                    print(
                        f"Task Embeddings already generated for {d1_id}, {self.e_model}"
                    )
                continue

            # Locate the text for d1
            # Also apply the instruction
            d1_text = self.get_detailed_instruct(
                task_description=TASK_DESCRIPTION, query=d1_records[d1_id]
            )
            # Locate the texts for d2
            d2_texts = [d2_records[d2_id] for d2_id in d2_candidate_ids]

            # At this point we want to merge the two keeping the d1_text at the top
            input_texts = [d1_text] + d2_texts

            # Now generate the embeddings
            embeddings = self.generate_sentence_embeddings(input_texts)

            # Finally we have to add the embeddings to the db table
            for idx, emb in enumerate(embeddings):
                # if idx == 0 then we are talking about the d1 record
                if idx == 0:
                    self.save_task_embeddings_to_db(
                        record_id=d1_id,
                        d1_reference_id=d1_id,
                        is_d2=False,
                        embeddings=embeddings[idx][:],
                        model=self.e_model,
                    )
                else:
                    self.save_task_embeddings_to_db(
                        # NOTE: We have to add -1 here since we are counting from 0
                        record_id=d2_candidate_ids[idx - 1],
                        d1_reference_id=d1_id,
                        is_d2=True,
                        embeddings=embeddings[idx][:],
                        model=self.e_model,
                    )

    def embed(self, restart: bool = False):
        # First create the table and the column if needed
        self.create_db_table_if_needed(force=restart)

        # change the model to the device
        if self.verbose:
            print(f"Switching model to: {self.device}")

        # This applied to all except BAAI Models
        if (
            self.e_model != Embedding_Models.BGE_M3
            and self.e_model != Embedding_Models.BGE_EN_ICL
        ):
            self.model = self.model.to(self.device)

        # If it's a traditional model then the self.tokenizer attribute wont be null
        if self.tokenizer is not None and not self.use_task:
            # generate embeddings for d1
            self.embed_word_tokens(records=self.d1_records, dataset_name="d1")
            # generate embeddings for d2
            self.embed_word_tokens(records=self.d2_records, dataset_name="d2")

        elif self.tokenizer is not None and self.use_task:
            # generate embeddings
            self.embed_word_tokens_with_task(
                d1_records=self.d1_records, d2_records=self.d2_records, pairs=self.pairs
            )
        # if the tokenizer is none then we have a sentence transformer
        elif not self.use_task:
            # generate embeddings for d1
            self.embed_sentences(records=self.d1_records, dataset_name="d1")
            # generate embeddings for d2
            self.embed_sentences(records=self.d2_records, dataset_name="d2")

        # If a task is required for the sentence transformers
        else:
            # generate embeddings
            self.embed_sentences_with_task(
                d1_records=self.d1_records, d2_records=self.d2_records, pairs=self.pairs
            )
