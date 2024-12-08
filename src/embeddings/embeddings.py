import datetime
import json
import os

import duckdb
import numpy as np
import pandas as pd
import torch
from FlagEmbedding import BGEM3FlagModel
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
        verbose: bool = False,
        max_word_embeddings_size: int = 256,
        use_gpu=False,
    ):
        # depending on the model create
        # 1) Tokenizer, model
        # 2) model (in case of sentence transformer)

        self.verbose = verbose
        self.e_model = model
        self.max_word_embeddings_size = max_word_embeddings_size

        # Set the constant for the table name
        self.TABLE_NAME = "embeddings"

        # Load the jsons
        with open(d1_records_path, "r") as fp:
            self.d1_records = json.load(fp)

        with open(d2_records_path, "r") as fp:
            self.d2_records = json.load(fp)

        if not use_gpu:
            self.device = "cpu"
        # If the user requested a gpu use it if available
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        }

        sentence_transformers = {
            Embedding_Models.EMBER_V1,
            Embedding_Models.MINI_LM_V6,
            Embedding_Models.STELLA_EN,
            Embedding_Models.MINI_LM_L12_V2,
            Embedding_Models.E5_MISTRAL_7B,
        }

        # Models from Beijing Academy of Artificial Intelligence require special handling (BAAI)
        # They use their own library, not sentence transformers
        baai_models = {Embedding_Models.BGE_M3}

        if model in traditional_models:
            tokenizer = AutoTokenizer.from_pretrained(
                str(Embedding_Models.ROBERTA_LARGE)
            )
            emb_model = AutoModel.from_pretrained(str(Embedding_Models.ROBERTA_LARGE))

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
        elif model in baai_models:
            tokenizer = None
            emb_model = BGEM3FlagModel(str(model), use_fp16=True)
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
                f'Duck db connection initialized! Tables: \n{con.sql("SHOW TABLES;")}'
            )

        return con

    def create_db_table_if_needed(self, table_name: str, force=False):
        """Create a table to store the embeddings in duckdb

        If the force is set to True we will delete the table and recreate it
        If the force is set to False, we will create the table only if it's not created
        """

        if force:
            self.con.sql(f"""DROP TABLE IF EXISTS {table_name}""")

            if self.verbose:
                print(f"Table {table_name} was deleted. It will be recreated.")

        # This table will be empty
        # Columns will be added on the fly containing the name of the model that generated the embeddings
        self.con.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            record_id VARCHAR NOT NULL,
            dataset VARCHAR NOT NULL,
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

    def tokenize_data(self, texts: list) -> tuple:
        """Tokenize the data using the tokenizer

        Return the input ids and the masks
        """

        # Tokenize using batch_encode_plus method
        # This will return a json with input_ids and the attention masks
        encoding = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_word_embeddings_size,
            return_tensors="pt",  # Return PyTorch tensors
            add_special_tokens=True,  # Add special tokens CLS and SEP
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

        records = self.con.sql(f"""
        SELECT DISTINCT record_id
        from {self.TABLE_NAME}
        where model = '{model}' and dataset = '{dataset_name}'
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

    def generate_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        """Generate the embeddings using the model

        To get the whole text embeddings we will use the mean() of the words
        """
        # give the data to the model as input
        with torch.no_grad():
            ouputs = self.model(input_ids, attention_mask=attention_mask)

        # The embeddings are in the last_hidden_state with a shape of
        # torch.Size([1, x, y]), x: The number of tokens (words), y the size of the produced embeddings
        # to get a sentence representation we will do a mean()
        text_embeddings = ouputs.last_hidden_state
        text_embeddings = text_embeddings.mean(dim=1)
        # then we will convert to numpy from tensors
        # if we are in cuda, copy the tensor to cpu first
        if str(self.device) == "cuda":
            text_embeddings = text_embeddings.cpu()
        text_embeddings = text_embeddings.detach().numpy()
        # here the shape is (1, y)
        # to make things easier for further calculations we will flatten
        # the final shape will be (y,)
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
                input_ids=input_ids, attention_mask=attention_mask
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
        else:
            embeddings = self.model.encode([text])
        # here the shape is (1, embeddings_size)
        # to make things easier for further calculations we will flatten
        # the final shape will be (embeddings_size,)
        embeddings = embeddings.flatten()

        return embeddings

    def embed_sentences(self, records: list, dataset_name: str):
        """Generate embeddings for the sentences using setence transformers"""
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

            embeddings = self.generate_sentence_embeddings(text=text)

            # save them to the db
            self.save_embeddings_to_db(
                idx,
                embeddings=embeddings,
                model=self.e_model,
                dataset_name=dataset_name,
            )

    def embed(self, restart: bool = False):
        # First create the table and the column if needed
        self.create_db_table_if_needed(table_name=self.TABLE_NAME, force=restart)

        # change the model to the device
        if self.verbose:
            print(f"Switching model to: {self.device}")

        # This applied to all except BAAI Models
        if self.e_model != Embedding_Models.BGE_M3:
            self.model = self.model.to(self.device)

        # If it's a traditional model then the self.tokenizer attribute wont be null
        if self.tokenizer is not None:
            # generate embeddings for d1
            self.embed_word_tokens(records=self.d1_records, dataset_name="d1")
            # generate embeddings for d2
            self.embed_word_tokens(records=self.d2_records, dataset_name="d2")

        # if the tokenizer is none then we have a sentence transformer
        else:
            # generate embeddings for d1
            self.embed_sentences(records=self.d1_records, dataset_name="d1")
            # generate embeddings for d2
            self.embed_sentences(records=self.d2_records, dataset_name="d2")
