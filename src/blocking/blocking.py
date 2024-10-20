import json
import os
import time

import pandas as pd
import pyjedai
import pyjedai.datamodel
import pyjedai.vector_based_blocking
from tqdm import tqdm


class Blocker:
    """The class responsible for running the blocking

    Blocking will be done using the pyjedai package
    The suggestet blocking workflow can be seen here:
    https://pyjedai.readthedocs.io/en/latest/tutorials/pyTorchWorkflow.html

    NOTE: For this research we will run blocking twice:
    1. Using dataset_1 as index
    2. Using dataset_2 as index

    Finally we will keep only the pairs which appear in both produced blocks
    Those will be the final blocks that we are going to use
    """

    def __init__(
        self,
        config: dict,
        dataset_1_path: str,
        dataset_2_path: str,
        ground_truth_path: str,
        csv_separator: str = "|",
        verbose: bool = False,
    ):
        self.config = config
        self.dataset_1_path = dataset_1_path
        self.dataset_2_path = dataset_2_path
        self.ground_truth_path = ground_truth_path
        self.csv_separator = csv_separator
        self.verbose = verbose

        self.d1 = self.load_dataset(self.dataset_1_path)
        self.d2 = self.load_dataset(self.dataset_2_path)
        self.gt = self.load_dataset(self.ground_truth_path)
        self.reverse_gt = self.create_reverse_ground_truth()

        self.dataset_1_attrs = self.get_dataset_attributes(self.d1)
        self.dataset_2_attrs = self.get_dataset_attributes(self.d2)

        print("Datasests loaded successfully!")

        # We will change the default .cache for hugging face folder to one of our chosing
        # Here we have set it to $(pwd)/.cache
        os.environ["HF_HOME"] = self.config["blocking"]["cache_path"]
        if verbose:
            print(f'.cache folder changed to: {self.config["blocking"]["cache_path"]}')

    def load_dataset(self, dataset: str) -> list:
        """Load the given dataset from the given path"""

        df = pd.read_csv(dataset, sep=self.csv_separator, engine="python")
        if self.verbose:
            print(f"Loaded {dataset}.")

        return df

    def get_dataset_attributes(self, dataset: pd.DataFrame):
        """Return the attributes of a dataset

        To get the attributes we will skip the first column
        First column contains the index, which we dont need
        """
        attrs = dataset.columns[1:].to_list()

        if self.verbose:
            print("Attributes extracted for given dataset.")

        return attrs

    def create_reverse_ground_truth(self) -> pd.DataFrame:
        """Create a new gt with the columns reversed

        The original gt has dataset_1, dataset_2
        In order to do the second blocking we need to index with
        dataset_2
        As such, we need a gt with dataset_2, dataset_1
        """
        columns = self.gt.columns
        # reverse the columns
        columns = columns[::-1]

        # crete the reverse dataframe
        reverse_gt = self.gt[columns]

        if self.verbose:
            print(
                f"Created reverse gt with columns: {columns}. Original has columns: {self.gt.columns}"
            )

        return reverse_gt

    def create_data_object(
        self,
        first_dataset,
        second_dataset,
        gt,
        id_column_name_1,
        id_column_name_2,
        attributes_1,
        attributes_2,
    ) -> pyjedai.datamodel.Data:
        """Initialize the pyjedai Data Object using the two datasets

        This object is needed for the blocking workflow
        NOTE: Here, the order of the datasets matters. The first one is used as the index
        and the second one is queried based on the first
        """

        data = pyjedai.datamodel.Data(
            dataset_1=first_dataset,
            attributes_1=attributes_1,
            id_column_name_1=id_column_name_1,
            dataset_2=second_dataset,
            attributes_2=attributes_2,
            id_column_name_2=id_column_name_2,
            ground_truth=gt,
        )

        if self.verbose:
            print("Pyjedai Data item created")

        return data

    def run_blocking(
        self,
        vectorizer: str,
        similarity_search: str,
        top_k: int,
        similarity_distance: str,
        first_dataset,
        second_dataset,
        gt,
        attributes_1,
        attributes_2,
        with_entity_matching: bool = True,
    ):
        """Run the suggested blocking pipeline

        Block building using https://pyjedai.readthedocs.io/en/latest/tutorials/pyTorchWorkflow.html
        Using Embeddings sminilm faiss  search
        faiss.IndexIVFFlat is an implementation of an inverted file index with coarse quantization.
        This index is used to efficiently search for nearest neighbors of a query vector in a large dataset of vectors
        """

        # Create the data object
        data = self.create_data_object(
            first_dataset=first_dataset,
            second_dataset=second_dataset,
            gt=gt,
            id_column_name_1="id",
            id_column_name_2="id",
            attributes_1=attributes_1,
            attributes_2=attributes_2,
        )

        # Create the search index
        emb = pyjedai.vector_based_blocking.EmbeddingsNNBlockBuilding(
            vectorizer=vectorizer, similarity_search=similarity_search
        )

        if self.verbose:
            print("Created EmbeddingsNNBlockBuilding. Will build the blocks")

        # Build the blocks
        blocks, graph = emb.build_blocks(
            data,
            top_k=top_k,
            similarity_distance=similarity_distance,
            load_embeddings_if_exist=False,
            save_embeddings=False,
            verbose=self.verbose,
            with_entity_matching=with_entity_matching,
        )

        if self.verbose:
            print(
                f"Generated {len(blocks.keys())} total blocks. The classification report can be seen bellow."
            )

            results = emb.evaluate(
                blocks, with_classification_report=True, with_stats=True
            )

            print(json.dumps(results, ensure_ascii=True, indent=4))

        # Finally return the blocks and the graph
        return blocks, graph

    def create_final_pairs(self, blocks_1: dict, blocks_2: dict) -> dict:
        """Create the final paris by keeping common matches only

        #TODO: FIND THE ACTUAL CSV IDS BEFORE RUNNING THIS CODE!
        """
        final_blocks = {}

        pairs_1 = set()
        for key in blocks_1:
            for value in blocks_1[key]:
                pairs_1.add((key, value))

        pairs_2 = set()
        for key in blocks_2:
            for value in blocks_2[key]:
                pairs_2.add((value, key))

        if self.verbose:
            print(f"Blocks_1 contain: {len(pairs_1)} pairs")
            print(f"Blocks_2 contain: {len(pairs_2)} pairs")

        # final_pairs = pairs_1.intersection(pairs_2)
        final_pairs = pairs_1 & pairs_2

        return final_pairs

    def run_blocking_workflow(
        self,
        vectorizer: str,
        similarity_search: str,
        top_k: int,
        similarity_distance: str,
        with_entity_matching: bool = True,
    ) -> None:
        """Run the full blocking workflow

        As discussed with @Papadakis the complete blocking process
        is the following:

        1. Run blocking using d1 as index
        2. Run blocking using d2 as index
        3. For each block keep ids that appear IN BOTH stage_blocks

        The final step produces the final blocks. Those will then
        be saved in the appropriate dataset folder using the name
        final_blocks.json
        """
        # STAGE 1 - BLOCKING WITH D1, D2
        print("Will run blocking workflow 1/2")

        start_time = time.time()

        stage_1_blocks, stage_1_graph = self.run_blocking(
            vectorizer=vectorizer,
            similarity_search=similarity_search,
            top_k=top_k,
            similarity_distance=similarity_distance,
            first_dataset=self.d1,
            second_dataset=self.d2,
            gt=self.gt,
            attributes_1=self.dataset_1_attrs,
            attributes_2=self.dataset_2_attrs,
            with_entity_matching=with_entity_matching,
        )

        # STAGE 2 - BLOCKING WITH D2, D1
        print("Will run blocking workflow 2/2")

        stage_2_blocks, stage_2_graph = self.run_blocking(
            vectorizer=vectorizer,
            similarity_search=similarity_search,
            top_k=top_k,
            similarity_distance=similarity_distance,
            first_dataset=self.d2,
            second_dataset=self.d1,
            gt=self.reverse_gt,
            attributes_1=self.dataset_2_attrs,
            attributes_2=self.dataset_1_attrs,
            with_entity_matching=with_entity_matching,
        )

        # finally create the final blocks
        final_blocks = self.create_final_pairs(stage_1_blocks, stage_2_blocks)

        end_time = time.time()

        if self.verbose:
            print(f"Block Creation Total Time: {(end_time - start_time)} seconds")

        return final_blocks
