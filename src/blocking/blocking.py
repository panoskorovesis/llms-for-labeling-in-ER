import json
import os
import time
from collections import defaultdict

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
        self.dataset_1_path = dataset_1_path
        self.dataset_2_path = dataset_2_path

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
        return blocks, graph, data

    def create_final_pairs(
        self,
        blocks_1: dict,
        data1: pyjedai.datamodel.Data,
        blocks_2: dict,
        data2: pyjedai.datamodel.Data,
    ) -> dict:
        """Create the final paris by keeping common matches only"""

        # The data object contains dictionaries that maps the original csv ids to the
        # arbitrary ones found in blocks.
        # In order to create the pairs with the csv ids, we have to create the reverse of
        # those dicts

        blocking_id_to_csv_1 = {
            value: key for key, value in data1._ids_mapping_1.items()
        }
        blocking_id_to_csv_2 = {
            value: key for key, value in data1._ids_mapping_2.items()
        }

        pairs_1 = set()
        for key in blocks_1:
            for value in blocks_1[key]:
                original_key_id = blocking_id_to_csv_1[key]

                original_value_id = blocking_id_to_csv_2[value]

                pairs_1.add((original_key_id, original_value_id))

        # Do the same for the new data object
        blocking_id_to_csv_1 = {
            value: key for key, value in data2._ids_mapping_1.items()
        }
        blocking_id_to_csv_2 = {
            value: key for key, value in data2._ids_mapping_2.items()
        }

        pairs_2 = set()
        for key in blocks_2:
            for value in blocks_2[key]:
                original_key_id = blocking_id_to_csv_1[key]
                original_value_id = blocking_id_to_csv_2[value]

                # NOTE: Here we have to change the order since the documents
                # are given in opposite order ie the keys are the gt.D2 documents
                # and the values the gt.D1
                pairs_2.add((original_value_id, original_key_id))

        if self.verbose:
            print(f"Blocks_1 contain: {len(pairs_1)} pairs")
            print(f"Blocks_2 contain: {len(pairs_2)} pairs")

        # final_pairs = pairs_1.intersection(pairs_2)
        final_pairs = pairs_1 & pairs_2

        return final_pairs

    def save_records_of_interest(
        self,
        dataset: pd.DataFrame,
        attributes: list,
        id_col: str,
        ids_of_interest: dict,
        filename: str,
    ) -> None:
        """
        Save records of interest from the given dataset as a dictionary.

        This method converts the input DataFrame into string format and then creates records for ids present in parsed_pairs.
        The resulting records are saved as a JSON file.

        Args:
            parsed_pairs (dict): A dictionary where keys are ids and values are lists of related items.

        Returns:
            None
        """

        # Convert df into str format
        str_dataset = dataset.astype(str)

        records = {}
        for record_id in tqdm(
            ids_of_interest,
            desc=f"Constructing records {filename}",
            total=len(ids_of_interest),
        ):
            try:
                # Get the one record from the dataframe
                record = str_dataset[str_dataset[id_col] == str(record_id)].iloc[0]
                # Keep only the columns we are interested in
                record = list(record[attributes].values)
                # Join them with " " between them
                record = " ".join(record)
                # Save the completed record in the dictionary
                records[int(record_id)] = record
            except Exception as e:
                print(f'Oh no!: {record_id}')

        # Save the dictionary in the same folder as the datasets
        file_path = self.dataset_1_path.rsplit("/", maxsplit=1)[0]
        file_path += f"/{filename}"

        with open(file_path, "w") as fp:
            fp.write(json.dumps(records, indent=4))

        if self.verbose:
            print(f"Saved {len(records)} records to {file_path}")

    def organize_save_pairs(self, pairs: list) -> list:
        """Save the pairs in a csv file under the name pairs.json

        The pairs list contains tuples of d1, d2 ids
        We want to create a dictionary with the d1 attributes as key and
        a list of items as value.
        Each item is the d2 attributes of the cadidate found in the original pairs

        We will also create two files d1_pairs.csv and d2_pairs.csv
        Those will contain the id as key and the representation of a record as value
        This will be done for all the records of interest and will be used for the prompting
        """

        parsed_pairs = defaultdict(list)

        #NOTE: Here the parsed pairs contain:
        # Keys: The ids of d1
        # Values: Lists of ids of d2
        
        # Since we will need the unique ids of d2 bellow
        # we will also save them here
        ids_of_interest_2 = set()

        for pair in tqdm(pairs, desc="Parsing pairs", total=len(parsed_pairs)):
            parsed_pairs[pair[0]].append(pair[1])
            ids_of_interest_2.add(pair[1])

        if self.verbose:
            print("Parsing completed, will save pairs.json")

        # Get the path from d1 but keeping everything except the last /
        file_path = self.dataset_1_path.rsplit("/", maxsplit=1)[0]
        file_path += "/pairs.json"

        with open(file_path, "w") as fp:
            fp.write(json.dumps(parsed_pairs))

        if self.verbose:
            print("Will construct records for d1 and save them as a dictionary")

        # Save the records of interest for d1
        # The ids of interest here are the keys of the parsed pairs
        self.save_records_of_interest(
            dataset=self.d1,
            attributes=self.dataset_1_attrs,
            id_col="id",
            ids_of_interest=parsed_pairs.keys(),
            filename="d1_records.json",
        )

        self.save_records_of_interest(
            dataset=self.d2,
            attributes=self.dataset_2_attrs,
            id_col="id",
            ids_of_interest=ids_of_interest_2,
            filename="d2_records.json",
        )

        return parsed_pairs

    def run_blocking_workflow(
        self,
        vectorizer: str,
        similarity_search: str,
        top_k: int,
        similarity_distance: str,
        with_entity_matching: bool = True,
    ) -> list:
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

        stage_1_blocks, stage_1_graph, stage_1_data = self.run_blocking(
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

        stage_2_blocks, stage_2_graph, satage_2_data = self.run_blocking(
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

        # Extract the final pairs from the blocks
        final_pairs = self.create_final_pairs(
            stage_1_blocks, stage_1_data, stage_2_blocks, satage_2_data
        )

        # Organize and save the pairs into a csv file
        final_pairs = self.organize_save_pairs(final_pairs)

        end_time = time.time()

        if self.verbose:
            print(f"Block Creation Total Time: {(end_time - start_time)} seconds")

        return final_pairs
