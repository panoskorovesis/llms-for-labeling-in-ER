# faiss should already be installed as a part of the pyjedai package
import faiss
from src.utils.enums import SimilarityMetric, Embedding_Models
import duckdb
import json
from tqdm import tqdm
import numpy as np
import pandas as pd


class SimilarityCalculator:
    """This class implements similarity search using the faiss library

    This search will be done AFTER the blocking and the embedding steps
    It should generate a csv with the similarities, as well as some statistics
    """

    def __init__(
        self,
        db_path: str,
        pairs_path: str,
        ground_truth_path: str,
        verbose: bool = False,
        csv_separator: str = "|",
    ) -> None:
        """The constructor"""

        self.verbose = verbose
        self.csv_separator = csv_separator

        # Set the constant for the table name
        self.TABLE_NAME = "embeddings"

        # load the pairs.json file
        with open(pairs_path, "r") as fp:
            self.pairs = json.load(fp)

        # initialize the db connection
        self.con = self.initialize_duckdb_connection(db_path=db_path)

        # load the ground truth
        # also convert it to a dictionary for better search performance
        gt = self.load_dataset(dataset=ground_truth_path)
        self.gt = {str(row["D1"]): str(row["D2"]) for idx, row in gt.iterrows()}

        if self.verbose:
            print("Similarity Calculator Initialized")

    def initialize_duckdb_connection(self, db_path: str):
        """Initialize a duckdb connection

        The db will be saved in the same folder as the pair.json file
        The name will be results.duckdb

        This db will be used to keep track of time and the resuls of the process
        """

        # check that the path ends with results.duckdb
        if not db_path.endswith("results.duckdb"):
            raise ValueError("Database path must be like: /my/folder/results.duckdb")

        # Initialize the connection
        con = duckdb.connect(db_path)

        if self.verbose:
            print(
                f'Duck db connection initialized! Tables: \n{con.sql("SHOW TABLES;")}'
            )

        return con

    def load_dataset(self, dataset: str) -> list:
        """Load the given dataset from the given path"""

        df = pd.read_csv(dataset, sep=self.csv_separator, engine="python")
        if self.verbose:
            print(f"Loaded {dataset}.")

        return df

    def sort_by_field(
        self, scores: list, neighbor_ids: list, d2_ids: list, reverse=False
    ):
        """Sort the d2_ids and the distances based on the neighbor_ids"""
        return [d2_ids[i] for i in neighbor_ids]

    def initialize_faiss(
        self, embedding_model: Embedding_Models, similarity_metric: SimilarityMetric
    ):
        """Initialize faiss

        We have to create the index with the appropriate size
        The size of the index must be the vector dimensionality or embedding size
        To find this we will query the database using with the embedding model we want
        """

        # This is needed as the result is returned in a tuple (lenght,)
        # So we get [0]
        lengh = self.con.sql(
            f"""SELECT length(embeddings) as length from {self.TABLE_NAME} WHERE model = '{embedding_model}'"""
        ).fetchone()[0]

        if self.verbose:
            print(f"FAISS index will be created with size: {lengh}")

        # Create the index
        index = faiss.IndexFlatL2(lengh)

        # set the metric type
        if similarity_metric == SimilarityMetric.EUCLIDIAN:
            index.metric_type = faiss.METRIC_L2
        elif similarity_metric == SimilarityMetric.COSINE:
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError("Invalid similarity distance: ", self.similarity_distance)

        # return the index
        return index

    def get_embeddings_from_db(
        self, record_ids: list, dataset: str, model: Embedding_Models
    ):
        """Get the embeddings of the requested Ids, based on the model and the dataset

        Here we can pass all ids at once and make sure we return them along with the embeddings
        This will prevent any errors
        """

        # we specify the d1 dataset
        # again, the results are in a tuple, we need all except the last
        results = self.con.sql(f"""
                            SELECT record_id, embeddings from {self.TABLE_NAME} where model = '{model}' and dataset = '{dataset}' and record_id in {record_ids}
                            """).fetchall()
        # separate the ids and the embeddings
        # We use zip and *results which separates each tuple in a, b
        # Then zip allows us to iterate or do what we have done to get the separate elements
        ids, embeddings = zip(*results)
        # return them
        return ids, embeddings

    def extract_statistics(
        self,
        df: pd.DataFrame,
        embedding_model: Embedding_Models,
        similarity_metric: SimilarityMetric,
    ):
        """Extract statistics from the dataframe

        We are interested in
        The percentage of correct predicted id2
        The percentage where the correct id2 was in the first two spots
        The percentage where the correct id2 was in the first three spots
        The percentage where the correct id2 was in the first four spots
        The percentage where the correct id2 was in the five three spots
        etc
        """
        total_correct = 0
        correct_top_2 = 0
        total_top_2 = 0
        correct_top_3 = 0
        total_top_3 = 0
        correct_top_4 = 0
        total_top_4 = 0
        correct_top_5 = 0
        total_top_5 = 0
        d1_ids_with_no_match = 0

        for idx, row in df.iterrows():
            # if the d1_id is NOT in the gt dictionary we have to skip the calculations
            if row["d1_id"] not in self.gt:
                d1_ids_with_no_match += 1
                continue

            # set this to avoid contant dict lookups
            correct_d2 = self.gt[row["d1_id"]]
            sorted_d2_ids = row["sorted_d2_ids"]

            # count correct
            if row["predicted_d2_id"] == correct_d2:
                total_correct += 1

            # get top_2
            if len(sorted_d2_ids) >= 2:
                total_top_2 += 1

                if correct_d2 in sorted_d2_ids[:2]:
                    correct_top_2 += 1

            # get top_3
            if len(sorted_d2_ids) >= 3:
                total_top_3 += 1

                if correct_d2 in sorted_d2_ids[:3]:
                    correct_top_3 += 1

            # get top_4
            if len(sorted_d2_ids) >= 4:
                total_top_4 += 1

                if correct_d2 in sorted_d2_ids[:4]:
                    correct_top_4 += 1

        return {
            "EMBEDDING_MODEL": str(embedding_model),
            "SIMILARITY_METRIC": str(similarity_metric),
            "correct_percentage": total_correct / len(df),
            "correct": f"{total_correct}/{len(df)}",
            "in top_2_percentage": correct_top_2 / total_top_2,
            "in top_2": f"{correct_top_2}/{total_top_2}",
            "in top_3_percentage": correct_top_3 / total_top_3,
            "in top_3": f"{correct_top_3}/{total_top_3}",
            "in top_4_percentage": correct_top_4 / total_top_4,
            "in top_4": f"{correct_top_4}/{total_top_4}",
            "d1 ids without match": d1_ids_with_no_match,
        }

    def print_statistics(self, statistics: dict):
        """Print statistics in a pretty way"""

        label = f'------------- {statistics["EMBEDDING_MODEL"].upper()} - {statistics["SIMILARITY_METRIC"].upper()} -------------'

        print("-" * len(label))
        print(label)
        print("-" * len(label))

        print(json.dumps(statistics, indent=4))
        print('\n')


    def calculate_similarities(
        self, metric: SimilarityMetric, embedding_model: Embedding_Models
    ):
        """Calculate similarities using the requested metric and embedding model

        Those similarities will be calculated on the results of blocking
        As such we have to also access the corresponding blocking files
        """

        computed_similarities = {
            "d1_id": [],
            "sorted_d2_ids": [],
            "scores": [],
            "predicted_d2_id": [],
        }

        # first create the index
        self.faiss_index = self.initialize_faiss(
            embedding_model=embedding_model, similarity_metric=metric
        )
        # now we have to create the following vectors
        # d1_vectors
        # this will have the embeddings for all the keys in the pairs dictionary
        d1_ids, d1_embeddings = self.get_embeddings_from_db(
            record_ids=list(self.pairs.keys()), dataset="d1", model=embedding_model
        )

        # create a dictionary that maps the d1_ids to the incremental d1_vectors id
        d1_ids_to_vector_ids = {
            d1_id: position for position, d1_id in enumerate(d1_ids)
        }

        # make sure that the returned and the original ids match
        assert (
            set(d1_ids) - set(self.pairs.keys()) == set()
        ), "The was an error with the embeddings retrieval for d1"

        # create an array with all embeddings
        # float32 type is required from the FAISS documentation
        # https://github.com/facebookresearch/faiss/wiki/Getting-started
        d1_vectors = np.vstack(d1_embeddings).astype("float32")

        if self.verbose:
            print(f"D1 Embeddings matrix was created with shape: {d1_vectors.shape}")

        # Now do the same for d2
        # Since the values in this dictionary are lists with d2_ids
        # We first have to find the unique ids of interest
        unique_d2_ids = set()
        max_neighbors = -1
        for value_list in self.pairs.values():
            unique_d2_ids.update(value_list)
            # also keep track of the max_neighbors
            max_neighbors = max(max_neighbors, len(value_list))

        d2_ids, d2_embeddings = self.get_embeddings_from_db(
            record_ids=list(unique_d2_ids), dataset="d2", model=embedding_model
        )

        # create a dictionary that maps the d2_ids to the incremental d2_vectors_id
        d2_ids_to_vector_ids = {
            d2_id: position for position, d2_id in enumerate(d2_ids)
        }

        # make sure that the returned and the original ids match
        assert (
            set(d1_ids) - set(self.pairs.keys()) == set()
        ), "The was an error with the embeddings retrieval for d2"

        d2_vectors = np.vstack(d2_embeddings).astype("float32")

        # perform faiss search

        # Per the documentation if the given distance is "cosine"
        # we have to normalize_L2 the datasets in order for our code to work
        if metric == SimilarityMetric.COSINE:
            faiss.normalize_L2(d1_vectors)
            faiss.normalize_L2(d2_vectors)

        # We can now perform the search
        # Since we want to compare our findings with the llm predictions
        # For each d1_id we will create the index and search ONLY the d2_vectors that are of relevance
        # We can find these using the two dicionaries we have created
        for d1_id in tqdm(
            self.pairs.keys(),
            total=len(self.pairs),
            desc=f"Calculating {metric} similarities",
        ):
            # first reset the index
            self.faiss_index.reset()

            # first get the embeddings of the d1_id
            d1_id_idx = d1_ids_to_vector_ids[d1_id]
            # the shape here will be (embeddings_size,) we want it to be (1, emb_size)
            d1_vector = d1_vectors[d1_id_idx].reshape(1, -1)

            d2_neighbor_vectors = []
            # Now for each element in the pairs do the same:
            for d2_id in self.pairs[d1_id]:
                # get the embeddings of the d2_id
                d2_id_idx = d2_ids_to_vector_ids[d2_id]
                d2_vector = d2_vectors[d2_id_idx]
                # add them to the list
                d2_neighbor_vectors.append(d2_vector)

            # Convert the list to a numpy array
            d2_neighbor_vectors = np.vstack(d2_neighbor_vectors).astype("float32")

            self.faiss_index.add(d2_neighbor_vectors)

            # We are now ready to perform the actual search
            distances, neighbors = self.faiss_index.search(
                d1_vector, len(self.pairs[d1_id])
            )

            # Fix how euclidian distances appear
            if metric == SimilarityMetric.EUCLIDIAN:
                distances = 1 / (1 + distances)

            # neighbors will have a shape of (1, num_neighbors)
            # We want to get rid of the first dimention so we will flatten
            neighbors = neighbors.flatten().tolist()
            # same thing for the distances
            distances = distances.flatten().tolist()

            # Finally sort the ids by their distances in desc order
            # sorted_distances, sorted_d2_ids = self.sort_by_field(scores=distances, neighbor_ids=neighbors, d2_ids=self.pairs[d1_id])
            sorted_d2_ids = self.sort_by_field(
                scores=distances, neighbor_ids=neighbors, d2_ids=self.pairs[d1_id]
            )

            # Keep this data in a dictionary
            # We will later convert this to a csv
            computed_similarities["d1_id"].append(d1_id)
            computed_similarities["sorted_d2_ids"].append(sorted_d2_ids)
            computed_similarities["scores"].append(distances)
            computed_similarities["predicted_d2_id"].append(sorted_d2_ids[0])

        # Create a dataframe with the info
        df = pd.DataFrame(computed_similarities)

        stats = self.extract_statistics(
            df, embedding_model=embedding_model, similarity_metric=metric
        )

        self.print_statistics(statistics=stats)

