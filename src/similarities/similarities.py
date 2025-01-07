# faiss should already be installed as a part of the pyjedai package
import faiss
from src.utils.enums import SimilarityMetric, Embedding_Models
import datetime
import duckdb
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from pprint import pprint
import torch.nn.functional as F


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
        to_csv=False,
        normalize=False,
    ) -> None:
        """The constructor"""

        self.verbose = verbose
        self.csv_separator = csv_separator
        self.to_csv = to_csv
        self.normalize = normalize

        # load the pairs.json file
        with open(pairs_path, "r") as fp:
            self.pairs = json.load(fp)

        # initialize the db connection
        self.con = self.initialize_duckdb_connection(db_path=db_path)

        # get the dataset name
        self.dataset_name, self.csv_path = self.extract_dataset_name_and_csv_path(
            ground_truth_path=ground_truth_path
        )

        # Initialize the similarities csv
        self.initialize_results_csv(csv_path=self.csv_path)

        # Used to avoid writting multiple column names in the csv
        self.first_file_write = True

        # load the ground truth
        # also convert it to a dictionary for better search performance
        gt = self.load_dataset(dataset=ground_truth_path)
        self.gt = {str(row["D1"]): str(row["D2"]) for idx, row in gt.iterrows()}

        if self.verbose:
            print("Similarity Calculator Initialized")

    def extract_dataset_name_and_csv_path(self, ground_truth_path: str) -> tuple:
        """Get the dataset name from the ground_truth_path

        We know that the ground_truth_path contains the dataset. for example
        myfolder/D2/gt_clean.csv
        We will extract this "D2"
        The following code works given that all datasets are named D#

        Also return the csv path
        This will be as the ground_truth but with similarities.csv
        """
        dataset_number = ground_truth_path.split("/D")[-1].split("/")[0]

        csv_path = ground_truth_path.rsplit("/", maxsplit=1)[0]
        csv_path += "/similarities.csv"

        return f"D{dataset_number}", csv_path

    def initialize_results_csv(self, csv_path) -> str:
        """Create a new file at the specified path. The file name is ALREADY Included"""
        with open(csv_path, "w") as fp:
            fp.write("")

        if self.verbose:
            print(f"{csv_path} has been created!")

        return csv_path

    def _normalize(self, arr, p=2, axis=None):
        """Perform the equivalent of torch F.normalize"""
        norm = np.linalg.norm(arr, ord=p, axis=axis, keepdims=True)
        return arr / norm

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

        Depending on the use_task flag we have to choose between the embeddings table and
        the tast_embeddings table
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
        self, record_ids: list, dataset: str, model: Embedding_Models, use_task=False
    ):
        """Get the embeddings of the requested Ids, based on the model and the dataset

        Here we can pass all ids at once and make sure we return them along with the embeddings
        This will prevent any errors
        """

        if not use_task:
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
        else:
            # Get all relevant results from the db, then we will create the dictionary
            items = self.con.sql(
                f"""SELECT d1_reference_id, record_id, is_d2, embeddings from {self.TABLE_NAME} where model = '{model}' and d1_reference_id in {record_ids}"""
            ).fetchall()

            embeddings_dict = {}

            for i in tqdm(
                range(len(items)), total=len(items), desc="Gathering embeddings"
            ):
                """Item contains the following:
                [0] d1_reference_id
                [1] record_id
                [2] is_d2
                [3] embeddings
                """
                item = items[i]

                # TODO: TEST THIS
                # if the d1_reference_id is not in the dictionary add it
                if item[0] not in embeddings_dict:
                    embeddings_dict[item[0]] = {
                        "d1_embeddings": "",
                        "d2_ids": [],
                        "d2_embeddings": [],
                    }
                # If its not a d2_id its the embeddings for d1
                # TODO: Change this to true
                if item[2] == True:
                    embeddings_dict[item[0]]["d1_embeddings"] = item[3]
                # else it's the d2 embeddings. Here we will save the d2_id and the d2 embeddings
                else:
                    # Add d2 embeddings
                    embeddings_dict[item[0]]["d2_embeddings"].append(item[3])
                    # Add d2 id
                    embeddings_dict[item[0]]["d2_ids"].append(item[1])

            return embeddings_dict

    def calculate_metrics(
        self, metrics: dict, correct_d2_id: str, d2_candidate_ids: list, top_k: int
    ) -> dict:
        """Calculate TP, FP for each of the top_k first positions

        NOTE: We wont calculate FN as for recall we will use
        THE NUMBER OF TOTAL DUPLICATES PER DATASET as our denominator

        top_k must be >=1 and <= len(d2_candidate_ids)
        """

        # Now calculate the metrics
        # We will do this for k in [2, top_k]

        # First position will be handled separately
        if correct_d2_id == d2_candidate_ids[0]:
            metrics[1]["TP"] += 1
        else:
            metrics[1]["FP"] += 1

        for k in range(2, top_k + 1):
            # # if the d2_candidate_ids does not have as many elements break
            # if len(d2_candidate_ids) < k:
            #     break

            # check if the correct id is in the top_k positions
            if correct_d2_id in d2_candidate_ids[:k]:
                # count as correct
                metrics[k]["TP"] += 1
                # all others in top_k are considered FP
                metrics[k]["FP"] += min(k, len(d2_candidate_ids)) - 1
            # if it's not in the top_k
            else:
                # we have as many FP as K
                # we have + 1 FN
                metrics[k]["FP"] += k

        # return the calculated metrics
        return metrics

    def extract_statistics(
        self,
        df: pd.DataFrame,
        embedding_model: Embedding_Models,
        similarity_metric: SimilarityMetric,
        top_k: int = 5,
    ):
        """Extract statistics from the dataframe

        We are interested in
        The percentage of correct predicted id2
        The percentage where the correct id2 was in the first top_k spots

        This will have a clear tradeoff in recall / precision

        NOTE: We wont calculate FN as for recall we will use
        THE NUMBER OF TOTAL DUPLICATES PER DATASET as our denominator

        top_k must be >=1 and <= len(d2_candidate_ids)
        """

        # make sure the top_k is valid
        assert top_k >= 1, "Invalid top_k! It must be in >=1"

        recall_denominator = len(self.gt)

        # create the dicrionary based on top_k
        metrics = {}
        for k in range(1, top_k + 1):
            metrics[k] = {
                "TP": 0,
                "FP": 0,
            }

        d1_ids_with_no_match = 0

        for idx, row in df.iterrows():
            # if the d1_id is NOT in the gt dictionary we have to skip the calculations
            if row["d1_id"] not in self.gt:
                d1_ids_with_no_match += 1
                continue

            # set this to avoid contant dict lookups
            correct_d2 = self.gt[row["d1_id"]]
            sorted_d2_ids = row["sorted_d2_ids"]

            mertrics = self.calculate_metrics(
                metrics=metrics,
                correct_d2_id=correct_d2,
                d2_candidate_ids=sorted_d2_ids,
                top_k=top_k,
            )

        # Create the statistics dictionary
        statistics = {
            "EMBEDDING_MODEL": str(embedding_model),
            "SIMILARITY_METRIC": str(similarity_metric),
            "DATASET": self.dataset_name,
            "d1 ids without match": d1_ids_with_no_match,
        }

        for key in mertrics.keys():
            statistics[f"top_{key}_recall"] = mertrics[key]["TP"] / recall_denominator
            statistics[f"top_{key}_precision"] = mertrics[key]["TP"] / (
                mertrics[key]["TP"] + mertrics[key]["FP"]
            )

        return statistics

    def print_statistics(self, statistics: dict):
        """Print statistics in a pretty way"""

        label = f'------------- {statistics["EMBEDDING_MODEL"].upper()} - {statistics["SIMILARITY_METRIC"].upper()} -------------'

        print("-" * len(label))
        print(label)
        print("-" * len(label))

        print(json.dumps(statistics, indent=4))
        print("\n")

    def get_embeddings_generation_time(self, model: str):
        """Return the total embeddings generation time in minutes or hours"""
        # We expect only one row so we can use fetch one
        start_time, end_time = self.con.sql(
            f"""
            SELECT min(created_at), max(created_at)
            FROM {self.TABLE_NAME}
            WHERE model = '{model}'
            """
        ).fetchone()

        # get the elapsed time
        elapsed_time = end_time - start_time

        # if it's less than one hour return in minutes
        if elapsed_time < datetime.timedelta(hours=1):
            return f"{elapsed_time.total_seconds() / 60:.2f} minutes"
        else:
            return f"{elapsed_time.total_seconds() / 3600:.2f} hours"

    def save_report(self, data: dict, sep: str = ",") -> None:
        """Save the sstatistics in a csv file

        The file name will be similarities.csv
        The easiest way to do this is cast the dict to a csv and use pandas
        """

        # Add the total generation time to the results
        data["EMBEDDINGS_GENERATION_TIME"] = self.get_embeddings_generation_time(
            model=data["EMBEDDING_MODEL"]
        )

        df = pd.DataFrame(data, index=[1])

        if self.verbose:
            pprint(df.head())
            print("Will be written to the similarities csv")

        # Mode is always 'a' as the file is created once when the class is initialized
        # if its the first time include the headers
        if self.first_file_write:
            df.to_csv(self.csv_path, index=False, mode="a", sep=sep)
            self.first_file_write = False
        else:
            df.to_csv(self.csv_path, index=False, mode="a", sep=sep, header=False)

    def calculate_similarities_simple(
        self,
        metric: SimilarityMetric,
        embedding_model: Embedding_Models,
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

        # Normalize the data if required
        if self.normalize:
            d1_vectors = self._normalize(d1_vectors, p=2, axis=1)
            d2_vectors = self._normalize(d2_vectors, p=2, axis=1)

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

        # If requested also save the data to a csv file
        if self.to_csv:
            self.save_report(data=stats)

    def calculate_similarities_task(
        self,
        metric: SimilarityMetric,
        embedding_model: Embedding_Models,
        use_task: bool = True,
    ):
        """Calculate the embeddings from the task"""

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

        """
        Now get the embeddings from the db
        In this case, we have calculated the d2 embeddings based on d1
        So we will get a dictionary with the following structure
        {
            "d1_id" : {
                "d2_ids" : [d2_id_1, d2_id_2, ...]
                "d2_embeddings : [d2_embeddings_1, d2_embeddings_2]
            },
            ...
        }
        """
        embeddings_dict = self.get_embeddings_from_db(
            record_ids=list(self.pairs.keys()),
            dataset="d1",
            model=embedding_model,
            use_task=use_task,
        )

        # We can now perform the search
        # Since we want to compare our findings with the llm predictions
        # For each d1_id we will create the index and search ONLY the d2_vectors that are of relevance
        # We can find these using the two dicionaries we have created
        for d1_id in tqdm(
            self.pairs.keys(),
            total=len(self.pairs),
            desc=f"Calculating {metric} similarities",
        ):
            # first get the embeddings for the d1_id
            d1_vector = embeddings_dict[d1_id]["d1_embeddings"]
            # Convert to an array
            # float32 type is required from the FAISS documentation
            # https://github.com/facebookresearch/faiss/wiki/Getting-started
            d1_vector = np.array(d1_vector).astype("float32")
            # Here the shape is (emb_length, )
            # We will add one more dim to apply the normalization
            d1_vector = d1_vector.reshape(1, d1_vector.shape[0])

            # Now do the same for d2. Since we have many neighbors for each d1
            # We have to do this in a loop
            d2_ids = embeddings_dict[d1_id]["d2_ids"]
            # Create a helper dictionary
            d2_ids_to_vector_ids = {
                d2_id: idx for idx, d2_id in enumerate(embeddings_dict[d1_id]["d2_ids"])
            }

            d2_vectors = []
            for d2_embedding in embeddings_dict[d1_id]["d2_embeddings"]:
                # Add the vectors to the embeddings list
                d2_vectors.append(d2_embedding)

            # Convert the list to a numpy array
            d2_vectors = np.vstack(d2_vectors).astype("float32")

            # NOTE: Here we have not make any changes to the order so
            # if needed we can use the d2_ids_to_vector_ids for indexing

            # Now Normalize the data if required
            if self.normalize:
                d1_vector = self._normalize(d1_vector, p=2, axis=1)
                d2_vectors = self._normalize(d2_vectors, p=2, axis=1)

            # perform faiss search

            # Per the documentation if the given distance is "cosine"
            # we have to normalize_L2 the datasets in order for our code to work
            if metric == SimilarityMetric.COSINE:
                faiss.normalize_L2(d1_vector)
                faiss.normalize_L2(d2_vectors)

            # first reset the index
            self.faiss_index.reset()

            self.faiss_index.add(d2_vectors)

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

        # If requested also save the data to a csv file
        if self.to_csv:
            self.save_report(data=stats)

    def calculate_similarities(
        self,
        metric: SimilarityMetric,
        embedding_model: Embedding_Models,
        use_task=False,
    ):
        """Calculate similarities using the requested metric and embedding model

        Those similarities will be calculated on the results of blocking
        As such we have to also access the corresponding blocking files

        We have two options
        1) Normal embeddings
        2) Embeddings using task

        In the second case we have to look to another db table
        From there we can take for each d1 all the relevant d2 vectors
        """

        # Set the constant for the table name depending on the use_task flag
        if not use_task:
            self.TABLE_NAME = "embeddings"
        else:
            self.TABLE_NAME = "task_embeddings"

        if not use_task:
            self.calculate_similarities_simple(
                metric=metric, embedding_model=embedding_model
            )
        else:
            self.calculate_similarities_task(
                metric=metric, embedding_model=embedding_model, use_task=use_task
            )
