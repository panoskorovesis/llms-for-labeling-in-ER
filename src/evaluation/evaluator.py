import json

import duckdb
import pandas as pd

from src.utils.enums import Models, PromptTypes


class Evaluator:
    """The evaluator class

    The purpose of this class is to calculate and report metrics such as
    1) F1
    2) Recall

    The pairs must be inside a duckdb database, under the "results" table
    """

    def __init__(
        self,
        ground_truth_path: str,
        csv_separator: str = "|",
        to_csv=None,
        verbose: bool = False,
    ):
        # Initialize variables
        self.verbose = verbose
        self.csv_separator = csv_separator
        self.to_csv = to_csv
        # This will be set to False after writing to the csv
        # It's used to avoid having multiple headers
        self.first_file_write = True

        # Load connections and files
        # We know the results.duckdb database file is in the same folder as gt
        self.con = self.initialize_duckdb_connection(ground_truth_path)
        self.gt_pairs = self.load_ground_truth_pairs(ground_truth_path)

        # if to_csv is given then we have to write the results to an csv file
        # Here we will set the path and create the file
        self.csv_path = self.get_clean_path(ground_truth_path)
        self.csv_path = self.initialize_results_csv(self.csv_path)

        if self.verbose:
            print("Evaluator initialized.")

    def get_clean_path(self, path: str) -> str:
        """Given a path that contains a file at the end, return only the clean path

        This is done by rsplit with maxsplit = 1
        """
        clean_path = path.rsplit("/", maxsplit=1)[0]
        return clean_path

    def initialize_results_csv(self, csv_path) -> str:
        """Create a new file at the specified path with the name
        "evaluation.csv". Return the full path
        """
        # Create the file or empty it if it's already created
        full_path = csv_path + "/evalutation.csv"
        with open(full_path, "w") as fp:
            fp.write("")

        if self.verbose:
            print(f"{full_path} has been created!")

        return full_path

    def initialize_duckdb_connection(
        self, ground_truth_path: str
    ) -> duckdb.DuckDBPyConnection:
        """Initialize the db connection

        We will replace the last part of the gt path with "results.duckdb" to get the connection
        """
        # Keep everything up until the last slash
        db_path = ground_truth_path.rsplit("/", maxsplit=1)[0]
        # Add the db name
        db_path += "/results.duckdb"

        # Connect to the db
        # Here we can set the connection to read only
        # Scores will be send to stdout
        con = duckdb.connect(db_path, read_only=True)

        if self.verbose:
            print(
                f'Duck db connection initialized! Tables: \n{con.sql("SHOW TABLES;")}'
            )

        return con

    def load_ground_truth_pairs(self, ground_truth_path: str) -> dict:
        """Load the gt csv and extract the ground_truth_pairs

        Since this is for a clean - clean ER the pairs can be in the form of
        a dictionary for efficiency
        """
        # load the csv
        df = pd.read_csv(ground_truth_path, sep=self.csv_separator)

        if self.verbose:
            print(f"Total Ground Truth pairs: {len(df)}")

        # Create a dictionary as follows:
        # Use the first column as a key
        # Use the second as a value
        return {row[df.columns[0]]: row[df.columns[1]] for idx, row in df.iterrows()}

    def get_predicted_pairs(
        self, model: Models, prompt_type: PromptTypes, run_id: str
    ) -> list:
        """Return the results in a list of tuples"""
        results = self.con.sql(
            f"""SELECT d1_id, result 
            FROM results
            WHERE result is not NULL
            and run_id = '{run_id}'
            and model = '{model}'
            and prompt_type = '{prompt_type}';"""
        ).fetchall()

        return results

    def get_elapsed_time(self, model: Models, prompt_type: PromptTypes, run_id: str):
        """Return the total elapsed time

        We calculate the time by DATE_DIFF Between the MIN and the MAX time
        in the run
        The elapsed time is presented in hours
        """
        results = self.con.sql(
            f"""SELECT MIN(inserted_at), MAX(inserted_at), DATE_DIFF('minutes', MIN(inserted_at), MAX(inserted_at)) / 60.0
            FROM results
            WHERE run_id = '{run_id}'
            and model = '{model}'
            and prompt_type = '{prompt_type}';"""
        ).fetchone()

        return {
            "Started At": results[0],
            "Ended At": results[1],
            "Total Hours": results[2],
        }

    def calculate_metrics(self, predicted_pairs: list):
        """Calculate the following metrics

        1. Accuracy
        2. Precision
        3. Recall
        4. F1
        """

        TP = 0
        FP = 0
        for pair in predicted_pairs:
            d1_id, d2_id = pair
            # Convert the ids back to integers
            # as gt has them as integers
            d1_id = int(d1_id)
            d2_id = int(d2_id)

            # try to find the d1_id in the gt pairs
            # here the following are possible
            # 1) d1_id does not exist in gt == NO MATCH // command returns None
            # 2) d1_id exists in gt == MATCH // command returns the CORRECT d2_id
            result = self.gt_pairs.get(d1_id)

            # True positive
            if result == d2_id:
                TP += 1

            # False positive: It's matched in the predicted but with the wrong item
            elif result is not None and result != d2_id:
                FP += 1

        # False Negatives in this context are records present in gt but NOT in the predicted pairs
        # First lets get the d1_id of all the predicted pairs
        d1_ids = {int(pair[0]) for pair in predicted_pairs}
        # Next lets do the same for the gt_pairs
        gt_d1_ids = set(self.gt_pairs.keys())
        # We can find what we are looking for by taking gt_d1_ids - d1_ids and then the length
        FN = len(gt_d1_ids - d1_ids)

        metrics = {
            # Accuracy = Correct / All or (TP + TN) / (TP + TN + FP + FN)
            "Accuracy": float(TP) / len(predicted_pairs)
            if len(predicted_pairs) != 0
            else 0.0,
            # Precision = Correct / All positive or TP / (TP + FP)
            "Precision": float(TP) / (TP + FP) if (TP + FP) != 0 else 0.0,
            # Recall = Correct / All Correct or (TP) / (TP + FN)
            "Recall": float(TP) / (TP + FN) if (TP + FN) != 0 else 0.0,
            # F1 = precision * recall / (precision + recall)
            "F1": 2 * float(TP) / (2 * TP + FP + FN) if (TP + FP + FN) != 0 else 0.0,
        }

        # Convert the metrics to percentages
        metrics["Accuracy"] *= 100
        metrics["F1"] *= 100
        metrics["Precision"] *= 100
        metrics["Recall"] *= 100

        if self.verbose:
            print(f"TOTAL in GT: {len(self.gt_pairs)}\nTP: {TP}\nFP: {FP}\nFN: {FN}\n")

        return metrics

    def generate_report(
        self,
        model: Models,
        prompt_type: PromptTypes,
        metrics: dict,
        run_id: str,
        elapsed_time: float,
    ):
        print("***************************************")
        print("******EVALUATION REPORT****************")
        print("***************************************")
        print(
            f"MODEL: {model}\nPROMPT TYPE: {prompt_type}\nRUN ID: {run_id}\nELAPSED TIME: {elapsed_time}"
        )
        print("----------")
        for key, value in metrics.items():
            print(f"{key.upper()}: {value}")
        print("***************************************")
        print("***************************************")
        print("***************************************\n\n")

    def save_report(
        self,
        model: Models,
        prompt_type: PromptTypes,
        run_id: str,
        metrics: dict,
        elapsed_time: dict,
        sep: str = ",",
    ) -> None:
        """Crete a dataframe with the given information and then save to the csv file"""
        data = {"Model": str(model), "Prompt Type": prompt_type, "Run ID": run_id}

        # Add the two additional dictionaries
        data.update(elapsed_time)
        data.update(metrics)

        df = pd.DataFrame(data, index=[0])

        if self.verbose:
            print(df.head())
            print("Will be written to the evaluation csv")

        # Mode is always 'a' as the file is created once when the class is initialized
        # if its the first time include the headers
        if self.first_file_write:
            df.to_csv(self.csv_path, index=False, mode="a", sep=sep)
            self.first_file_write = False
        else:
            df.to_csv(self.csv_path, index=False, mode="a", sep=sep, header=False)

    def evaluate(self, model: Models, prompt_type: PromptTypes, run_id: str) -> dict:
        """Calculate the the metrics and return them in a dictionary

        We want to calculate
        1) Precision
        2) Recall
        3) F1
        """

        # first get the predicted pairs from the db
        # Those pairs were given by various code methods
        predicted_pairs = self.get_predicted_pairs(
            model=model, prompt_type=prompt_type, run_id=run_id
        )

        if self.verbose:
            print(
                f"{len(predicted_pairs)} Predicted pairs. \nModel: {model} \nPrompt: {prompt_type} \nRun ID: {run_id}"
            )

        metrics = self.calculate_metrics(predicted_pairs)

        # Calculate the elapsed_time
        elapsed_time = self.get_elapsed_time(
            model=model, prompt_type=prompt_type, run_id=run_id
        )

        # Print the metrics
        self.generate_report(
            metrics=metrics,
            model=model,
            prompt_type=prompt_type,
            run_id=run_id,
            elapsed_time=elapsed_time,
        )

        # if we also have to_csv we have to add the results to the csv file
        if self.to_csv:
            self.save_report(
                model=model,
                prompt_type=prompt_type,
                run_id=run_id,
                metrics=metrics,
                elapsed_time=elapsed_time,
            )

        return metrics
