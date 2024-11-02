import datetime
import json

import duckdb
import pandas as pd
from tqdm import tqdm

from src.llms.ask_llm import LLM
from src.utils.enums import BruteForceMode, Models, PromptTypes
from src.utils.utils import load_config


class BruteForce:
    """This class will be responsible for running the brute force pipeline

    Everything will be orchistrated by the "run" method

    The prompting method applied here can be found in this paper: https://arxiv.org/pdf/2405.16884
    #TODO: See what we can do about few shot

    """

    def __init__(
        self,
        config_path: dict,
        pairs_path: str,
        d1_records_path: str,
        d2_records_path: str,
        ground_truth_path: str,
        verbose: bool = False,
    ) -> None:
        """Initialize the class

        The brute force wont need the datasets but the files created by the Blocking class:
        1) The pairs json
        2) The d1 records of interest json
        3) The d2 records of interest json

        It will also need the gt to calculate statistics at the end
        """
        self.config = load_config(config_path=config_path)

        # Load the jsons
        with open(d1_records_path, "r") as fp:
            self.d1_records = json.load(fp)

        with open(d2_records_path, "r") as fp:
            self.d2_records = json.load(fp)

        with open(pairs_path, "r") as fp:
            self.pairs = json.load(fp)

        self.ground_truth = pd.read_csv(ground_truth_path)

        self.verbose = verbose

        # Initialize a duckdb connection
        self.con = self.initialize_duckdb_connection(pairs_path=pairs_path)

        if self.verbose:
            print("BruteForce initialized.")

    def initialize_duckdb_connection(self, pairs_path: str):
        """Initialize a duckdb connection

        The db will be saved in the same folder as the pair.json file
        The name will be results.duckdb

        This db will be used to keep track of time and the resuls of the process
        """

        # Keep everything up until the last slash
        db_path = pairs_path.rsplit("/", maxsplit=1)[0]
        # Add the db name
        db_path += "/results.duckdb"

        con = duckdb.connect(db_path)

        if self.verbose:
            print(
                f'Duck db connection initialized! Tables: \n{con.sql("SHOW TABLES;")}'
            )

        return con

    def create_db_table_if_needed(self, table_name: str, force=False):
        """Create a table to store the results in duckdb

        If the force is set to True we will delete the table and recreate it
        If the force is set to False, we will create the table only if it's not created
        """

        if force:
            self.con.sql(f"""DROP TABLE IF EXISTS {table_name}""")

            if self.verbose:
                print(f'Table {table_name} was deleted. It will be recreated.')

        self.con.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            d1_id VARCHAR NOT NULL,
            d2_candidates_id VARCHAR[] NOT NULL,
            result VARCHAR NULL,
            request_status VARCHAR NOT NULL,
            inserted_at DATE NOT NULL,
        );
        """)

    def run_matching(
        self, temperature: float, num_predict: int = 128, restart=False
    ) -> None:
        """Run the MATCHING method on all the available pairs

        The pairs are saved in the following form

        "d1_id_1" : [
            "d2_id_1",
            "d2_id_2"
            "d2_id_3"
            "d2_id_4"
        ]

        We want to iterate over the keys and get every value in order to ask the model for the pair
        d1_id_x, d2_id_x_y

        Save the results in a duckdb table. This is done to allow us to continue if needed

        The restart flag signals that we want to abandon the previous run and start again
        """
        self.create_db_table_if_needed(table_name="matching_results", force=restart)

        for d1_id in tqdm(self.pairs, desc="Gathering results"):
            for d2_id in self.pairs[d1_id]:

                #TODO: Skip processed pairs
                
                # Get the record texts from the dx_records.json
                d1_text = self.d1_records[d1_id]
                d2_text = self.d2_records[d2_id]

                # Ask the llm
                # NOTE: The ask_llm method expects one record as str and it's possible candidates
                # In the matching method, the candidates should be exactly one
                status, response = self.llm.ask_llm(
                    prompt_type=PromptTypes.MATCHING_PROMPT,
                    record=d1_text,
                    candidate_records=[d2_text],
                    temperature=temperature,
                    num_predict=num_predict,
                )

                if self.verbose:
                    print(f'{d1_id} - {d2_id}: {response}')

                # Gather the results by saving them in a duckdb table
                self.con.execute("""INSERT INTO matching_results (d1_id, d2_candidates_id, result, request_status, inserted_at) VALUES (?, ?, ?, ?, ?)""", [ d1_id, [d2_id], d2_id if response == "yes" else None, str(status), datetime.datetime.now() ])

    def run(
        self,
        mode: BruteForceMode,
        model: Models,
        temperature: float,
        num_predict: int = 128,
        restart=False
    ):
        """This is the main method that executes the brute force

        It has 3 modes:
        MATCHING
        COMPARING
        SELECTING

        The SELECTING Mode implements the COMEM framework as described in https://arxiv.org/pdf/2405.16884

        MATCHING:
            This is the simplest mode. We take all pairs and ask the model for each one
            #TODO: Complete this description based on email answer

        Arguments
        ---------
            mode: BruteForceMode: The selected run mode

        Returns
        -------
            None
        """

        # First create the LLM
        self.llm = LLM(
            model=model,
            username=self.config["ollama"]["username"],
            password=self.config["ollama"]["password"],
            server_port=self.config["ollama"]["port"],
            server_url=self.config["ollama"]["url"],
            verbose=self.verbose,
        )

        if mode == BruteForceMode.MATCHING:
            self.run_matching(temperature=temperature, num_predict=num_predict, restart=restart)

        else:
            raise NotImplementedError(f"Mode: {mode} is not yet implemented!")
