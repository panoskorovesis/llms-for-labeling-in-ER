import datetime
import json
import subprocess
from typing import Union

import duckdb
import pandas as pd
from tqdm import tqdm

from src.llms.ask_llm import LLM
from src.utils.enums import (
    BruteForceMode,
    Models,
    PromptGroups,
    PromptTypes,
    ValidationStatus,
)
from src.utils.utils import load_config


class BruteForce:
    """This class will be responsible for running the brute force pipeline

    Everything will be orchistrated by the "run" method

    The prompting method applied here can be found in this paper: https://arxiv.org/pdf/2405.16884

    """

    def __init__(
        self,
        config_path: str,
        pairs_path: str,
        restart_script_path: str,
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

        self.restart_script_path = restart_script_path
        # make sure it contains the executable
        assert self.restart_script_path.endswith(
            "restart_ollama_with_gpu.sh"
        ), f"Please provide the path WITH the script"

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

    def restart_ollama_server(self) -> None:
        """Restart the ollama server

        This is needed in cases where the model bugs and keeps answering wrong
        Restart will be performed by the subprocess library
        It will be used to execute a bash script
        """
        print("Will restart the ollama server")
        results = subprocess.Popen(
            f"bash {self.restart_script_path}", shell=True, stdout=subprocess.PIPE
        )
        # wait the script completion
        results.wait()

        if self.verbose:
            print(results.stdout)

        print("Restart completed, will continue")

    def validate_prompt(self, mode: BruteForceMode, prompt_type: PromptTypes):
        """Make sure the given prompt_type is compatible with the mode

        Raise an exception in case of issue
        """
        if mode == BruteForceMode.MATCHING:
            assert (
                prompt_type in PromptGroups.MATCHING_GROUP.value
            ), f"Acceptable prompts for MATCHING: {PromptGroups.MATCHING_GROUP}"

        elif mode == BruteForceMode.COMPARING:
            assert (
                prompt_type in PromptGroups.COMPARING_GROUP.value
            ), f"Acceptable prompts for COMPARING: {PromptGroups.COMPARING_GROUP}"

        elif mode == BruteForceMode.SELECTING:
            assert (
                prompt_type in PromptGroups.SELECTING_GROUP.value
            ), f"Acceptable prompts for SELECTING: {PromptGroups.SELECTING_GROUP}"

    def create_db_table_if_needed(self, table_name: str, force=False):
        """Create a table to store the results in duckdb

        If the force is set to True we will delete the table and recreate it
        If the force is set to False, we will create the table only if it's not created
        """

        if force:
            self.con.sql(f"""DROP TABLE IF EXISTS {table_name}""")

            if self.verbose:
                print(f"Table {table_name} was deleted. It will be recreated.")

        self.con.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            d1_id VARCHAR NOT NULL,
            d2_candidates_id VARCHAR[] NOT NULL,
            result VARCHAR NULL,
            request_status VARCHAR NOT NULL,
            inserted_at TIMESTAMP NOT NULL,
            prompt_type VARCHAR NOT NULL,
            model VARCHAR NOT NULL,
            run_id VARCHAR NOT NULL,
        );
        """)

    def get_processed_records(
        self, promt_type: PromptTypes, run_id: str, model: Models
    ) -> set:
        """Get the processed pairs from duckdb

        We will get the pairs and return them as a set
        for better indexing.
        This will allow the code to continue the execution from where
        it stopped

        Returns
        -------
            set: The processed pairs
        """
        processed_pairs = set()

        # Fpr the Matching Prompt case we have to get the records with d1_id and matching prompt
        # And then create pairs from the d2_candidates_id list
        # Due to the matching this list contains only one pair, hence the code bellow
        if (
            promt_type == PromptTypes.MATCHING_PROMPT
            or promt_type == PromptTypes.MATCHING_PROMPT_ONLY
        ):
            pairs = self.con.sql(f"""
                SELECT DISTINCT d1_id, d2_candidates_id 
                FROM results 
                WHERE prompt_type = 'MATCHING_PROMPT'
                    and run_id = '{run_id}' 
                    and model = '{str(model)}'
                """).fetchall()
            # REMINDER: The d2_candidates_id is a list but due to the matching it should contain exactly
            # one item.
            # Keeping that in mind, let's create the processed_pairs
            for pair in pairs:
                # pair[0] : the d1_id
                # pair[1] : the list of d2_ids
                # pair[1][0] : the d2_id
                processed_pairs.add((pair[0], pair[1][0]))

        # For the comparing prompt case, for each d1_id we run the bubble sort and then
        # Write data in the db. as such we only have to get the d1_ids that are in the db
        elif promt_type in PromptGroups.COMPARING_GROUP.value:
            pairs = self.con.sql(f"""
                SELECT DISTINCT d1_id
                FROM results 
                WHERE prompt_type = '{PromptTypes.COMPARING_PROMPT}'
                    and run_id = '{run_id}' 
                    and model = '{str(model)}'
                """).fetchall()
            # NOTE: Fetchall will return these in a tuple.
            # For example (444, )
            # We want to keep only the ids and return them
            processed_pairs = {pair[0] for pair in pairs}

        # For the selecting prompt, matching and comparing are done and then the candidate is
        # written in the db. as such we only have to get the d1_ids that are in the db
        elif promt_type in PromptGroups.SELECTING_GROUP.value:
            pairs = self.con.sql(f"""
                    SELECT DISTINCT d1_id
                    FROM results 
                    WHERE prompt_type = '{PromptTypes.SELECTING_PROMPT}'
                        and run_id = '{run_id}' 
                        and model = '{str(model)}'
                    """).fetchall()
            # NOTE: Fetchall will return these in a tuple.
            # For example (444, )
            # We want to keep only the ids and return them
            processed_pairs = {pair[0] for pair in pairs}

        else:
            raise NotImplementedError(
                f"get_processed_records for {promt_type} is not valid!"
            )

        return processed_pairs

    def add_results_to_db(
        self,
        d1_id: str,
        candidate_pairs: list,
        d2_id: Union[str, None],
        status: str,
        prompt_type: PromptTypes,
        model: Models,
        run_id: str,
    ):
        """Add one line to the db with results for d1_id"""
        self.con.execute(
            """INSERT INTO results (d1_id, d2_candidates_id, result, request_status, inserted_at, prompt_type, model, run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                d1_id,
                candidate_pairs,
                d2_id,
                str(status),
                datetime.datetime.now(),
                str(prompt_type),
                str(model),
                run_id,
            ],
        )

    def run_matching(
        self,
        temperature: float,
        run_id: str,
        model: Models,
        prompt_type: PromptTypes,
        num_predict: int = 128,
        max_input_tokens: int = 2048,
        restart=False,
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
        # make sure the correct prompt is used
        self.validate_prompt(mode=BruteForceMode.MATCHING, prompt_type=prompt_type)

        self.create_db_table_if_needed(table_name="results", force=restart)

        # Get processed records
        # this method does only matching so we can set this as Matching
        processed_records = self.get_processed_records(
            promt_type=PromptTypes.MATCHING_PROMPT, run_id=run_id, model=model
        )

        for d1_id in tqdm(self.pairs, desc="Gathering results"):
            for d2_id in self.pairs[d1_id]:
                # If it's already processed, skip it
                if (d1_id, d2_id) in processed_records:
                    if self.verbose:
                        print(
                            f"{(d1_id, d2_id)} has already been processed. Will continue"
                        )
                    continue

                # Get the record texts from the dx_records.json
                d1_text = self.d1_records[d1_id]
                d2_text = self.d2_records[d2_id]

                # Ask the llm
                # NOTE: The ask_llm method expects one record as str and it's possible candidates
                # In the matching method, the candidates should be exactly one
                status, response = self.llm.ask_llm(
                    prompt_type=prompt_type,
                    record=d1_text,
                    candidate_records=[d2_text],
                    temperature=temperature,
                    num_predict=num_predict,
                    max_tokens=max_input_tokens,
                )

                if self.verbose:
                    print(f"{d1_id} - {d2_id}: {response} | {status}")

                # TODO: Change this if needed
                # ATM restart will only take place if the model is PHI
                # and the status is invalid
                # This was done as invalid answers in D3 dataset cause ALL The following to also be invalid
                if model == Models.PHI_3_INSTRUCT and status != ValidationStatus.VALID:
                    self.restart_ollama_server()

                # Gather the results by saving them in a duckdb table
                self.con.execute(
                    """INSERT INTO results (d1_id, d2_candidates_id, result, request_status, inserted_at, prompt_type, model, run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        d1_id,
                        [d2_id],
                        d2_id if response == "yes" else None,
                        str(status),
                        datetime.datetime.now(),
                        str(PromptTypes.MATCHING_PROMPT),
                        str(model),
                        run_id,
                    ],
                )

    def run_comparing(
        self,
        temperature: float,
        run_id: str,
        model: Models,
        prompt_type: PromptTypes,
        num_predict: int = 128,
        max_input_tokens: int = 2048,
        restart=False,
    ):
        """Run the comparing prompt method

        The way this works is the following:
        For each d1_id we take the candidate pairs and perform a bubble sort
        The criterion for the swap is given by asking the llm using d1, d2_i+1 d2_i
        We want the most similar to reach the top

        Then we ALWAYS select the element at the top of the list as the answer
        """
        self.validate_prompt(BruteForceMode.COMPARING, prompt_type=prompt_type)

        self.create_db_table_if_needed(table_name="results", force=restart)

        # Get processed records
        # this method does only matching so we can set this as Matching
        processed_records = self.get_processed_records(
            promt_type=PromptTypes.COMPARING_PROMPT, run_id=run_id, model=model
        )

        for d1_id in tqdm(self.pairs, desc="Gathering results"):
            # If it's already processed, skip it
            if d1_id in processed_records:
                if self.verbose:
                    print(f"{d1_id} has already been processed. Will continue")
                continue

            # Get the record texts from the dx_records.json
            d1_text = self.d1_records[d1_id]

            # Here we have to implement a bubble sort
            # The difference is that the condition with which we will switch is dependend on the LLM
            # set this for ease of access
            # use the constructor here to make sure its a deep copy
            candidate_pairs = list(self.pairs[d1_id])

            # Outer loop to iterate through the list n times
            for n in range(len(candidate_pairs) - 1, 0, -1):
                # Inner loop to compare adjecent elements
                for i in range(n):
                    # get the text of the two d2 records
                    d2_n_record = self.d2_records[candidate_pairs[i]]
                    d2_i_record = self.d2_records[candidate_pairs[i + 1]]

                    # Ask the llm
                    # NOTE: The ask_llm method expects one record as str and it's possible candidates
                    # In the comparing method, the candidates should be exactly two
                    status, response = self.llm.ask_llm(
                        prompt_type=PromptTypes.COMPARING_PROMPT,
                        record=d1_text,
                        candidate_records=[d2_i_record, d2_n_record],
                        temperature=temperature,
                        num_predict=num_predict,
                        max_tokens=max_input_tokens,
                    )

                    if self.verbose:
                        print(
                            f"{d1_id} - [{candidate_pairs[n]}, {candidate_pairs[i]}]: {response}"
                        )

                    # NOTE: Here algorithm says if i + 1 > i
                    # This goes in DESC order
                    # We can say that in our context the most similar should go up
                    # So if we got "record a" as a result we must swap

                    # If the status is invalid we wont do anything
                    if status == ValidationStatus.INVALID:
                        if self.verbose:
                            print(
                                f"Invalid responce for {candidate_pairs[i + 1]} - {candidate_pairs[i + 1]}. Will continue"
                            )
                            continue

                    if status == ValidationStatus.VALID and response == "record a":
                        if self.verbose:
                            print(
                                f"Will swap {candidate_pairs[i + 1]} with {candidate_pairs[i + 1]}"
                            )
                        # swap
                        candidate_pairs[i], candidate_pairs[i + 1] = (
                            candidate_pairs[i + 1],
                            candidate_pairs[i],
                        )

            # Here the candidate pairs have been ordered
            if self.verbose:
                print(
                    f"Ordering completed.\nInitial: {self.pairs[d1_id]}\nFinal: {candidate_pairs}"
                )

            # Finally after the bubble sort has been completed, take the id at position [0]
            # This will be our answer
            d2_id = candidate_pairs[0]

            if self.verbose:
                print(f"{d1_id} - {d2_id}: {response}")

            # Gather the results by saving them in a duckdb table
            self.add_results_to_db(
                d1_id=d1_id,
                candidate_pairs=candidate_pairs,
                d2_id=d2_id,
                status=status,
                prompt_type=PromptTypes.MATCHING_PROMPT,
                model=model,
                run_id=run_id,
            )

    def run_selecting(
        self,
        temperature: float,
        run_id: str,
        model: Models,
        prompt_type: PromptTypes,
        num_predict: int = 128,
        max_input_tokens: int = 2048,
        restart=False,
    ):
        """Run the Selecting Method

        This is the best method of the paper https://arxiv.org/pdf/2405.16884
        It uses the other two in the following way:
        1)  Apply selecting for all candidate pairs of d1_id
        2) Apply matching to sort those selected (ONLY IF) step 1 has > 1 items
        3) Apply selecting to find the final answer
        """

        OPTIMAL_MATCHING_PROMPT = PromptTypes.MATCHING_PROMPT_ONLY

        # for the matching step we will select the best performing prompt
        self.validate_prompt(
            mode=BruteForceMode.MATCHING, prompt_type=OPTIMAL_MATCHING_PROMPT
        )
        # recreate the tables if needed
        self.create_db_table_if_needed(table_name="results", force=restart)

        # Get processed records
        processed_records = self.get_processed_records(
            promt_type=PromptTypes.SELECTING_PROMPT, run_id=run_id, model=model
        )

        # Begin the process
        for d1_id in tqdm(self.pairs, desc="Gathering results"):
            # if d1 has already been processed continue
            if d1_id in processed_records:
                if self.verbose:
                    print(f"{d1_id} has already been processed. Will continue")
                continue

            # this list will keep the candidates returned by matching
            candidate_pairs = []

            ###################
            ###################
            ###################
            # STEP 1 - MATCHING
            ###################
            ###################
            ###################
            for d2_id in self.pairs[d1_id]:
                # Get the record texts from the dx_records.json
                d1_text = self.d1_records[d1_id]
                d2_text = self.d2_records[d2_id]

                # Ask the llm
                # NOTE: The ask_llm method expects one record as str and it's possible candidates
                # In the matching method, the candidates should be exactly one
                status, response = self.llm.ask_llm(
                    prompt_type=OPTIMAL_MATCHING_PROMPT,
                    record=d1_text,
                    candidate_records=[d2_text],
                    temperature=temperature,
                    num_predict=num_predict,
                    max_tokens=max_input_tokens,
                )

                if self.verbose:
                    print(f"{d1_id} - {d2_id}: {response}")

                # Now if the response is "yes" we have to add it to the matching results for the next step
                if response == "yes":
                    candidate_pairs.append(d2_id)

            ####################
            ####################
            ####################
            # STEP 2 - COMPARING
            ####################
            ####################
            ####################

            # This step ONLY has meaning if there are len(candidate_pairs) > 1
            if len(candidate_pairs) > 1:
                # Bubble Sort Algorithm
                # Outer loop to iterate through the list n times
                for n in range(len(candidate_pairs) - 1, 0, -1):
                    # Inner loop to compare adjecent elements
                    for i in range(n):
                        # get the text of the two d2 records
                        d2_n_record = self.d2_records[candidate_pairs[i]]
                        d2_i_record = self.d2_records[candidate_pairs[i + 1]]

                        # Ask the llm
                        # NOTE: The ask_llm method expects one record as str and it's possible candidates
                        # In the comparing method, the candidates should be exactly two
                        status, response = self.llm.ask_llm(
                            prompt_type=PromptTypes.COMPARING_PROMPT,
                            record=d1_text,
                            candidate_records=[d2_i_record, d2_n_record],
                            temperature=temperature,
                            num_predict=num_predict,
                            max_tokens=max_input_tokens
                        )

                        if self.verbose:
                            print(
                                f"{d1_id} - [{candidate_pairs[n]}, {candidate_pairs[i]}]: {response}"
                            )

                        # NOTE: Here algorithm says if i + 1 > i
                        # This goes in DESC order
                        # We can say that in our context the most similar should go up
                        # So if we got "record a" as a result we must swap

                        # If the status is invalid we wont do anything
                        if status == ValidationStatus.INVALID:
                            if self.verbose:
                                print(
                                    f"Invalid responce for {candidate_pairs[i + 1]} - {candidate_pairs[i + 1]}. Will continue"
                                )
                                continue

                        if status == ValidationStatus.VALID and response == "record a":
                            if self.verbose:
                                print(
                                    f"Will swap {candidate_pairs[i + 1]} with {candidate_pairs[i + 1]}"
                                )
                            # swap
                            candidate_pairs[i], candidate_pairs[i + 1] = (
                                candidate_pairs[i + 1],
                                candidate_pairs[i],
                            )

                # Here the candidate pairs have been ordered
                if self.verbose:
                    print(
                        f"Ordering completed.\nInitial: {self.pairs[d1_id]}\nFinal: {candidate_pairs}"
                    )

            ####################
            ####################
            ####################
            # STEP 3 - SELECTING
            ####################
            ####################
            ####################

            # Get the texts for the candidate records
            candidate_pairs_text = [
                self.d2_records[record] for record in candidate_pairs
            ]

            # Only ask the model is there is anything left
            if len(candidate_pairs_text) > 0:
                # Finally we just have to ask the model
                # NOTE: Here the candidate records can be as many as needed
                status, response = self.llm.ask_llm(
                    prompt_type=PromptTypes.SELECTING_PROMPT,
                    record=d1_text,
                    candidate_records=candidate_pairs_text,
                    temperature=temperature,
                    num_predict=num_predict,
                    max_tokens=max_input_tokens
                )

            # Else set it by hand
            # TODO: Make sure this does not disturb validation
            else:
                status, response = ValidationStatus.VALID, "0"

            # NOTE: Here we dont have to explicitly check for VALID or INVALID
            if self.verbose:
                print(f"{d1_id} - {d2_id}: {response} | {status}")

            # Now we have to find from the candidate pairs the correct one
            # They were given incremental IDS in the prompt so we will get the
            # one at response - 1
            selected = None
            if status == ValidationStatus.VALID and response != "0":
                selected = candidate_pairs[int(response) - 1]
            else:
                pass

            # Gather the results by saving them in a duckdb table
            self.add_results_to_db(
                d1_id=d1_id,
                candidate_pairs=candidate_pairs,
                d2_id=selected,
                status=status,
                # This allows us to use SELECTING or SELECTING ONLY
                prompt_type=prompt_type,
                model=model,
                run_id=run_id,
            )

    def run(
        self,
        mode: BruteForceMode,
        model: Models,
        run_id: str,
        temperature: float,
        prompt_type: PromptTypes,
        num_predict: int = 128,
        max_input_tokens: int = 2048,
        restart=False,
    ):
        """This is the main method that executes the brute force

        It has 3 modes:
        MATCHING
        COMPARING
        SELECTING

        The SELECTING Mode implements the COMEM framework as described in https://arxiv.org/pdf/2405.16884

        MATCHING:
            This is the simplest mode. We take all pairs and ask the model for each one
            #TODO: Complete this description

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

        # If the model is DEEPSEEK_R1 We want more tokens to allow the thinking process
        if model == Models.DEEPSEEK_R1:
            num_predict = 8192

        if mode == BruteForceMode.MATCHING:
            self.run_matching(
                temperature=temperature,
                run_id=run_id,
                model=model,
                prompt_type=prompt_type,
                num_predict=num_predict,
                restart=restart,
                max_input_tokens=max_input_tokens
            )

        elif mode == BruteForceMode.COMPARING:
            self.run_comparing(
                temperature=temperature,
                run_id=run_id,
                model=model,
                prompt_type=prompt_type,
                num_predict=num_predict,
                restart=restart,
                max_input_tokens=max_input_tokens
            )

        elif mode == BruteForceMode.SELECTING:
            self.run_selecting(
                temperature=temperature,
                run_id=run_id,
                model=model,
                prompt_type=prompt_type,
                num_predict=num_predict,
                restart=restart,
                max_input_tokens=max_input_tokens
            )

        else:
            raise NotImplementedError(f"Mode: {mode} is not yet implemented!")
