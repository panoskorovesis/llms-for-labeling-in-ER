from enum import Enum
from typing import Union


class SimilarityMetric(Enum):
    """Enum class for the different supported similarities"""

    COSINE = 1
    EUCLIDIAN = 2

    def __str__(self) -> str:
        """Return the distance as a string"""
        if self.name == "COSINE":
            return "cosine"
        elif self.name == "EUCLIDIAN":
            return "euclidian"


class Embedding_Models(Enum):
    """Enum class for the different supported embeddings models

    Those models can be separated into two categories
    1) transformers
    2) sentence transformers

    The second ones are considered better equipped for our job
    """

    ROBERTA_LARGE = 1
    QWEN_2_5_7B = 2
    STELLA_EN = 3
    EMBER_V1 = 4
    MINI_LM_V6 = 5
    MINI_LM_L12_V2 = 6
    BGE_M3 = 7
    BGE_EN_ICL = 8
    E5_MISTRAL_7B = 9
    GEMMA_2 = 10
    SFR_EMBEDDING_MISTRAL = 11
    GEMMA_EMBEDDINGS = 12
    FALCON_3 = 13
    PHI_3 = 14

    def __str__(self) -> str:
        """Return the appropriate model name"""
        if self.name == "ROBERTA_LARGE":
            return "xlm-roberta-large"
        elif self.name == "QWEN_2_5_7B":
            return "Qwen/Qwen2.5-7B-Instruct"
        elif self.name == "STELLA_EN":
            return "dunzhang/stella_en_1.5B_v5"
        elif self.name == "EMBER_V1":
            return "llmrails/ember-v1"
        elif self.name == "MINI_LM_V6":
            return "sentence-transformers/all-MiniLM-L6-v2"
        elif self.name == "MINI_LM_L12_V2":
            return "sentence-transformers/all-MiniLM-L12-v2"
        elif self.name == "BGE_M3":
            return "BAAI/bge-m3"
        elif self.name == "BGE_EN_ICL":
            return "BAAI/bge-en-icl"
        elif self.name == "E5_MISTRAL_7B":
            return "intfloat/e5-mistral-7b-instruct"
        elif self.name == "GEMMA_2":
            return "google/gemma-2-9b-it"
        elif self.name == "SFR_EMBEDDING_MISTRAL":
            return "Salesforce/SFR-Embedding-Mistral"
        elif self.name == "GEMMA_EMBEDDINGS":
            return "google/Gemma-Embeddings-v1.0"
        elif self.name == "FALCON_3":
            return "tiiuae/Falcon3-10B-Instruct"
        elif self.name == "PHI_3":
            return "microsoft/Phi-3-medium-128k-instruct"


class Models(Enum):
    """Enum class for the different models used.

    The model names must match the model names as they appear
    on the ollama server.

    Attributes:
        LLAMA3_1: The llama3.1:8b model
        PHI_3: The phi3:latest model
        MISTRAL_NEMO: The mistral-nemo:latest model
        QWEN_2_5: The qwen2.5:latest model

        # NOTE: There are no QWEN_INSTRUCT, MISTRAL_NEMO

    Methods:
        __str__: Return the     appropriate model name
    """

    LLAMA_3_1_Q8_INSTRUCT = 9
    LLAMA3_1_INSTRUCT = 1
    PHI_3_INSTRUCT = 3
    MISTRAL_NEMO_INSTRUCT = 4
    QWEN_2_5 = 5
    GEMMA_2 = 6
    SOLAR = 7
    GRANITE_CODE = 8
    FALCON_3 = 10

    def __str__(self) -> str:
        """Return the appropriate model name

        Override the __str__ method to return the
        correct name
        """
        if self.name == "LLAMA3_1_INSTRUCT":
            return "llama3.1:8b-instruct-q5_K_M"
        elif self.name == "LLAMA_3_1_Q8_INSTRUCT":
            return "llama3.1:8b-instruct-q8_0"
        elif self.name == "PHI_3_INSTRUCT":
            return "phi3.5:3.8b-mini-instruct-q8_0"
        elif self.name == "MISTRAL_NEMO_INSTRUCT":
            return "mistral-nemo:12b-instruct-2407-q5_K_M"
        elif self.name == "QWEN_2_5":
            return "qwen2.5:14b"
        elif self.name == "GEMMA_2":
            return "gemma2:9b-instruct-q5_K_M"
        elif self.name == "SOLAR":
            return "solar-pro:22b"
        elif self.name == "GRANITE_CODE":
            return "granite-code:20b-instruct-8k-q5_K_M"
        elif self.name == "FALCON_3":
            return "falcon3:10b-instruct-q8_0"


class PromptTypes(Enum):
    # Prompts without examples
    MATCHING_PROMPT = 1
    MATCHING_PROMPT_ONLY = 2
    COMPARING_PROMPT = 3
    COMPARING_PROMPT_ONLY = 4
    SELECTING_PROMPT = 5
    SELECTING_PROMPT_ONLY = 6

    def __str__(self) -> str:
        return self.name


class PromptGroups(Enum):
    MATCHING_GROUP = {PromptTypes.MATCHING_PROMPT, PromptTypes.MATCHING_PROMPT_ONLY}
    COMPARING_GROUP = {PromptTypes.COMPARING_PROMPT, PromptTypes.COMPARING_PROMPT_ONLY}
    SELECTING_GROUP = {PromptTypes.SELECTING_PROMPT, PromptTypes.SELECTING_PROMPT_ONLY}

    def __str__(self) -> str:
        return self.value


class ValidationStatus(Enum):
    # Possible validation statuses

    # The following indicate an error
    NO_RESPONSE = 1
    INVALID = 2
    # The following indicate sucess
    VALID = 3

    def __str__(self) -> str:
        return self.name


class BruteForceMode(Enum):
    # The possible brute force modes
    MATCHING = 1
    COMPARING = 3
    SELECTING = 5

    def __str__(self) -> str:
        return self.name


class Prompt:
    """This class contains the prompts for the different prompt types."""

    # Our addition to the prompts presented bellow is the YES, NO Limitation and the
    # The Anser is:
    # This last part is inspired from DAIL-SQL Code: https://github.com/BeachWang/DAIL-SQL
    # and paper https://arxiv.org/pdf/2308.15363

    prompts = {
        # This is the matching prompt from the paper https://arxiv.org/pdf/2405.16884
        PromptTypes.MATCHING_PROMPT: """Do the two entity records refer to the same real-world entity?
        Answer "Yes" if they do and "No" if they do not.
        Record 1: RECORD_PLACEHOLDER
        Record 2: RECORD_OPTION_PLACEHOLDER_1
        """,
        # This is our enhanced version of this prompt. We use the"ONLY" keywords to restrict the
        # model's responces
        # We have also added The answer is: (Inspired from DAIL-SQL: https://arxiv.org/abs/2308.15363)
        PromptTypes.MATCHING_PROMPT_ONLY: """Do the two entity records refer to the same real-world entity?
        Answer ONLY "Yes" if they do and ONLY "No" if they do not.
        Record 1: RECORD_PLACEHOLDER
        Record 2: RECORD_OPTION_PLACEHOLDER_1

        The answer is:
        """,
        # This os the comparing prompt from the paper https://arxiv.org/pdf/2405.16884
        PromptTypes.COMPARING_PROMPT: """Which of the following two records is more likely to refer to the same real-world entity as the given record? Answer with the corresponding record identifier "Record A" or "Record B"
        Given entity record: RECORD_PLACEHOLDER
        Record A: RECORD_OPTION_PLACEHOLDER_1
        Record B: RECORD_OPTION_PLACEHOLDER_2 
        """,
        # This is our enhanced version of this prompt. We use the"ONLY", "and nothing else" keywords to restrict the
        # model's responces
        PromptTypes.COMPARING_PROMPT_ONLY: """Which of the following two records is more likely to refer to the same real-world entity as the given record? Answer ONLY with the corresponding record identifier "Record A" or "Record B" and nothing else.
        Given entity record: RECORD_PLACEHOLDER
        Record A: RECORD_OPTION_PLACEHOLDER_1
        Record B: RECORD_OPTION_PLACEHOLDER_2
        
        The answer is: 
        """,
        # This is the SELECTING prompt from the paper https://arxiv.org/pdf/2405.16884
        PromptTypes.SELECTING_PROMPT: """
        Select a record from the following candidates that refers to the same real-world entity as the given record. Answer with the corresponding record number surrounded by "[]" or "[0]" if there is none.
        Given entity record: RECORD_PLACEHOLDER
        """,
        # TODO: SELECTING_PROMPT ONLY
    }

    @classmethod
    def get_prompt(cls, prompt_type: PromptTypes) -> Union[str, None]:
        """Return the prompt for the given prompt type.

        Args:
            prompt_type (PromptTypes): The prompt type

        Returns:
            str: The prompt
        """
        try:
            return cls.prompts[prompt_type]
        except KeyError:
            print(f"Requested prompt type {prompt_type} is not a valid option!")

    @classmethod
    def add_records_to_prompt(
        cls, prompt_type: PromptTypes, record: str, candidate_records: list
    ) -> str:
        """Add the record and its possible options to the prompt

        The initial prompt contains placeholders for the record and the record options
        Depending on the type of prompt we will use a diffent way to add the records to the prompt
        """

        prompt = cls.get_prompt(prompt_type=prompt_type)

        # If it's the MATCHING
        if prompt_type in PromptGroups.MATCHING_GROUP.value:
            # The record options must have a size of 1
            assert (
                len(candidate_records) == 1
            ), "For the MATCHING PROMPT, the record options must contain ONLY ONE OPTION"

            prompt = prompt.replace("RECORD_PLACEHOLDER", record).replace(
                "RECORD_OPTION_PLACEHOLDER_1", candidate_records[0]
            )
            return prompt

        # If it's the comparing prompt
        elif prompt_type in PromptGroups.COMPARING_GROUP.value:
            # The record options must have a size of 2
            assert (
                len(candidate_records) == 2
            ), "For the COMPARING PROMPT, the record options must contain EXACTLY TWO OPTIONS"

            prompt = (
                prompt.replace("RECORD_PLACEHOLDER", record)
                .replace("RECORD_OPTION_PLACEHOLDER_1", candidate_records[0])
                .replace("RECORD_OPTION_PLACEHOLDER_2", candidate_records[1])
            )

            return prompt

        # If it's the SELECTING
        if prompt_type == PromptTypes.SELECTING_PROMPT:
            prompt = prompt.replace("RECORD_PLACEHOLDER", record)
            # Add the record options
            for idx, record_option in enumerate(candidate_records):
                prompt += f"[{idx + 1}] {record_option}\n"

            return prompt
