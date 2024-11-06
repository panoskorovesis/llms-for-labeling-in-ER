from enum import Enum
from typing import Union


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
        __str__: Return the appropriate model name
    """

    LLAMA3_1_INSTRUCT = 1
    PHI_3_INSTRUCT = 3
    MISTRAL_NEMO_INSTRUCT = 4
    QWEN_2_5 = 5

    def __str__(self) -> str:
        """Return the appropriate model name

        Override the __str__ method to return the
        correct name
        """
        if self.name == "LLAMA3_1_INSTRUCT":
            return "llama3.1:8b-instruct-q5_K_M"
        elif self.name == "PHI_3_INSTRUCT":
            return "phi3.5:3.8b-mini-instruct-q8_0"
        elif self.name == "MISTRAL_NEMO_INSTRUCT":
            return "mistral-nemo:12b-instruct-2407-q5_K_M"
        elif self.name == "QWEN_2_5":
            return "qwen2.5:14b"


class PromptTypes(Enum):
    # Prompts without examples
    MATCHING_PROMPT = 1
    COMPARING_PROMPT = 2
    SELECTION_PROMPT = 3
    FINETUNED_PROMPT = 4

    # Prompts with examples
    EASY_EXAMPLES_PROMPT = 5
    MEDIUM_EXAMPLES_PROMPT = 6
    HARD_EXAMPLES_PROMPT = 7
    MIXED_EXAMPLES_PROMPT = 8

    def __str__(self) -> str:
        return self.name


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
    COMPARING = 2
    SELECTING = 3

    def __str__(self) -> str:
        return self.name

class Prompt:
    """This class contains the prompts for the different prompt types."""

    # Our addition to the prompts presented bellow is the YES, NO Limitation and the
    # The Anser is:
    # This last part is inspired from DAIL-SQL Code: https://github.com/BeachWang/DAIL-SQL
    # and paper https://arxiv.org/pdf/2308.15363

    # This is the matching prompt from the paper https://arxiv.org/pdf/2405.16884
    prompts = {
        PromptTypes.MATCHING_PROMPT: """Do the two entity records refer to the same real-world entity?
Answer "Yes" if they do and "No" if they do not.
Record 1: RECORD_PLACEHOLDER
Record 2: RECORD_OPTION_PLACEHOLDER_1
""",
        # This os the comparing prompt from the paper https://arxiv.org/pdf/2405.16884
        PromptTypes.COMPARING_PROMPT: """Which of the following two records is more likely to refer to the same real-world entity as the given record? Answer with the corresponding record identifier "Record A" or "Record B"
        Given entity record: RECORD_PLACEHOLDER
        Record A: RECORD_OPTION_PLACEHOLDER_1
        Record B: RECORD_OPTION_PLACEHOLDER_2

        The Answer is:""",
        # This is the selection prompt from the paper https://arxiv.org/pdf/2405.16884
        PromptTypes.SELECTION_PROMPT: """
        Select a record from the following candidates that refers to the same real-world entity as the given record. Answer with the corresponding record number surrounded by "[]" or "[0]" if there is none.
        Given entity record: RECORD_PLACEHOLDER
        """,
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
        if prompt_type == PromptTypes.MATCHING_PROMPT:
            # The record options must have a size of 1
            assert (
                len(candidate_records) == 1
            ), "For the MATCHING PROMPT, the record options must contain ONLY ONE OPTION"

            prompt = prompt.replace("RECORD_PLACEHOLDER", record).replace(
                "RECORD_OPTION_PLACEHOLDER_1", candidate_records[0]
            )
            return prompt

        # If it's the comparing prompt
        elif prompt_type == PromptTypes.COMPARING_PROMPT:
            # The record options must have a size of 2
            assert (
                len(candidate_records) == 2
            ), "For the COMPARING PROMPT, the record options must contain EXACTLY TWO OPTIONS"

            prompt = (
                prompt.replace("RECORD_PLACEHOLDER", record)
                .replace("RECORD_OPTION_PLACEHOLDER_1", candidate_records[0])
                .replace("RECORD_OPTION_PLACEHOLDER_2", candidate_records[1])
            )

        # If it's the SELECTION
        if prompt_type == PromptTypes.SELECTION_PROMPT:
            prompt = prompt.replace("RECORD_PLACEHOLDER", record)
            # Add the record options
            for idx, record_option in enumerate(candidate_records):
                prompt += f"[{idx}] {record_option}\n"

            # Finally add the Answer is
            prompt += "\n The Answer is:"

            return prompt
