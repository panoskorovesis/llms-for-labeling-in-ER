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

    Methods:
        __str__: Return the appropriate model name
    """

    LLAMA3_1 = 1
    PHI_3 = 2
    MISTRAL_NEMO = 4
    QWEN_2_5 = 5

    def __str__(self) -> str:
        """Return the appropriate model name

        Override the __str__ method to return the
        correct name
        """
        if self.name == "LLAMA3_1":
            return "llama3.1:8b"
        elif self.name == "PHI_3":
            return "phi3:latest"
        elif self.name == "MISTRAL_NEMO":
            return "mistral-nemo:latest"
        elif self.name == "QWEN_2_5":
            return "qwen2.5:latest"


class PromptTypes(Enum):
    # Prompts without examples
    GENERIC_PROMPT = 1
    FINETUNED_PROMPT = 2

    # Prompts with examples
    EASY_EXAMPLES_PROMPT = 3
    MEDIUM_EXAMPLES_PROMPT = 4
    HARD_EXAMPLES_PROMPT = 5
    MIXED_EXAMPLES_PROMPT = 6

    def __str__(self) -> str:
        return self.name


class Prompts:
    """This class contains the prompts for the different prompt types."""

    prompts = {
        PromptTypes.GENERIC_PROMPT: """Determine if the two following product descriptions correspond to the same product.
        First Description: PRODUCT_PLACEHOLDER_1
        Second Description: PRODUCT_PLACEHOLDER_2
        
        Answer ONLY WITH "YES", "NO".

        The Answer is: """,
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
