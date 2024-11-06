import time
from typing import Union
from nltk.tokenize import RegexpTokenizer
import requests

from src.utils.enums import Models, Prompt, PromptTypes, ValidationStatus


class LLM:
    def __init__(
        self,
        model: Models,
        username: str,
        password: str,
        server_url: str,
        server_port: int = 8200,  # the port Caddy is running on, specified on docker-compose.yml
        max_request_tries: int = 3,
        request_timeout: int = 60,
        request_interval: int = 5,
        verbose: str = False,
    ):
        # The username, password are required to
        # pass the Caddy authentication
        self.username = username
        self.password = password
        self.server_url = server_url

        self.max_request_tries = max_request_tries
        self.request_timeout = request_timeout
        self.request_interval = request_interval

        self.server_port = server_port

        self.model = model

        self.verbose = verbose

        if self.verbose:
            print(f"LLM initialized with model: {model}")

    def remove_punctuation(self, sentence: str):
        """Remove punctuation using nltk RegexpTokenizer

        Source: https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer

        Tokens contains a list of all data except punct. We will merge them again with " " and return them
        """
        tokenizer = RegexpTokenizer(r"\w+")

        tokens = tokenizer.tokenize(sentence)

        return " ".join(tokens).strip()

    def clean_response(
        self,
        text: str,
        prompt_type: PromptTypes,
    ):
        """Extract only the required information from the model's response

        This is dependant on the prompt_type
        """

        if prompt_type == PromptTypes.MATCHING_PROMPT:
            if text == "no" or text.startswith("no "):
                return "no"
            elif text == "yes" or text.startswith("yes "):
                return "yes"
            else:
                raise ValueError(
                    f"Text did not contain expected content for {prompt_type}. Text: {text}"
                )

    def validate_response(self, rsp: str, prompt_type: PromptTypes) -> ValidationStatus:
        """Depending on the prompt type make sure that the rsp is the expected one

        The first check is to make sure "response" key is present

        Due to the fact that llms like to add extra information we wil check how the sentence starts
        We have some possible cases. First we have to  remove the punkt

        1) starts with "No " or "Yes " or some equivalent of the other prompts
        2) is exactly "No", "Yes", "Record A" etc

        If it's none of the above we will print it to see how we can handle it
        """

        if rsp.get("response") is None:
            return ValidationStatus.NO_RESPONSE

        # First clean the text
        text = self.remove_punctuation(rsp["response"]).lower()

        if prompt_type == PromptTypes.MATCHING_PROMPT:
            # Here the expected answers should be "Yes" or "No"
            if (
                text == "no"
                or text == "yes"
                or text.startswith("no ")
                or text.startswith("yes ")
            ):
                return ValidationStatus.VALID, self.clean_response(text, prompt_type=prompt_type)
            else:
                if self.verbose:
                    print(f"INVALID for prompt: {prompt_type}. TEXT: {text}")

                return ValidationStatus.INVALID, ""

        # TODO: Continue this
        raise NotImplementedError(
            f"Validation for {prompt_type} is not yet implemented!"
        )

    def send_request(
        self, prompt: str, temperature: float, num_predict: int = 128
    ) -> str:
        """Send a request to the LLM server.

        This function will send a request to the LLM server
        and return the response.

        All requests of interest must be sent at /api/generate

        Some useful parameters are
        temperature: The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
        num_predict: Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
        Returns:
            str: The response from the LLM server.
        """

        # Prepare the variables for the request
        url = f"{self.server_url}:{self.server_port}/api/generate"

        # For the authentication we need Basic Auth as we have a username and password
        headers = {
            "Content-Type": "application/json",
        }

        auth = (self.username, self.password)

        # TODO: Parameter tunning such as temperature
        # From
        # 1) https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
        # 2) https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        data = {
            "model": str(self.model),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }

        for i in range(self.max_request_tries):
            try:
                # send the request
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    auth=auth,
                    timeout=self.request_timeout,
                )
                # in case there is an error, raise an exception
                if response.status_code != 200:
                    response.raise_for_status()

                # everyting is ok, we can return the response text
                # given that the response is in json format we can directly load it and return it
                return response.json()

            except Exception as e:
                print(f"Error while sending request: {e}")
                # sleep using the request interval and exponential backoff
                time.sleep(self.request_interval * i)
                # continue to the next iteration
                continue

        # if we reach this point, we have failed to send the request
        return None

    def ask_llm(
        self,
        prompt_type: PromptTypes,
        record: str,
        candidate_records: list,
        temperature: float,
        num_predict: int = 128,
    ) -> Union[str, None]:
        """Ask the llm using a specific prompt, return the rsp if it's valid else None"""

        # First get the prompt template
        prompt = Prompt.get_prompt(prompt_type)

        # Add the data
        prompt = Prompt.add_records_to_prompt(
            prompt_type=prompt_type, record=record, candidate_records=candidate_records
        )

        if self.verbose:
            print(f"{self.model} Will be asked:\n{prompt}")

        # send the request
        rsp = self.send_request(
            prompt=prompt, temperature=temperature, num_predict=num_predict
        )

        # in case of error -> None
        if rsp is None or rsp.get("error") is not None:
            print(
                f"Request to the api for prompt_type: {prompt_type}, record: {record} FAILED! Will return None"
            )
            return None

        # Validate the rsp
        status, clean_response = self.validate_response(rsp, prompt_type)

        return status, clean_response

    # TODO: Add a method to generate statistics from the response
    # Hint: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
