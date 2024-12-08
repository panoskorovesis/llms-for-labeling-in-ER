import time
from typing import Union
from nltk.tokenize import RegexpTokenizer
import requests
import tiktoken

from src.utils.enums import Models, Prompt, PromptTypes, ValidationStatus, PromptGroups


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

        if prompt_type in PromptGroups.MATCHING_GROUP.value:
            if text == "no" or text.startswith("no "):
                return "no"
            elif text == "yes" or text.startswith("yes "):
                return "yes"
            else:
                raise ValueError(
                    f"Text did not contain expected content for {prompt_type}. Text: {text}"
                )
        elif prompt_type in PromptGroups.COMPARING_GROUP.value:
            # First we have to check of the in case to not mess with the rest
            if "record a is more likely to refer to the same real world entity" in text:
                return "record a"

            elif (
                "record b is more likely to refer to the same real world entity" in text
            ):
                return "record b"

            elif text == "record a" or text == "record b":
                return text
            # NOTE: The double space here is because at this step we have replace punkt
            # with a space
            elif text.startswith("the answer is record a") or text.startswith(
                "the answer is  record a"
            ):
                return "record a"
            elif text.startswith("the answer is record b") or text.startswith(
                "the answer is  record b"
            ):
                return "record b"
            else:
                raise ValueError(
                    f"Text did not contain expected content for {prompt_type}. Text: {text}"
                )
        else:
            raise NotImplementedError(
                f"clean_response() is not implemented for {prompt_type}"
            )

    def validate_response(
        self, rsp: str, prompt_type: PromptTypes, max_number: int = None
    ) -> ValidationStatus:
        """Depending on the prompt type make sure that the rsp is the expected one

        The first check is to make sure "response" key is present

        Due to the fact that llms like to add extra information we wil check how the sentence starts
        We have some possible cases. First we have to  remove the punkt

        1) starts with "No " or "Yes " or some equivalent of the other prompts
        2) is exactly "No", "Yes", "Record A" etc

        If it's none of the above we will print it to see how we can handle it
        """

        if rsp.get("response") is None:
            return ValidationStatus.NO_RESPONSE, ""

        # First clean the text
        text = self.remove_punctuation(rsp["response"]).lower()

        print(
            f"------------------RAW RSP---------------------\n{text}\n----------------------------------------------\n\n"
        )

        if prompt_type in PromptGroups.MATCHING_GROUP.value:
            # Here the expected answers should be "Yes" or "No"
            if (
                text == "no"
                or text == "yes"
                or text.startswith("no ")
                or text.startswith("yes ")
            ):
                return ValidationStatus.VALID, self.clean_response(
                    text, prompt_type=prompt_type
                )
            else:
                if self.verbose:
                    print(f"INVALID for prompt: {prompt_type}. TEXT: {text}")

                return ValidationStatus.INVALID, ""

        # here we are looking for two two things
        # 1) "record a" or "record b"
        # This is more frequent in the ONLY ... and nothing else prompt
        # 2) "the answer is record a" or "the answer is record b"
        # This is more frequent in the same prompt as the paper
        elif prompt_type in PromptGroups.COMPARING_GROUP.value:
            if (
                text == "record a"
                or text == "record b"
                or text.startswith("the answer is record a")
                or text.startswith("the answer is record b")
                # NOTE: The double space here is because at this step we have replace punkt
                # with a space
                or text.startswith("the answer is  record b")
                or text.startswith("the answer is  record a")
                # NOTE: Some answers are also valid but do not contain the required information at the start
                or "record a is more likely to refer to the same real world entity"
                in text
                or "record b is more likely to refer to the same real world entity"
                in text
            ):
                return ValidationStatus.VALID, self.clean_response(text, prompt_type)
            else:
                if self.verbose:
                    print(f"INVALID for prompt: {prompt_type}. TEXT: {text}")

                return ValidationStatus.INVALID, ""

        elif prompt_type in PromptGroups.SELECTING_GROUP.value:
            """Here the value must be a valid number
            Normally it should be in [] but the models tend to ignore this command
            So we want to check THE FIRST WORD
            """
            word = text.strip().split(" ")[0]

            try:
                if int(word) <= max_number:
                    return ValidationStatus.VALID, int(word)
                # Answer is out of bounds
                else:
                    print(
                        f"Answer OUT OF BOUNDS. Total Options: {max_number}. Answer: {word}"
                    )
                    return ValidationStatus.INVALID, ""
            except Exception as e:
                if self.verbose:
                    print(f"Validation failed: {e}")
                return ValidationStatus.INVALID, ""

        else:
            raise ValueError(f"Prompt Type: {prompt_type} is not Supported!")

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

    def truncate_prompt_if_needed(self, text: str, max_tokens: int = 2048):
        """If the prompt exeeds the maximum tokens then we will truncate it

        To get an estimation of the tokens we will use tiktoken package
        We will add 10% to the tokens returned by this package, in order to be sure
        We will also use gpt-4 as a reference model
        """
        # create the encoder
        encoder = tiktoken.encoding_for_model(model_name="gpt-4")
        # count the tokens
        tokens = encoder.encode(text)

        if self.verbose:
            print(f"@@@@ TOTAL TOKENS: {len(tokens)}")

        # if they are > max truncate
        # We will keep max_tokens - 300 just to be safe
        # We will also keep the last 100 tokens as they may contain important information
        # We can see that PHI3 has an issue when the tokens are above 1700
        # We will set the max_tokens accordingly
        #TODO: What are we going to do with this? Keep it or not?
        if self.model == Models.PHI_3_INSTRUCT and 1==0:
            max_tokens = 1650
        # for the rest of the cases, given that's it's an estimation we take 100 out of the max to be safe
        else:
            max_tokens -= 100

        if len(tokens) > max_tokens:
            if self.verbose:
                print(
                    f"Prompt has {len(tokens)} tokens! We will keep: {max_tokens - 200}"
                )

            tokens_to_keep = max_tokens - 700
            # Truncate, keeping the last 100
            truncated_tokens = tokens[:tokens_to_keep] + tokens[-100:]

            if self.verbose:
                print(f"Remaining tokens: {len(truncated_tokens)}")

            # Decode the tokens back into text
            text = encoder.decode(truncated_tokens)

        return text

    def ask_llm(
        self,
        prompt_type: PromptTypes,
        record: str,
        candidate_records: list,
        temperature: float,
        num_predict: int = 128,
        max_tokens=2048,
    ) -> Union[str, None]:
        """Ask the llm using a specific prompt, return the rsp if it's valid else None"""

        # First get the prompt template
        prompt = Prompt.get_prompt(prompt_type)

        # Add the data
        prompt = Prompt.add_records_to_prompt(
            prompt_type=prompt_type, record=record, candidate_records=candidate_records
        )

        # If the prompt size is bigger than the max then we have to truncate
        # Reminder: Default size is 2048 tokens
        # To be sure, since we are using an estimation we will compromize to max - 200
        # This will only be applied if the user requested the max tokens or more
        if max_tokens >= 2048:
            max_tokens = 1848

        prompt = self.truncate_prompt_if_needed(prompt, max_tokens=max_tokens)

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
        # For the SELECTING Prompt we need a max_number equal to the number of canditates
        # We can set this for all records since it's only used in the selecting
        # This is done to simplify the code
        status, clean_response = self.validate_response(
            rsp, prompt_type, max_number=len(candidate_records)
        )

        return status, clean_response

    # TODO: Add a method to generate statistics from the response
    # Hint: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
