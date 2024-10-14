import requests
from src.utils.enums import PromptTypes, Models, Prompts
import time


class LLM:
    def __init__(
        self,
        prompt_type: PromptTypes,
        model: Models,
        username: str,
        password: str,
        server_url: str,
        server_port: int = 8200,  # the port Caddy is running on, specified on docker-compose.yml
        max_request_tries: int = 3,
        request_timeout: int = 60,
        request_interval: int = 5,
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

        self.prompt_type = prompt_type
        self.model = model

        self.prompt = Prompts.get_prompt(prompt_type)

    def send_request(self) -> str:
        """Send a request to the LLM server.

        This function will send a request to the LLM server
        and return the response.

        All requests of interest must be sent at /api/generate

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

        #TODO: Parameter tunning such as temperature
        # From 
        # 1) https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
        # 2) https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        data = {
            "model": str(self.model),
            "prompt": self.prompt,
            "stream" : False
        }

        for i in range(self.max_request_tries):
            try:
                # send the request
                response = requests.post(url, headers=headers, json=data, auth=auth, timeout=self.request_timeout)
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


    # TODO: Add a method to generate statistics from the response
    # Hint: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion