import logging
import os
import time

import dotenv
from langchain_ollama import ChatOllama

from .base_language_model import BaseLanguageModel
from gfmrag.kg_construction.langchain_util import init_langchain_model
from gfmrag.kg_construction.utils import extract_json_dict

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


class Ollama(BaseLanguageModel):
    """A class that interacts with Ollama through its API.

    This class provides functionality to generate text using Ollama models while handling
    token limits, retries, and various input formats.

    Args:
        model_name (str): The name or path of the Ollama model to use
        retry (int, optional): Number of retries for failed API calls. Defaults to 5

    Attributes:
        retry (int): Maximum number of retry attempts for failed API calls
        model_name (str): Name of the Ollama model being used
        maximun_token (int): Maximum token limit for the specified model
        client (ChatOllama): ChatOllama client instance for API interactions

    Methods:
        token_len(text): Calculate the number of tokens in a given text
        generate_sentence(llm_input, system_input): Generate response using the Ollama model

    Raises:
        KeyError: If the specified model is not found when calculating tokens
        Exception: If generation fails after maximum retries
    """

    def __init__(self, model_name: str, maximum_token: int,retry: int = 5):
        self.retry = retry
        self.model_name = model_name
        self.maximun_token = maximum_token

        self.client = init_langchain_model('ollama', self.model_name)

    def token_len(self, text: str) -> int:
        """Returns the number of tokens used by a list of messages."""
        
        return 0

    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """Generate a response using the Ollama API.

        This method sends a request to the Ollama API and returns the generated response.
        It handles both single string inputs and message lists, with retry logic for failed attempts.

        Args:
            llm_input (Union[str, list]): Either a string containing the user's input or a list of message dictionaries
                in the format [{"role": "role_type", "content": "message_content"}, ...]
            system_input (str, optional): System message to be prepended to the conversation. Defaults to "".

        Returns:
            Union[str, Exception]: The generated response text if successful, or the Exception if all retries fail.
                The response is stripped of leading/trailing whitespace.

        Raises:
            Exception: If all retry attempts fail, returns the last encountered exception.

        Notes:
            - Automatically truncates inputs that exceed the maximum token limit
            - Uses exponential backoff with 30 second delays between retries
            - Sets temperature to 0.0 for deterministic outputs
            - Timeout is set to 60 seconds per API call
        """

        # If the input is a list, it is assumed that the input is a list of messages
        if isinstance(llm_input, list):
            message = llm_input
        else:
            message = []
            if system_input:
                message.append({"role": "system", "content": system_input})
            message.append({"role": "user", "content": llm_input})
        cur_retry = 0
        num_retry = self.retry
        # Check if the input is too long
        message_string = "\n".join([m["content"] for m in message])
        input_length = self.token_len(message_string)
        if input_length > self.maximun_token:
            print(
                f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens."
            )
            llm_input = llm_input[: self.maximun_token]
        error = Exception("Failed to generate sentence")
        while cur_retry <= num_retry:
            try:
                response = self.client.invoke(message).content
                response = extract_json_dict(response) 
                return response
            except Exception as e:
                logger.error("Message: ", llm_input)
                logger.error("Number of token: ", self.token_len(message_string))
                logger.error(e)
                #time.sleep(30)
                cur_retry += 1
                error = e
                continue
        return error
