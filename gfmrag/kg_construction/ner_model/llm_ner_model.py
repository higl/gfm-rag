# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/named_entity_extraction_parallel.py
import logging
from typing import Literal
import re

from langchain_community.chat_models import ChatLlamaCpp
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from gfmrag.kg_construction.langchain_util import init_langchain_model
from gfmrag.kg_construction.utils import extract_json_dict, processing_phrases
from gfmrag.kg_construction.openie_extraction_instructions import ner_prompts

from .base_model import BaseNERModel

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

query_prompt_template = """
Question: {}

"""


class LLMNERModel(BaseNERModel):
    """A Named Entity Recognition (NER) model that uses Language Models (LLMs) for entity extraction.

    This class implements entity extraction using various LLM backends (OpenAI, Together, Ollama, llama.cpp)
    through the Langchain interface. It processes text input and returns a list of extracted named entities.

    Args:
        llm_api (Literal["openai", "nvidia", "together", "ollama", "llama.cpp"]): The LLM backend to use. Defaults to "openai".
        model_name (str): Name of the specific model to use. Defaults to "gpt-4o-mini".
        max_tokens (int): Maximum number of tokens in the response. Defaults to 1024.

    Methods:
        __call__: Extracts named entities from the input text.

    Raises:
        Exception: If there's an error in extracting or processing named entities.
    """

    def __init__(
        self,
        llm_api: Literal[
            "openai", "nvidia", "together", "ollama", "llama.cpp"
        ] = "openai",
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 1024,
    ):
        """Initialize the LLM-based NER model.

        Args:
            llm_api (Literal["openai", "nvidia", "together", "ollama", "llama.cpp"]): The LLM API provider to use.
                Defaults to "openai".
            model_name (str): Name of the language model to use.
                Defaults to "gpt-4o-mini".
            max_tokens (int): Maximum number of tokens for model output.
                Defaults to 1024.
        """

        self.llm_api = llm_api
        self.model_name = model_name
        self.max_tokens = max_tokens

        self.client = init_langchain_model(llm_api, model_name)

    @staticmethod
    def _unwrap_response(resp: any) -> any:
        """Return the underlying content/dict/string from a LangChain/LLM response object."""
        # LangChain message-like objects often expose `.content`
        try:
            if hasattr(resp, "content"):
                return resp.content
        except Exception:
            pass
        # Some clients may return a dict already
        if isinstance(resp, dict):
            return resp
        # If it's a string, return as-is
        if isinstance(resp, str):
            return resp
        # If it's list/tuple, return as-is
        if isinstance(resp, (list, tuple)):
            return resp
        # Fallback: try to coerce to string
        try:
            return str(resp)
        except Exception:
            return ""

    def __call__(self, text: str) -> list:
        """
        Call the LLM NER pipeline and return a list (possibly empty) of named entities.
        """
        # Build prompt object using the shared prompt template (same as other modules)
        try:
            prompt_obj = ner_prompts.format_prompt(user_input=text)
        except Exception:
            # Fallback to simple template string if prompt object not available
            prompt_obj = None
            raw_prompt = query_prompt_template.format(text)

        response_content = []

        try:
            # For ChatOpenAI, ChatOllama, ChatLlamaCpp, call with prompt_obj.to_messages() when available
            if isinstance(self.client, ChatOpenAI):
                if prompt_obj is not None and hasattr(prompt_obj, "to_messages"):
                    chat_completion = self.client.invoke(
                        prompt_obj.to_messages(),
                        temperature=0,
                        max_tokens=self.max_tokens,
                        stop=["\n\n"],
                        response_format={"type": "json_object"},
                    )
                else:
                    chat_completion = self.client.invoke(
                        raw_prompt if prompt_obj is None else prompt_obj,
                        temperature=0,
                        max_tokens=self.max_tokens,
                        stop=["\n\n"],
                        response_format={"type": "json_object"},
                    )
                raw = self._unwrap_response(chat_completion)
                if isinstance(raw, str):
                    response_content = extract_json_dict(raw)
                else:
                    response_content = raw

            elif isinstance(self.client, ChatOllama) or isinstance(self.client, ChatLlamaCpp):
                # LangChain wrappers may return AIMessage-like objects; use prompt_obj when possible
                if prompt_obj is not None and hasattr(prompt_obj, "to_messages"):
                    resp = self.client.invoke(prompt_obj.to_messages())
                else:
                    resp = self.client.invoke(raw_prompt if prompt_obj is None else prompt_obj)
                raw = self._unwrap_response(resp)
                if isinstance(raw, str):
                    response_content = extract_json_dict(raw)
                else:
                    response_content = raw

            else:
                # Generic client path
                if prompt_obj is not None and hasattr(prompt_obj, "to_messages"):
                    chat_completion = self.client.invoke(prompt_obj.to_messages(), temperature=0)
                else:
                    chat_completion = self.client.invoke(raw_prompt if prompt_obj is None else prompt_obj, temperature=0)
                raw = self._unwrap_response(chat_completion)
                if isinstance(raw, str):
                    response_content = extract_json_dict(raw)
                else:
                    response_content = raw

            # Normalize result to list of entities if possible
            if isinstance(response_content, dict) and "named_entities" in response_content:
                entities = response_content["named_entities"]
            else:
                entities = response_content

            # Fallback normalization to list of strings
            normalized = []
            if isinstance(entities, list):
                for e in entities:
                    if isinstance(e, dict):
                        txt = e.get("text") or e.get("entity") or e.get("name") or e.get("value") or ""
                        if txt:
                            normalized.append(str(txt))
                    elif isinstance(e, (str, int, float)):
                        normalized.append(str(e))
            elif isinstance(entities, dict):
                # keys or values may encode entities
                for v in entities.values():
                    if isinstance(v, list):
                        for it in v:
                            normalized.append(str(it))
                    else:
                        normalized.append(str(v))
            elif isinstance(entities, str):
                # comma/line separated
                normalized = [p.strip() for p in re.split(r"[,\n;]+", entities) if p.strip()]

        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
            normalized = []

        # deduplicate & return
        seen = set(); out = []
        for x in normalized:
            if x not in seen and x:
                seen.add(x); out.append(x)
        return out
