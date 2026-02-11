# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/openie_with_retrieval_option_parallel.py
import json
import re
import logging
from itertools import chain
from typing import Literal, Any, List, Optional

import numpy as np
from langchain_community.chat_models import ChatLlamaCpp
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from gfmrag.kg_construction.langchain_util import init_langchain_model
from gfmrag.kg_construction.openie_extraction_instructions import (
    ner_prompts,
    openie_post_ner_prompts,
)
from gfmrag.kg_construction.utils import extract_json_dict

from .base_model import BaseOPENIEModel

try:
    import spacy
except Exception:
    spacy = None

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


# lazy spaCy loader
_SPACY_NLP = None
def _get_spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None and spacy is not None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception:
            try:
                # best-effort: try the package name
                import importlib, subprocess, sys
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                _SPACY_NLP = spacy.load("en_core_web_sm")
            except Exception:
                _SPACY_NLP = None
    return _SPACY_NLP


def _normalize_triple_item(x: Any) -> str:
    """Return string representation for triple item; tolerate dict/list/str."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        # join tokens or fields
        return " ".join([str(e).strip() for e in x if e])
    if isinstance(x, dict):
        # pick likely fields
        for k in ("text","name","value","label"):
            if k in x:
                return str(x[k]).strip()
        # fallback: join values
        return " ".join([str(v).strip() for v in x.values() if v])
    return str(x).strip()


def parse_llm_triples(raw_output: Any) -> List[dict]:
    """
    Robustly parse LLM output into list of triples of form:
      {'subject': str, 'predicate': str, 'object': str}
    Accepts various shapes: list of dicts, dict, list of lists/tuples, plain string (attempt regex).
    """
    triples: List[dict] = []
    try:
        if raw_output is None:
            return []
        # If it's already a list
        if isinstance(raw_output, list):
            for item in raw_output:
                if isinstance(item, dict):
                    s = _normalize_triple_item(item.get("subject") or item.get("subj") or item.get("s"))
                    p = _normalize_triple_item(item.get("predicate") or item.get("pred") or item.get("p"))
                    o = _normalize_triple_item(item.get("object") or item.get("obj") or item.get("o"))
                    if s or p or o:
                        triples.append({"subject": s, "predicate": p, "object": o})
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    s = _normalize_triple_item(item[0])
                    p = _normalize_triple_item(item[1]) if len(item) > 1 else ""
                    o = _normalize_triple_item(item[2]) if len(item) > 2 else ""
                    triples.append({"subject": s, "predicate": p, "object": o})
                elif isinstance(item, str):
                    # try splitting "A - located in - B" or "A located in B"
                    text = item.strip()
                    parts = [p.strip() for p in re.split(r"\s*[-–—]\s*|\s+:\s+|\s*;\s*", text) if p.strip()]
                    if len(parts) >= 3:
                        triples.append({"subject": parts[0], "predicate": parts[1], "object": parts[2]})
                    else:
                        # regex simple "X located in Y"
                        m = re.search(r"(.+?)\s+(located in|is in|in)\s+(.+)", text, re.IGNORECASE)
                        if m:
                            triples.append({"subject": m.group(1).strip(), "predicate": m.group(2).strip(), "object": m.group(3).strip()})
        elif isinstance(raw_output, dict):
            # single triple-like dict
            s = _normalize_triple_item(raw_output.get("subject") or raw_output.get("subj") or raw_output.get("s"))
            p = _normalize_triple_item(raw_output.get("predicate") or raw_output.get("pred") or raw_output.get("p"))
            o = _normalize_triple_item(raw_output.get("object") or raw_output.get("obj") or raw_output.get("o"))
            if s or p or o:
                triples.append({"subject": s, "predicate": p, "object": o})
        elif isinstance(raw_output, str):
            text = raw_output.strip()
            # attempt to find multiple triples separated by newlines
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in re.split(r"\s*[-–—]\s*|\s+:\s+|\s*;\s*", line) if p.strip()]
                if len(parts) >= 3:
                    triples.append({"subject": parts[0], "predicate": parts[1], "object": parts[2]})
                else:
                    m = re.search(r"(.+?)\s+(located in|is in|in|born in|works at|founded)\s+(.+)", line, re.IGNORECASE)
                    if m:
                        triples.append({"subject": m.group(1).strip(), "predicate": m.group(2).strip(), "object": m.group(3).strip()})
        # filter out empty triples
        triples = [t for t in triples if (t.get("subject") or t.get("predicate") or t.get("object"))]
    except Exception as e:
        logger.error("Failed to parse LLM triples robustly: %s", e)
        return []
    return triples


def extract_named_entities_with_fallback(text: str, llm_entities: Optional[List[dict]] = None) -> List[dict]:
    """
    llm_entities: optional list/dict returned by LLM. If empty or invalid, use spaCy to extract PERSON/ORG/LOC etc.
    Returns list of entities in normalized dict form: {'text':..., 'type':..., 'start':..., 'end':...}
    """
    ents: List[dict] = []
    # First try LLM-provided entities if any and valid
    if llm_entities:
        try:
            if isinstance(llm_entities, list):
                for e in llm_entities:
                    if isinstance(e, dict) and ("text" in e or "entity" in e or "label" in e):
                        text_val = e.get("text") or e.get("entity") or e.get("name") or ""
                        label = e.get("type") or e.get("label") or e.get("entity_type") or ""
                        ents.append({"text": text_val.strip(), "type": label, "start": e.get("start"), "end": e.get("end")})
            elif isinstance(llm_entities, dict):
                # single entity
                text_val = llm_entities.get("text") or llm_entities.get("entity") or ""
                label = llm_entities.get("type") or llm_entities.get("label") or ""
                ents.append({"text": text_val.strip(), "type": label, "start": llm_entities.get("start"), "end": llm_entities.get("end")})
        except Exception:
            ents = []

    # If no entities from LLM, try spaCy fallback
    if not ents:
        nlp = _get_spacy_nlp()
        if nlp is not None:
            try:
                doc = nlp(text or "")
                for e in doc.ents:
                    ents.append({"text": e.text, "type": e.label_, "start": e.start_char, "end": e.end_char})
            except Exception as e:
                logger.warning("spaCy NER fallback failed: %s", e)
        else:
            # crude heuristic: pick capitalized noun chunks as entities
            words = re.findall(r"\b[A-Z][a-z0-9\-]{1,}\b(?:\s+[A-Z][a-z0-9\-]{1,}\b)*", text or "")
            for w in words:
                ents.append({"text": w, "type": "MISC"})
    return ents


def _dedup_list(seq: List[Any]) -> List[Any]:
    """Deduplicate preserving order and coerce numpy strings to Python str."""
    out = []
    seen = set()
    for x in seq or []:
        # convert numpy.str_ and other objects to plain str
        try:
            sx = str(x)
        except Exception:
            sx = x
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


class LLMOPENIEModel(BaseOPENIEModel):
    """
    A class for performing Open Information Extraction (OpenIE) using Large Language Models.

    This class implements OpenIE functionality by performing Named Entity Recognition (NER)
    and relation extraction using various LLM backends like OpenAI, Together, Ollama, or llama.cpp.

    Args:
        llm_api (Literal["openai", "together", "ollama", "llama.cpp"]): The LLM backend to use.
            Defaults to "openai".
        model_name (str): Name of the specific model to use. Defaults to "gpt-4o-mini".
        max_ner_tokens (int): Maximum number of tokens for NER output. Defaults to 1024.
        max_triples_tokens (int): Maximum number of tokens for relation triples output.
            Defaults to 4096.

    Attributes:
        llm_api: The LLM backend being used
        model_name: Name of the model being used
        max_ner_tokens: Token limit for NER
        max_triples_tokens: Token limit for relation triples
        client: Initialized language model client

    Methods:
        ner: Performs Named Entity Recognition on input text
        openie_post_ner_extract: Extracts relation triples after NER
        __call__: Main method to perform complete OpenIE pipeline

    Examples:
        >>> openie_model = LLMOPENIEModel()
        >>> result = openie_model("Emmanuel Macron is the president of France")
        >>> print(result)
        {'passage': 'Emmanuel Macron is the president of France', 'extracted_entities': ['Emmanuel Macron', 'France'], 'extracted_triples': [['Emmanuel Macron', 'president of', 'France']]}
    """

    def __init__(
        self,
        llm_api: Literal[
            "openai", "nvidia", "together", "ollama", "llama.cpp"
        ] = "openai",
        model_name: str = "gpt-4o-mini",
        max_ner_tokens: int = 1024,
        max_triples_tokens: int = 4096,
    ):
        """Initialize LLM-based OpenIE model.

        Args:
            llm_api (Literal["openai", "nvidia", "together", "ollama", "llama.cpp"]): The LLM API provider to use.
                Defaults to "openai".
            model_name (str): Name of the language model to use. Defaults to "gpt-4o-mini".
            max_ner_tokens (int): Maximum number of tokens for NER processing. Defaults to 1024.
            max_triples_tokens (int): Maximum number of tokens for triple extraction. Defaults to 4096.

        Attributes:
            llm_api: The selected LLM API provider
            model_name: Name of the language model
            max_ner_tokens: Token limit for NER
            max_triples_tokens: Token limit for triples
            client: Initialized language model client
        """
        self.llm_api = llm_api
        self.model_name = model_name
        self.max_ner_tokens = max_ner_tokens
        self.max_triples_tokens = max_triples_tokens

        self.client = init_langchain_model(llm_api, model_name)

    def ner(self, text: str) -> list:
        """Perform NER via LLM and return a plain list of entity strings (deduplicated)."""
        ner_messages = ner_prompts.format_prompt(user_input=text)

        response_content = []
        try:
            if isinstance(self.client, ChatOpenAI):  # JSON mode
                chat_completion = self.client.invoke(
                    ner_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_ner_tokens,
                    stop=["\n\n"],
                    response_format={"type": "json_object"},
                )
                response_content = chat_completion.content
                # response_content expected to be JSON object
                if isinstance(response_content, str):
                    response_content = extract_json_dict(response_content)

            elif isinstance(self.client, ChatOllama) or isinstance(self.client, ChatLlamaCpp):
                response_content = self.client.invoke(ner_messages.to_messages()).content
                response_content = extract_json_dict(response_content)

            else:  # no JSON mode
                chat_completion = self.client.invoke(ner_messages.to_messages(), temperature=0)
                response_content = chat_completion.content
                response_content = extract_json_dict(response_content)

            # Normalize to list of strings
            if isinstance(response_content, dict) and "named_entities" in response_content:
                raw = response_content["named_entities"]
            else:
                raw = response_content

            # If raw is a dict with entity->type mapping, use keys or values
            entities = []
            if isinstance(raw, list):
                for e in raw:
                    if isinstance(e, dict):
                     # try to extract textual field
                     txt = e.get("text") or e.get("entity") or e.get("name") or e.get("value") or ""
                     entities.append(txt)
                    else:
                        entities.append(e)
            elif isinstance(raw, dict):
                # if dict of lists or dict of dicts
                for v in raw.values():
                    if isinstance(v, list):
                        entities.extend(v)
                    elif isinstance(v, (str, int, float)):
                        entities.append(v)
            elif isinstance(raw, str):
                # maybe comma-separated
                parts = [p.strip() for p in re.split(r"[,\n;]+", raw) if p.strip()]
                entities.extend(parts)
            # dedup & coerce to strings
            entities = _dedup_list(entities)

        except Exception as e:
            logger.error(f"Error in extracting named entities: {e}")
            entities = []

        return entities

    def openie_post_ner_extract(self, text: str, entities: list) -> str:
        """Query LLM for triples; return raw string/JSON-like response (unchanged)."""
        named_entity_json = {"named_entities": entities}
        openie_messages = openie_post_ner_prompts.format_prompt(
            passage=text, named_entity_json=json.dumps(named_entity_json)
        )
        try:
            if isinstance(self.client, ChatOpenAI):  # JSON mode
                chat_completion = self.client.invoke(
                    openie_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_triples_tokens,
                    response_format={"type": "json_object"},
                )
                response_content = chat_completion.content

            elif isinstance(self.client, ChatOllama) or isinstance(self.client, ChatLlamaCpp):
                print(f'Message: {openie_messages.to_messages()}')
                response_content = self.client.invoke(openie_messages.to_messages()).content
                # try to get JSON-like dict
                response_content = extract_json_dict(response_content)
            else:  # no JSON mode
                chat_completion = self.client.invoke(
                    openie_messages.to_messages(),
                    temperature=0,
                    max_tokens=self.max_triples_tokens,
                )
                response_content = chat_completion.content
                response_content = extract_json_dict(response_content)

        except Exception as e:
            logger.error(f"Error in OpenIE: {e}")
            response_content = "{}"

        # Return as-is (string or dict); caller will parse safely
        return response_content

    def __call__(self, text: str) -> dict:
        res = {"passage": text, "extracted_entities": [], "extracted_triples": []}

        # 1) NER via LLM
        doc_entities = self.ner(text)
        # dedup / normalize (ensure plain strings)
        doc_entities = _dedup_list(doc_entities)

        if not doc_entities:
            logger.warning("No entities extracted by NER. Will attempt to derive entities from triples if present.")

        # 2) Extract triples (LLM)
        raw_triples_resp = self.openie_post_ner_extract(text, doc_entities)

        # 3) Parse raw_triples_resp into structured triples
        parsed_triples = []
        try:
            # response may be dict already or string
            if isinstance(raw_triples_resp, dict):
             parsed = raw_triples_resp
            else:
             parsed = extract_json_dict(str(raw_triples_resp))
        except Exception as e:
            logger.error(f"Failed to parse OpenIE response JSON: {e}")
            parsed = None

        # parsed may be None, dict, or other. Try to extract triples key or parse free text
        if isinstance(parsed, dict) and "triples" in parsed:
         parsed_triples = parsed.get("triples") or []
        else:
         # attempt to parse raw text with our flexible parser
         try:
             parsed_triples = parse_llm_triples(parsed or raw_triples_resp)
         except Exception as e:
             logger.error(f"Failed flexible parse of triples: {e}")
             parsed_triples = []

        # 4) If no entities but triples exist, derive entities from subjects/objects
        if (not doc_entities) and parsed_triples:
            derived = []
            for t in parsed_triples:
             try:
                 subj = _normalize_triple_item(t[0]) if isinstance(t, (list,tuple)) else _normalize_triple_item(t.get("subject") if isinstance(t, dict) else None)
                 obj = _normalize_triple_item(t[2]) if isinstance(t, (list,tuple)) else _normalize_triple_item(t.get("object") if isinstance(t, dict) else None)
             except Exception:
                 # fallback generic handling for dict-shaped triples
                 subj = _normalize_triple_item(t.get("subject")) if isinstance(t, dict) else ""
                 obj = _normalize_triple_item(t.get("object")) if isinstance(t, dict) else ""
             if subj: derived.append(subj)
             if obj: derived.append(obj)
            doc_entities = _dedup_list(derived)
            if doc_entities:
             logger.info("Derived entities from triples as NER fallback: %s", doc_entities)

        # 5) Finalize result structure
        res["extracted_entities"] = doc_entities
        # Normalize parsed_triples to list of [s,p,o]
        norm_triples = []
        for t in parsed_triples:
            if isinstance(t, dict):
                s = _normalize_triple_item(t.get("subject"))
                p = _normalize_triple_item(t.get("predicate"))
                o = _normalize_triple_item(t.get("object"))
                norm_triples.append([s,p,o])
            elif isinstance(t, (list,tuple)):
                parts = [_normalize_triple_item(x) for x in t]
                # pad to length 3
                while len(parts) < 3: parts.append("")
                norm_triples.append(parts[:3])
            elif isinstance(t, str):
                # try splitting heuristically
                parts = [p.strip() for p in re.split(r"\s*[-–—]\s*|\s+:\s+|\s*;\s*|\s+,\s+", t) if p.strip()]
                if len(parts) >= 3:
                    norm_triples.append(parts[:3])
        res["extracted_triples"] = norm_triples

        return res
