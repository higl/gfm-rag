import re
import json
import spacy
from typing import Dict, List
from vllm import LLM, SamplingParams
from .base_model import BaseOPENIEModel

SYSTEM_PROMPT = "Extract factual open IE triples. Use short, natural relation phrases."

USER_TEMPLATE = """TEXT:
{passage}

REQUIREMENTS:
- Output JSON with fields: entities, triples.
- entities: unique surface forms.
- triples: list of [head, relation, tail], using entities as head/tail.
- Use short relation phrases (e.g., 'founded', 'located in', 'won').
- Avoid coreference; replace pronouns with nearest named mention present in TEXT.
- No hallucinations; only assert what is explicit in TEXT.
JSON:
"""

class QwenOpenIE(BaseOPENIEModel):
    def __init__(self, max_ctx: int = 4096, batch_size: int = 4):
        self.max_ctx = max_ctx
        self.batch_size = batch_size
        
        # Load spaCy for NER
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize vLLM with Qwen2.5-3B-Instruct
        self.llm = LLM(
            model="Qwen/Qwen2.5-3B-Instruct-AWQ",
            tensor_parallel_size=1,
            max_model_len=max_ctx,
            gpu_memory_utilization=0.8
        )
        
        # Sampling parameters for deterministic JSON
        self.sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=1024,
            stop=["```", "\n\n\n"]
        )

    def __call__(self, passage: str) -> Dict:
        # Extract named entities with spaCy
        ents = set()
        doc = self.nlp(passage)
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT", "PRODUCT"):
                ents.add(self._normalize_text(ent.text))

        # Create prompt
        prompt = self._create_chat_prompt(passage)
        
        # Generate with vLLM
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                # Process entities
                entities = [self._normalize_text(e) for e in parsed.get("entities", []) if e]
                entities = list({e for e in entities if len(e) > 1})
                entities = list({*entities, *ents})  # Merge with spaCy entities
                
                # Process triples
                triples = []
                for triple in parsed.get("triples", []):
                    if isinstance(triple, list) and len(triple) == 3:
                        h, r, t = [self._normalize_text(x) for x in triple]
                        if h and r and t:
                            triples.append([h, self._clean_relation(r), t])
                
                return {
                    "passage": passage,
                    "extracted_entities": entities,
                    "extracted_triples": triples
                }
        except Exception as e:
            print(f"Error in QwenOpenIE: {e}")
        
        # Fallback to spaCy entities only
        return {
            "passage": passage,
            "extracted_entities": list(ents),
            "extracted_triples": []
        }

    def _create_chat_prompt(self, passage: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(passage=passage)}
        ]
        
        # Format for Qwen chat template
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.strip(' ,.;:()[]{}"\'-')
        return text

    def _clean_relation(self, relation: str) -> str:
        relation = self._normalize_text(relation).lower()
        # Remove common auxiliary verbs
        relation = re.sub(r'^(is|was|were|are|be|been)\s+', '', relation)
        return relation if relation else "related_to"
