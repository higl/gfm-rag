import hashlib
import os
import pickle
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from gfmrag.kg_construction.utils import processing_phrases

from .base_model import BaseELModel


class SimpleELModel(BaseELModel):
    """Simple Entity Linking Model using SentenceTransformers.

    This class implements a simple entity linking model using SentenceTransformers
    for encoding entities and computing cosine similarity for entity matching.
    """

    def __init__(
        self,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        root: str = "tmp",
        force: bool = False,
        **kwargs: str,
    ) -> None:
        """Initialize the Simple entity linking model.

        Args:
            model_name_or_path (str): Name or path of the sentence transformer model
            root (str): Root directory for storing indices
            force (bool): Whether to force recomputation of existing indices
        """
        self.model_name_or_path = model_name_or_path
        self.root = root
        self.force = force
        
        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name_or_path)
        
        # Placeholders for indexed data
        self.entity_list = None
        self.entity_embeddings = None
        self.index_path = None

    def index(self, entity_list: List[str]) -> None:
        """Index a list of entities using SentenceTransformers.

        Args:
            entity_list (list): List of entity strings to be indexed
        """
        # Create cache directory
        os.makedirs(self.root, exist_ok=True)
        
        # Get md5 fingerprint of the entity list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        cache_file = os.path.join(self.root, f"simple_el_index_{fingerprint}.pkl")
        
        if os.path.exists(cache_file) and not self.force:
            # Load cached embeddings
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.entity_list = cached_data['entities']
                self.entity_embeddings = cached_data['embeddings']
                self.index_path = cache_file
        else:
            # Process entities and compute embeddings
            processed_entities = [processing_phrases(entity) for entity in entity_list]
            self.entity_list = processed_entities
            
            # Compute embeddings
            self.entity_embeddings = self.model.encode(processed_entities)
            
            # Cache the results
            cache_data = {
                'entities': self.entity_list,
                'embeddings': self.entity_embeddings
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.index_path = cache_file

    def __call__(self, queries: List[str], topk: int = 5) -> Dict[str, List[dict]]:
        """
        Compute top-k entity neighbors for each query phrase.
        Returns a dict: { query_string: [{"entity": entity_idx}, ...], ... }
        Robust to empty inputs and mismatched embedding shapes.
        """
        # If no queries provided, return empty dict
        if not queries:
            return {}

        # Compute query embeddings (expect list or numpy array)
        try:
            query_embeddings = self.model.encode(queries)  # existing embed method
        except Exception:
            query_embeddings = None

        # Ensure entity embeddings exist on model (may be loaded elsewhere)
        entity_embeddings = getattr(self, "entity_embeddings", None)

        # Normalize to numpy arrays
        query_embeddings = np.array(query_embeddings) if query_embeddings is not None else np.array([])
        entity_embeddings = np.array(entity_embeddings) if entity_embeddings is not None else np.array([])

        # If either is empty, return empty neighbor lists for each query
        if query_embeddings.size == 0 or entity_embeddings.size == 0:
            return {q: [] for q in queries}

        # Ensure 2D arrays (n_samples, n_features)
        if query_embeddings.ndim == 1:
            # single vector per all queries? reshape conservatively to (n_queries, dim) if length matches dim
            # if length equals entity dim then assume single query vector
            if entity_embeddings.ndim >= 2 and query_embeddings.shape[0] == entity_embeddings.shape[1]:
                query_embeddings = query_embeddings.reshape(1, -1)
            else:
                # try to interpret as multiple 1-d features -> reshape to (-1, 1)
                query_embeddings = query_embeddings.reshape(-1, 1)
        if entity_embeddings.ndim == 1:
            entity_embeddings = entity_embeddings.reshape(1, -1)

        # final check: dims must match
        if query_embeddings.shape[1] != entity_embeddings.shape[1]:
            # mismatch -> cannot compare, return empty lists
            return {q: [] for q in queries}

        # compute similarities
        try:
            similarities = cosine_similarity(query_embeddings, entity_embeddings)
        except Exception:
            return {q: [] for q in queries}

        out: Dict[str, List[dict]] = {}
        for i, q in enumerate(queries):
            if i < similarities.shape[0]:
                row = similarities[i]
                if row.size == 0:
                    out[q] = []
                    continue
                ids = np.argsort(row)[::-1][:topk]
                # Wrap each ID in a dict with "entity" key as expected by downstream code
                out[q] = [{"entity": int(idx)} for idx in ids]
            else:
                # If fewer computed rows than queries, give empty
                out[q] = []
        return out
