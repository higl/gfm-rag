# import numpy as np
# import faiss
# from typing import Dict, List
# from sentence_transformers import SentenceTransformer
# from .base_model import BaseELModel

# class SBertEL(BaseELModel):
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)
#         self.index = None
#         self.id2text = []
#         self.vecs = None

#     def index(self, phrases: List[str]) -> None:
#         """Build FAISS index for the given phrases."""
#         self.id2text = phrases
        
#         # Encode phrases
#         vectors = self.model.encode(
#             phrases, 
#             normalize_embeddings=True, 
#             batch_size=256, 
#             show_progress_bar=False,
#             convert_to_numpy=True
#         )
        
#         # Build FAISS index
#         dimension = vectors.shape[1]
#         self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
#         self.index.add(vectors.astype(np.float32))
#         self.vecs = vectors

#     def __call__(self, phrases: List[str], topk: int = 100) -> Dict:
#         """Find similar entities for each phrase."""
#         if self.index is None:
#             return {phrase: [] for phrase in phrases}
        
#         # Encode query phrases
#         query_vectors = self.model.encode(
#             phrases, 
#             normalize_embeddings=True, 
#             batch_size=256, 
#             show_progress_bar=False,
#             convert_to_numpy=True
#         ).astype(np.float32)
        
#         # Search for similar entities
#         similarities, indices = self.index.search(query_vectors, topk + 1)  # +1 to include self
        
#         result = {}
#         for i, phrase in enumerate(phrases):
#             neighbors = []
#             for j, score in zip(indices[i], similarities[i]):
#                 if j < 0 or j >= len(self.id2text):
#                     continue
                
#                 candidate = self.id2text[j]
#                 if candidate == phrase:  # Skip self
#                     continue
                
#                 neighbors.append({
#                     "entity": candidate,
#                     "norm_score": float(score)
#                 })
            
#             result[phrase] = neighbors
        
#         return result
