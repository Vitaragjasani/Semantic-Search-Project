"""Turn text into dense vectors with a local sentence-transformer model.

What an embedding is: a fixed-length array of floats that a model derives
from text such that texts with similar meaning land close together in that
vector space. This differs from a generative LLM: a generative model
predicts the next token to produce more text, while an embedding model
collapses an entire input into one fixed-size vector and is never used to
generate text. The two have different training objectives, so you cannot
substitute one for the other.

Query and document embeddings must come from the same model because each
embedding model defines its own vector space -- a query vector from model A
compared against document vectors from model B is comparing two unrelated
coordinate systems, and the resulting distances are meaningless even though
the arithmetic succeeds silently. This module's embed_documents and
embed_query always go through the one model instance loaded in __init__.

Cosine similarity measures the angle between two vectors, ignoring their
length -- a sentence repeated twice would otherwise appear to have a
"bigger" vector without being more relevant. If every vector is first
L2-normalized to unit length (as this module does), inner product becomes
numerically identical to cosine similarity. That's the whole reason Phase 4
builds the FAISS index as IndexFlatIP (inner product) instead of needing a
separate cosine-distance index type.
"""
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

# BGE models are trained with this instruction prefix on the query side only
# -- not on documents/passages -- and it measurably improves retrieval for
# this model family. Other sentence-transformer families (e.g. MiniLM) have
# no such convention, so no prefix is added for them.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingDimensionError(Exception):
    pass


@dataclass
class EmbeddingResult:
    vectors: np.ndarray  # shape (n, dim), L2-normalized, dtype float32
    dimension: int


class EmbeddingService:
    def __init__(self, model_name: str, expected_dimension: int, batch_size: int = 32):
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

        actual_dimension = self._model.get_sentence_embedding_dimension()
        if actual_dimension != expected_dimension:
            raise EmbeddingDimensionError(
                f"Model '{model_name}' produces {actual_dimension}-dim vectors, "
                f"but EMBEDDING_DIMENSION is configured as {expected_dimension}. "
                "Update EMBEDDING_DIMENSION to match the model."
            )

    def _uses_bge_query_instruction(self) -> bool:
        return "bge" in self.model_name.lower()

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                vectors=np.empty((0, self.expected_dimension), dtype=np.float32),
                dimension=self.expected_dimension,
            )
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return EmbeddingResult(vectors=vectors, dimension=vectors.shape[1])

    def embed_query(self, text: str) -> np.ndarray:
        query_text = _BGE_QUERY_INSTRUCTION + text if self._uses_bge_query_instruction() else text
        vector = self._model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]
        return vector
