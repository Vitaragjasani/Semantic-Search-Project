"""Persisted FAISS vector index over chunk embeddings.

Exact vs. approximate search: IndexFlatIP computes exact inner-product
similarity against every stored vector -- no approximation, so it always
returns the true top-K. That costs O(n) per query, which is fine up to a
few hundred thousand vectors on CPU. Approximate nearest-neighbor indexes
(IVF, HNSW, ...) trade a small, tunable chance of missing the true top-K
for much faster search at larger scale. This project starts with
IndexFlatIP because correctness and simplicity matter more than raw speed
at this document scale; swapping in an ANN index later is a drop-in change
behind the same add/search interface.

A FAISS index only ever returns an integer row position (0, 1, 2, ...).
That position is NOT a chunk ID -- it's just "the n-th vector added, in
insertion order." This module is the only place that reads that raw
position; everywhere else works through the persisted `_id_map`
translation (position -> chunk_id) returned by search().

Vectors are expected to already be L2-normalized (EmbeddingService does
this), so inner product here is equivalent to cosine similarity.
"""
import json
from pathlib import Path

import faiss
import numpy as np


class FaissStoreError(Exception):
    pass


class FaissStore:
    INDEX_FILENAME = "faiss_index.bin"
    ID_MAP_FILENAME = "faiss_id_map.json"

    def __init__(self, dimension: int, index_dir: Path):
        self.dimension = dimension
        self.index_dir = index_dir
        self._index_path = index_dir / self.INDEX_FILENAME
        self._id_map_path = index_dir / self.ID_MAP_FILENAME
        self._index = faiss.IndexFlatIP(dimension)
        self._id_map: list[str] = []  # row position -> chunk_id

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        """Append new vectors to the existing index. This is the path used
        when ingesting a newly uploaded document: only that document's
        chunks are embedded and added here -- previously indexed vectors
        are untouched and never recomputed.
        """
        if vectors.shape[0] != len(chunk_ids):
            raise FaissStoreError(
                f"Got {vectors.shape[0]} vectors but {len(chunk_ids)} chunk_ids."
            )
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.dimension:
            raise FaissStoreError(
                f"Vector dimension {vectors.shape[1]} does not match "
                f"index dimension {self.dimension}."
            )
        self._index.add(vectors.astype(np.float32))
        self._id_map.extend(chunk_ids)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise FaissStoreError("top_k must be positive")
        if query_vector.shape != (self.dimension,):
            raise FaissStoreError(
                f"Query vector has shape {query_vector.shape}, "
                f"expected ({self.dimension},)."
            )
        if self.size == 0:
            return []

        k = min(top_k, self.size)
        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, positions = self._index.search(query, k)

        results = []
        for score, position in zip(scores[0], positions[0]):
            if position == -1:
                continue
            results.append((self._id_map[position], float(score)))
        return results

    def rebuild(self, vectors: np.ndarray, chunk_ids: list[str]) -> None:
        """Replace the index from scratch. Used after document deletion:
        FAISS's in-place remove_ids on a flat index also shifts every
        later row's position, which would require re-deriving the whole
        id_map anyway. A full rebuild from the remaining chunks' already-
        computed vectors is just as correct and much simpler -- it costs
        re-inserting unchanged vectors, not recomputing embeddings.
        """
        self._index = faiss.IndexFlatIP(self.dimension)
        self._id_map = []
        self.add(vectors, chunk_ids)

    def get_vectors(self, chunk_ids: list[str]) -> np.ndarray:
        """Reads back already-stored vectors for the given chunk_ids,
        without recomputing any embeddings. IndexFlatIP keeps every added
        vector verbatim, so faiss's reconstruct() just returns it.

        This is what makes document deletion cheap: rebuild() needs the
        remaining chunks' vectors, and this fetches them from the index
        itself instead of re-embedding every surviving chunk.
        """
        if not chunk_ids:
            return np.empty((0, self.dimension), dtype=np.float32)
        position_by_chunk_id = {chunk_id: i for i, chunk_id in enumerate(self._id_map)}
        vectors = np.empty((len(chunk_ids), self.dimension), dtype=np.float32)
        for row, chunk_id in enumerate(chunk_ids):
            if chunk_id not in position_by_chunk_id:
                raise FaissStoreError(f"chunk_id '{chunk_id}' is not present in this index.")
            vectors[row] = self._index.reconstruct(position_by_chunk_id[chunk_id])
        return vectors

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        self._id_map_path.write_text(json.dumps(self._id_map), encoding="utf-8")

    def load(self) -> bool:
        """Loads a previously saved index if one exists. Returns False
        (leaving the store as a fresh empty index) when there's nothing to
        load yet, e.g. first run -- callers should not treat that as an
        error.
        """
        if not self._index_path.exists() or not self._id_map_path.exists():
            return False
        loaded_index = faiss.read_index(str(self._index_path))
        if loaded_index.d != self.dimension:
            raise FaissStoreError(
                f"Saved index dimension {loaded_index.d} does not match "
                f"configured dimension {self.dimension}."
            )
        self._index = loaded_index
        self._id_map = json.loads(self._id_map_path.read_text(encoding="utf-8"))
        return True
