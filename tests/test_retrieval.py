import numpy as np
import pytest

from src.retrieval.faiss_store import FaissStore, FaissStoreError

DIM = 8


def _unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def test_search_on_empty_index_returns_empty_list(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    results = store.search(_unit_vector(1), top_k=5)
    assert results == []


def test_add_and_search_returns_expected_chunk_ids(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    vectors = np.stack([_unit_vector(i) for i in range(5)])
    chunk_ids = [f"chunk_{i}" for i in range(5)]
    store.add(vectors, chunk_ids)

    # Querying with the exact vector for chunk_2 should return chunk_2 first.
    results = store.search(vectors[2], top_k=3)
    assert results[0][0] == "chunk_2"
    assert results[0][1] > 0.99  # near-perfect self-similarity


def test_top_k_larger_than_index_size_is_clamped(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    vectors = np.stack([_unit_vector(i) for i in range(3)])
    store.add(vectors, ["a", "b", "c"])

    results = store.search(_unit_vector(0), top_k=100)
    assert len(results) == 3


def test_incremental_add_preserves_earlier_entries(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    store.add(np.stack([_unit_vector(0), _unit_vector(1)]), ["a", "b"])
    store.add(np.stack([_unit_vector(2)]), ["c"])

    assert store.size == 3
    results = store.search(_unit_vector(1), top_k=1)
    assert results[0][0] == "b"


def test_add_rejects_mismatched_vector_and_id_counts(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    with pytest.raises(FaissStoreError):
        store.add(np.stack([_unit_vector(0), _unit_vector(1)]), ["only_one"])


def test_add_rejects_wrong_dimension(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    wrong_dim_vectors = np.zeros((1, DIM + 1), dtype=np.float32)
    with pytest.raises(FaissStoreError):
        store.add(wrong_dim_vectors, ["a"])


def test_search_rejects_wrong_query_dimension(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    store.add(np.stack([_unit_vector(0)]), ["a"])
    with pytest.raises(FaissStoreError):
        store.search(np.zeros(DIM + 1, dtype=np.float32), top_k=1)


def test_search_rejects_nonpositive_top_k(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    store.add(np.stack([_unit_vector(0)]), ["a"])
    with pytest.raises(FaissStoreError):
        store.search(_unit_vector(0), top_k=0)


def test_save_and_load_round_trip(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    vectors = np.stack([_unit_vector(i) for i in range(4)])
    chunk_ids = [f"chunk_{i}" for i in range(4)]
    store.add(vectors, chunk_ids)
    store.save()

    reloaded = FaissStore(dimension=DIM, index_dir=tmp_path)
    found = reloaded.load()

    assert found is True
    assert reloaded.size == 4
    results = reloaded.search(vectors[3], top_k=1)
    assert results[0][0] == "chunk_3"


def test_load_with_no_saved_index_returns_false(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    assert store.load() is False
    assert store.size == 0


def test_load_rejects_dimension_mismatch(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    store.add(np.stack([_unit_vector(0)]), ["a"])
    store.save()

    mismatched = FaissStore(dimension=DIM + 1, index_dir=tmp_path)
    with pytest.raises(FaissStoreError):
        mismatched.load()


def test_rebuild_replaces_index_contents(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    store.add(np.stack([_unit_vector(i) for i in range(3)]), ["a", "b", "c"])

    # Simulate deleting document "b": rebuild from the remaining chunks only.
    remaining_vectors = np.stack([_unit_vector(0), _unit_vector(2)])
    store.rebuild(remaining_vectors, ["a", "c"])

    assert store.size == 2
    results = store.search(_unit_vector(1), top_k=3)
    returned_ids = {chunk_id for chunk_id, _ in results}
    assert "b" not in returned_ids


def test_get_vectors_reconstructs_without_recomputation(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    vectors = np.stack([_unit_vector(i) for i in range(3)])
    store.add(vectors, ["a", "b", "c"])

    reconstructed = store.get_vectors(["c", "a"])
    np.testing.assert_allclose(reconstructed[0], vectors[2], atol=1e-6)
    np.testing.assert_allclose(reconstructed[1], vectors[0], atol=1e-6)


def test_get_vectors_empty_list_returns_empty_array(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    assert store.get_vectors([]).shape == (0, DIM)


def test_get_vectors_rejects_unknown_chunk_id(tmp_path):
    store = FaissStore(dimension=DIM, index_dir=tmp_path)
    store.add(np.stack([_unit_vector(0)]), ["a"])
    with pytest.raises(FaissStoreError):
        store.get_vectors(["not_in_index"])
