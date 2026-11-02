import numpy as np
import pytest

from src.embeddings.embedding_service import EmbeddingDimensionError, EmbeddingService

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSION = 384


@pytest.fixture(scope="module")
def service():
    # Loaded once per test module: real model load takes a couple of
    # seconds, and every test here needs the same instance anyway.
    return EmbeddingService(model_name=MODEL_NAME, expected_dimension=DIMENSION)


def test_wrong_expected_dimension_raises():
    with pytest.raises(EmbeddingDimensionError):
        EmbeddingService(model_name=MODEL_NAME, expected_dimension=999)


def test_embed_documents_shape_and_dimension(service):
    result = service.embed_documents(["hello world", "another sentence"])
    assert result.vectors.shape == (2, DIMENSION)
    assert result.dimension == DIMENSION
    assert result.vectors.dtype == np.float32


def test_embed_documents_empty_list_returns_empty_array(service):
    result = service.embed_documents([])
    assert result.vectors.shape == (0, DIMENSION)


def test_embed_documents_vectors_are_unit_normalized(service):
    result = service.embed_documents(["some text to embed", "more text"])
    norms = np.linalg.norm(result.vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_query_is_unit_normalized(service):
    vector = service.embed_query("what is the capital of France?")
    assert vector.shape == (DIMENSION,)
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-5


def test_similar_texts_are_closer_than_unrelated_texts(service):
    query = service.embed_query("What are the health benefits of eating apples?")
    related_doc = service.embed_documents(
        ["Apples are a good source of fiber and vitamin C."]
    ).vectors[0]
    unrelated_doc = service.embed_documents(
        ["The stock market closed lower on Tuesday amid interest rate concerns."]
    ).vectors[0]

    sim_related = float(np.dot(query, related_doc))
    sim_unrelated = float(np.dot(query, unrelated_doc))

    assert sim_related > sim_unrelated
