
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from src.rag_chain import (
    UNKNOWN_ANSWER,
    answer_question,
    retrieve_documents,
)


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content=(
                "Customers can return eligible products within 7 days "
                "of delivery. TechStore sells laptops and smartphones."
            ),
            metadata={"source": "company_policy.txt"},
        )
    ]


@pytest.fixture
def fake_vector_store(sample_documents):
    store = Mock()
    store.similarity_search.return_value = sample_documents
    return store


@pytest.fixture
def fake_llm():
    llm = Mock()
    llm.invoke.return_value = (
        "Customers can return eligible products within 7 days of delivery."
    )
    return llm


def test_retrieve_documents_returns_matching_chunks(fake_vector_store):
    results = retrieve_documents(
        fake_vector_store,
        "What is the return policy?",
    )

    assert len(results) == 1
    fake_vector_store.similarity_search.assert_called_once()


def test_empty_question_is_rejected(fake_vector_store, fake_llm):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question("   ", fake_vector_store, fake_llm)


def test_supported_question_returns_answer(fake_vector_store, fake_llm):
    result = answer_question(
        "How long do customers have to return a product?",
        fake_vector_store,
        fake_llm,
    )

    assert "7 days" in result["answer"]
    assert result["sources"] == ["company_policy.txt"]


def test_unsupported_product_returns_unknown(fake_vector_store, fake_llm):
    result = answer_question(
        "Does TechStore sell bicycles?",
        fake_vector_store,
        fake_llm,
    )

    assert result["answer"] == UNKNOWN_ANSWER
    fake_llm.invoke.assert_not_called()


def test_documented_product_question_uses_llm(fake_vector_store, fake_llm):
    result = answer_question(
        "Does TechStore sell laptops?",
        fake_vector_store,
        fake_llm,
    )

    assert result["answer"] != UNKNOWN_ANSWER
    fake_llm.invoke.assert_called_once()


def test_invalid_retrieval_count_is_rejected(fake_vector_store):
    with pytest.raises(ValueError, match="at least 1"):
        retrieve_documents(
            fake_vector_store,
            "What is the return policy?",
            k=0,
        )
