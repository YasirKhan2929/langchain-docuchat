
from pathlib import Path

import pytest

from src.document_loader import load_documents, split_documents


def test_load_text_document(tmp_path: Path):
    document_path = tmp_path / "policy.txt"
    document_path.write_text(
        "Customers can return products within 7 days.",
        encoding="utf-8",
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert "7 days" in documents[0].page_content
    assert documents[0].metadata["source"] == "policy.txt"


def test_split_documents_preserves_source():
    from langchain_core.documents import Document

    document = Document(
        page_content="This is a sample document. " * 100,
        metadata={"source": "sample.txt"},
    )

    chunks = split_documents([document], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "sample.txt" for chunk in chunks)


def test_missing_directory_raises_error(tmp_path: Path):
    missing_directory = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        load_documents(missing_directory)


def test_empty_directory_raises_error(tmp_path: Path):
    with pytest.raises(ValueError, match="No TXT or PDF documents found"):
        load_documents(tmp_path)


def test_invalid_chunk_size_raises_error():
    from langchain_core.documents import Document

    document = Document(page_content="Sample text.")

    with pytest.raises(ValueError, match="chunk_size"):
        split_documents([document], chunk_size=0)


def test_invalid_chunk_overlap_raises_error():
    from langchain_core.documents import Document

    document = Document(page_content="Sample text.")

    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents([document], chunk_size=100, chunk_overlap=100)
