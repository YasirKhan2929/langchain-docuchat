
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def load_documents(directory: str | Path) -> list[Document]:
    """Load all supported TXT and PDF files from a directory."""
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    files = sorted(
        file
        for file in directory.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    documents = []

    for file_path in files:
        if file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            loader = PyPDFLoader(str(file_path))

        loaded_documents = loader.load()

        for document in loaded_documents:
            document.metadata["source"] = file_path.name

        documents.extend(loaded_documents)

    if not documents:
        raise ValueError(
            f"No TXT or PDF documents found in {directory}"
        )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Document]:
    """Split documents into smaller chunks for retrieval."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    return splitter.split_documents(documents)
