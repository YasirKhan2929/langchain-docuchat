
import re
from pathlib import Path

from transformers import pipeline
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from src.document_loader import load_documents, split_documents


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

UNKNOWN_ANSWER = "I don't know based on the available information."

# Product names explicitly listed in company_policy.txt.
KNOWN_PRODUCTS = {
    "laptops",
    "smartphones",
    "keyboards",
    "mice",
    "computer accessories",
}

PRODUCT_QUESTION_PATTERNS = (
    r"\bdoes\s+techstore\s+sell\b",
    r"\bdoes\s+techstore\s+stock\b",
    r"\bdo\s+you\s+sell\b",
    r"\bdo\s+you\s+stock\b",
    r"\bis\s+.+\s+available\b",
)


def build_vector_store(
    documents_directory: str | Path = "data/documents",
) -> FAISS:
    """Load documents, split them, and create a local FAISS index."""
    documents = load_documents(documents_directory)
    chunks = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.from_documents(chunks, embeddings)


def retrieve_documents(
    vector_store: FAISS,
    question: str,
    k: int = 3,
) -> list[Document]:
    """Retrieve the most relevant chunks for a question."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    return vector_store.similarity_search(question, k=k)


def load_local_llm() -> HuggingFacePipeline:
    """Load Qwen locally using CPU inference."""
    text_generation_pipeline = pipeline(
        task="text-generation",
        model=LLM_MODEL,
        tokenizer=LLM_MODEL,
        device=-1,
        max_new_tokens=80,
        do_sample=False,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=text_generation_pipeline)


def _get_sources(documents: list[Document]) -> list[str]:
    """Return unique source filenames in sorted order."""
    return sorted({
        document.metadata.get("source", "Unknown")
        for document in documents
    })


def _is_product_question(question: str) -> bool:
    """Identify questions that ask whether a product is sold or available."""
    normalized = question.lower().strip()
    return any(
        re.search(pattern, normalized)
        for pattern in PRODUCT_QUESTION_PATTERNS
    )


def _product_is_documented(question: str, context: str) -> bool:
    """Check whether a named product is explicitly documented."""
    normalized_question = question.lower()
    normalized_context = context.lower()

    # Only treat a product as documented if its name occurs in the
    # retrieved context, not merely because it is a common product.
    for product in KNOWN_PRODUCTS:
        if product in normalized_question:
            return product in normalized_context

    return False


def answer_question(
    question: str,
    vector_store: FAISS,
    llm: HuggingFacePipeline,
    k: int = 2,
) -> dict:
    """Answer from retrieved context, abstaining on unsupported products."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    documents = retrieve_documents(vector_store, question, k=k)
    context = "\n\n".join(
        document.page_content for document in documents
    )
    sources = _get_sources(documents)

    # Do not infer that an unlisted product is unavailable.
    if _is_product_question(question):
        if not _product_is_documented(question, context):
            return {
                "question": question,
                "answer": UNKNOWN_ANSWER,
                "sources": sources,
            }

    prompt = f"""<|im_start|>system
You are DocuChat, a document question-answering assistant.
Answer using only facts explicitly stated in the context.
If the answer is not explicitly supported by the context, reply exactly:
{UNKNOWN_ANSWER}
Do not guess, infer missing facts, or add advice.
Keep your answer concise, preferably one sentence.
<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""

    answer = llm.invoke(prompt).strip()

    # Stop accidental continuation if the model emits another section.
    for marker in (
        "<|im_end|>",
        "<|im_start|>",
        "\nContext:",
        "\nQuestion:",
    ):
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()

    # If the model explicitly abstains, preserve that response.
    if UNKNOWN_ANSWER.lower() in answer.lower():
        answer = UNKNOWN_ANSWER

    # Basic overlap guard only. This is not a factuality guarantee.
    context_words = {
        word.lower()
        for word in re.findall(r"\b[\w'-]+\b", context)
        if len(word) > 2
    }
    answer_words = {
        word.lower()
        for word in re.findall(r"\b[\w'-]+\b", answer)
        if len(word) > 2
    }

    if not answer or not (answer_words & context_words):
        answer = UNKNOWN_ANSWER

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
