
from pathlib import Path

import streamlit as st

from src.rag_chain import (
    UNKNOWN_ANSWER,
    answer_question,
    build_vector_store,
    load_local_llm,
)


PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_DIR / "data" / "documents"


st.set_page_config(
    page_title="DocuChat",
    page_icon="📚",
    layout="centered",
)

st.title("📚 DocuChat")
st.write("Ask questions about your documents and get answers grounded in their content.")

with st.sidebar:
    st.header("About DocuChat")
    st.write("Your documents stay on your local machine during processing.")
    st.write("Supported document types: TXT and PDF.")
    st.caption("Answers depend on the content and quality of your documents.")

if not DOCUMENTS_DIR.exists():
    st.error("The documents directory was not found.")
    st.stop()

@st.cache_resource(show_spinner="Loading documents and building the search index...")
def get_vector_store():
    return build_vector_store(DOCUMENTS_DIR)

@st.cache_resource(show_spinner="Loading the local language model...")
def get_llm():
    return load_local_llm()

question = st.text_input(
    "Enter your question",
    placeholder="How long is the return period?",
)

if st.button("Ask DocuChat", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner("Searching documents and generating an answer..."):
                vector_store = get_vector_store()
                llm = get_llm()
                result = answer_question(question, vector_store, llm)

            st.subheader("Answer")
            st.write(result["answer"])

            if result["answer"] == UNKNOWN_ANSWER:
                st.info("The available documents did not provide a supported answer.")

            st.subheader("Sources")
            if result["sources"]:
                for source in result["sources"]:
                    st.write(f"- {source}")
            else:
                st.write("No sources found.")

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
