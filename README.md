# 📚 LangChain DocuChat — Local Document Q&A with RAG

A document question-answering application built with **LangChain, Retrieval-Augmented Generation (RAG), FAISS, Hugging Face, and Streamlit**. DocuChat lets users ask questions about PDF and TXT documents and receive answers based on retrieved document content using a locally running Qwen language model.

## 🚀 Project Overview

Large language models can generate answers that sound convincing but may not be supported by your documents. DocuChat addresses this challenge by retrieving relevant document sections before generating an answer.

The application uses a local language model for inference and includes a fallback response for unsupported product questions.

### Key Features

- 📄 **Document loading:** Supports PDF and TXT files.
- ✂️ **Text chunking:** Splits documents into smaller, overlapping chunks.
- 🧠 **Text embeddings:** Uses Sentence Transformers to represent text as numerical vectors.
- 🔎 **Semantic retrieval:** Searches for relevant document chunks using FAISS.
- 🤖 **Local LLM:** Uses Qwen2.5-0.5B-Instruct through Hugging Face Transformers.
- 💬 **Question answering:** Combines retrieved context with a prompt to generate answers.
- 📚 **Source attribution:** Displays the filenames associated with retrieved content.
- 🖥️ **Web interface:** Provides an interactive Streamlit application.
- 🧪 **Automated tests:** Includes tests for document loading, splitting, retrieval, and answer behavior.
- 🛡️ **Unknown-answer fallback:** Returns a predefined response for unsupported product questions.

## 🏗️ Architecture

```text
           PDF / TXT Documents
                    |
                    v
             Document Loader
                    |
                    v
             Text Splitter
                    |
                    v
        Hugging Face Embeddings
                    |
                    v
             FAISS Vector Store
                    |
                    v
            User's Question
                    |
                    v
          Semantic Retrieval
                    |
                    v
          Retrieved Text Chunks
                    |
                    v
       Prompt + Retrieved Context
                    |
                    v
          Local Qwen Language Model
                    |
                    v
           Answer + Source Names
                    |
                    v
            Streamlit Interface
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| LangChain | Document processing and LLM integration |
| Hugging Face Transformers | Loading and running the local language model |
| Sentence Transformers | Generating text embeddings |
| FAISS | Vector similarity search |
| Qwen2.5-0.5B-Instruct | Local text generation |
| Streamlit | Interactive web interface |
| PyPDF | Reading PDF documents |
| Pytest | Automated testing |

## 📁 Project Structure

```text
langchain-docuchat/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── documents/
│       └── company_policy.txt
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   └── rag_chain.py
└── tests/
    ├── test_document_loader.py
    └── test_rag_chain.py
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or newer
- Git
- Internet access for the initial model downloads
- Sufficient RAM and disk space for the embedding and language models

The application is configured to run model inference on the CPU. Performance depends on your hardware.

### 1. Clone the Repository

```bash
git clone https://github.com/YasirKhan2929/langchain-docuchat.git
cd langchain-docuchat
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the virtual environment's Python executable directly without activating it.

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

If you did not activate the virtual environment, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Run the Application

```powershell
python -m streamlit run app.py
```

Or, without activating the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that address in your browser.

**Note:** The first run may take longer because the embedding model and language model need to download and load.

## 📄 Adding Your Own Documents

Place supported documents in:

```text
data/documents/
```

Supported extensions:

- `.txt`
- `.pdf`

The current application loads documents from this directory when it builds its vector store. Restarting the application or clearing its cached resources may be necessary after changing documents.

## 🧪 Running Tests

Run the automated test suite:

```powershell
python -m pytest -v
```

If you are using the virtual environment without activation:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

**Current test status:** 12 tests passed in the development environment.

The tests cover document loading, text splitting, empty-question validation, retrieval behavior, supported-answer behavior, and the unsupported-product fallback.

## 💡 Example Questions

Using the included TechStore company policy document, try:

| Question | Expected behavior |
|---|---|
| How long do customers have to return a product? | Retrieves the seven-day return policy |
| What payment methods does TechStore accept? | Retrieves the documented payment methods |
| How long does standard delivery take? | Retrieves the standard delivery timeframe |
| Does TechStore sell bicycles? | Returns the configured unknown-answer response |

Actual generated answers depend on retrieval results and model behavior.

## 🔍 How RAG Works in DocuChat

**Retrieval-Augmented Generation (RAG)** combines information retrieval with language generation.

1. **Load:** Read supported documents from the documents directory.
2. **Split:** Divide document text into smaller chunks.
3. **Embed:** Convert chunks into vector representations.
4. **Index:** Store vectors in a FAISS index for similarity search.
5. **Retrieve:** Find chunks relevant to a user's question.
6. **Generate:** Provide the retrieved context and question to the local Qwen model.
7. **Display:** Show the generated answer and associated source filenames.

This approach helps ground answers in supplied documents, but it does not guarantee factual correctness. The unsupported-product check is a targeted safeguard, not a general guarantee that every unsupported question will be rejected.

## 🔐 Privacy and Limitations

- Document processing and model inference are designed to run locally.
- Initial model downloads require an internet connection.
- No external hosted LLM API is required by the current implementation.
- The application does not currently provide user authentication or access controls.
- Source filenames identify retrieved documents but do not guarantee that every statement in an answer is supported.
- The unknown-answer behavior is not a complete hallucination-prevention system.
- Large documents and local model inference may require additional memory and processing time.

Avoid placing confidential documents in the application unless you have reviewed the security and data-handling requirements of your environment.

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

- Building a basic RAG pipeline
- Loading and splitting PDF and TXT documents
- Generating text embeddings
- Vector indexing and semantic similarity search
- Integrating a local Hugging Face language model
- Creating a question-answering interface with Streamlit
- Writing automated tests with Pytest
- Organizing a Python project for GitHub

## 🔮 Future Improvements

- Add conversation history and memory
- Improve answer-grounding and citation accuracy
- Add configurable retrieval thresholds and chunk sizes
- Support DOCX and Markdown documents
- Add document upload through the web interface
- Improve error handling and model-loading feedback
- Add evaluation datasets and retrieval-quality metrics
- Explore more advanced local models and optimized inference

## 👨‍💻 Author

**Yasir Khan**

GitHub: [@YasirKhan2929](https://github.com/YasirKhan2929)

---

*Built as a hands-on project to learn LangChain, RAG, vector databases, local language models, and practical GenAI application development.*