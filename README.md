# ⚖️ LawBot: Pakistan Penal Code Assistant

LawBot is a local, privacy-focused RAG chatbot designed to answer legal questions from the Pakistan Penal Code. It uses a local DeepSeek model for reasoning and FAISS for document retrieval. All processing runs offline unless you enable the optional Hybrid mode for web search through Perplexity.

---

## 📂 Project Structure

```
data/                     Place your PDF documents here
local_deepseek_model/     Stores the downloaded LLM weights
ppc_vector_db_ref/        Stores the FAISS vector database
app.py                 Main Streamlit application
ingestion.py                 Creates the vector database from PDFs
local_deepseek_model.py   Downloads the model from HuggingFace
```

---

## 🚀 Setup and Installation

Follow these steps to set up the project.

### 1. Install Dependencies

Ensure Python is installed. Using a virtual environment is recommended.

```bash
pip install -r requirements.txt
```

### 2. Download the Model For using HuggingFacePipeline

This downloads the DeepSeek model into the local_deepseek_model directory and Embedding model in nomic-embed directory.

```bash
python local_deepseek_model.py
```
```bash
huggingface-cli download nomic-ai/nomic-embed-text-v1 --local-dir ./nomic-embed
```

### 3. Ingest Documents

Place your PDF files inside the `data/` directory, then run:
For reference related ingestion:
```bash
python ingestion.py
```

Confirm from the logs that the database was created in `ppc_vector_db_ref`.

### 4. Run the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

---

## ⚙️ Configuration

### Switching Modes

The application provides two modes, configurable in the sidebar.

* 📚 Local Database: Uses local documents and the offline DeepSeek model.
* 🌐 Web Search: Uses the Perplexity MCP server when an API key is provided.

### Perplexity API Key (Optional)

To enable web search, open `app.py` and update the following variable:

```
PERPLEXITY_API_KEY = "pplx-xxxxxxxx..."
```

---

## 📜 License

This project is intended for educational and research use.

---
