import os
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DATA_DIR = 'data'
DB_PATH = 'ppc_vector_db_final'
MODEL_NAME = "./nomic-embed"
MODEL_REVISION = ""
LOG_FILE = "logs/lawgpt_ingest.log"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting Ingestion Process ---")

    # 1. Validate Data Directory
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory '{DATA_DIR}' not found. Please create it and add your PDFs.")
        return

    # 2. Load Documents
    logger.info(f"Scanning '{DATA_DIR}' for PDF files...")
    try:
        loader = DirectoryLoader(
            DATA_DIR, 
            glob="./*.pdf",  # Changed to *.pdf to catch any PDF
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document pages.")
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        return

    if not documents:
        logger.warning("No documents found. Exiting.")
        return

    # 3. Split Text
    logger.info("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""] # Optimized for legal text
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"Sample chunk metadata: {texts[0].metadata}")
    logger.info(f"Created {len(texts)} text chunks.")

    # 4. Initialize Embeddings
    logger.info(f"Loading embedding model: {MODEL_NAME}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={
                "trust_remote_code": True, 
                "revision": MODEL_REVISION
            }
        )
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return

    # 5. Create and Save Vector Store
    logger.info(f"Creating FAISS index at '{DB_PATH}'...")
    try:
        faiss_db = FAISS.from_documents(texts, embeddings)
        faiss_db.save_local(DB_PATH)
        logger.info("Ingestion complete. Vector database saved successfully.")
    except Exception as e:
        logger.error(f"Failed to create/save vector store: {e}")

if __name__ == "__main__":
    main()