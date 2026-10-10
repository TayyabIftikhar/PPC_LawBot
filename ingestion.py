import os
import re
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DATA_DIR = 'data'
DB_PATH = 'ppc_vector_db_ref'
MODEL_NAME = "./nomic-embed"
MODEL_REVISION = ""
LOG_DIR = "logs"
DEFAULT_SOURCE = "Pakistan Penal Code.pdf"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "lawgpt_ingest.log")

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

# ======================================
# Section Extraction
# ======================================
def extract_sections_from_text(text, page_number):
    """
    Extract PPC sections from text of a single page.
    Returns a list of Document objects, one per section, with page metadata.
    """
    section_pattern = r'(?:Section\s+)?(\d{1,3})(?:\s*[:.]\s*)?([^\n]*(?:\n(?!(?:Section\s+)?\d{1,3}[:.]).*)*)'
    sections = []
    matches = re.finditer(section_pattern, text, re.MULTILINE | re.IGNORECASE)

    for match in matches:
        section_num = match.group(1).strip()
        section_content = match.group(2).strip()
        if section_content:
            doc = Document(
                page_content=f"Section {section_num}\n{section_content}",
                metadata={
                    "section_number": int(section_num),
                    "page": page_number,
                    "source": DEFAULT_SOURCE
                }
            )
            sections.append(doc)
    return sections

def extract_sections_from_pdf(pdf_path):
    """
    Load a PDF and extract all sections from each page, preserving page number and PDF filename.
    """
    logger.info(f"Loading PDF: {pdf_path}")
    sections = []

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        for i, page in enumerate(pages, start=1):
            page_text = page.page_content
            page_number = page.metadata.get("page", i)  # fallback to i if not in metadata

            extracted = extract_sections_from_text(page_text, page_number)
            for doc in extracted:
                doc.metadata["pdf"] = os.path.basename(pdf_path) or DEFAULT_SOURCE
                sections.append(doc)

        logger.info(f"Extracted {len(sections)} sections from {pdf_path}")
        return sections
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {e}")
        return []

# ======================================
# Main Ingestion
# ======================================
def main():
    logger.info("--- Starting PPC Section-based Ingestion Process ---")

    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory '{DATA_DIR}' not found. Please create it and add your PPC PDF.")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in '{DATA_DIR}'. Exiting.")
        return

    logger.info(f"Found {len(pdf_files)} PDF file(s): {pdf_files}")

    all_sections = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        sections = extract_sections_from_pdf(pdf_path)
        all_sections.extend(sections)

    if not all_sections:
        logger.warning("No sections extracted. Please check your PDF format.")
        return

    logger.info(f"Total sections extracted: {len(all_sections)}")

    # Show sample sections in logs
    logger.info("=== Sample Sections ===")
    for i, section in enumerate(all_sections[:3]):
        logger.info(f"\nSection {i+1}:")
        logger.info(f"Metadata: {section.metadata}")
        logger.info(f"Content (first 200 chars): {section.page_content[:200]}...")

    # Initialize embeddings
    logger.info(f"Loading embedding model: {MODEL_NAME}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={
                "trust_remote_code": True,
                "revision": MODEL_REVISION,
                "device": "cpu"
            }
        )
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return

    # Create and save vector store
    logger.info(f"Creating FAISS index at '{DB_PATH}' with {len(all_sections)} sections...")
    try:
        faiss_db = FAISS.from_documents(all_sections, embeddings)
        faiss_db.save_local(DB_PATH)
        logger.info(f"✅ Ingestion complete! Vector database saved with {len(all_sections)} sections.")
        logger.info(f"Database path: {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to create/save vector store: {e}")

if __name__ == "__main__":
    main()
