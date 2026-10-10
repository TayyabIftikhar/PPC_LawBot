import os
import re
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# --- Configuration ---
DATA_DIR = 'data'
DB_PATH = 'ppc_vector_db_final'
# MODEL_NAME = "./nomic-embed"
MODEL_NAME = "nomic-embed-text:latest"
MODEL_REVISION = ""
LOG_FILE = "logs/lawbot_ingest.log"

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

# def extract_sections_from_text(text):
#     """
#     Extract PPC sections from text.
#     Looks for patterns like "Section 302", "Section 302.", "302.", etc.
#     Returns a list of Document objects, one per section.
#     """
#     # Pattern to match section headers: "Section XXX" or "XXX." or variations
#     # This matches: Section 302, Section 302:, Section 302., or just 302 followed by text
#     section_pattern = r'(?:Section\s+)?(\d{1,3})(?:\s*[:.]\s*)?([^\n]*(?:\n(?!(?:Section\s+)?\d{1,3}[:.]).*)*)'
    
#     sections = []
#     matches = re.finditer(section_pattern, text, re.MULTILINE | re.IGNORECASE)
    
#     for match in matches:
#         section_num = match.group(1).strip()
#         section_content = match.group(2).strip()
        
#         if section_content:  # Only include if there's content
#             # Create a Document with metadata
#             doc = Document(
#                 page_content=f"Section {section_num}\n{section_content}",
#                 metadata={
#                     "section_number": int(section_num),
#                     "source": "Pakistan Penal Code",
#                     "type": "section"
#                 }
#             )
#             sections.append(doc)
    
#     return sections

def extract_sections_from_text(text, page_number):
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
                    "source": "Pakistan Penal Code",
                    "page": page_number,
                    "type": "section"
                }
            )
            sections.append(doc)
    
    return sections


# def extract_sections_from_pdf(pdf_path):
#     """
#     Load a PDF and extract all sections.
#     """
#     logger.info(f"Loading PDF: {pdf_path}")
#     try:
#         loader = PyPDFLoader(pdf_path)
#         pages = loader.load()
        
#         # Combine all pages into one text
#         # full_text = "\n\n".join([page.page_content for page in pages])
        
#         # Extract sections
#         # sections = extract_sections_from_text(full_text)
        
#         sections = []
#         for i, page in enumerate(pages):
#             page_number = i + 1
#             page_sections = extract_sections_from_text(page.page_content, page_number)
#             sections.extend(page_sections)
        
#         logger.info(f"Extracted {len(sections)} sections from {pdf_path}")
        
#         return sections
#     except Exception as e:
#         logger.error(f"Failed to process PDF {pdf_path}: {e}")
#         return []

def extract_sections_from_pdf(pdf_path):
    logger.info(f"Loading PDF: {pdf_path}")
    sections = []

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        for page in pages:
            page_text = page.page_content
            page_number = page.metadata.get("page", "Unknown")

            extracted = extract_sections_from_text(page_text)

            for doc in extracted:
                doc.metadata.update({
                    "page": page_number,
                    "pdf": os.path.basename(pdf_path),
                    "source": "Pakistan Penal Code"
                })
                sections.append(doc)

        logger.info(f"Extracted {len(sections)} sections from {pdf_path}")
        return sections

    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {e}")
        return []


def main():
    logger.info("--- Starting PPC Section-based Ingestion Process ---")

    # 1. Validate Data Directory
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory '{DATA_DIR}' not found. Please create it and add your PPC PDF.")
        return

    # 2. Find all PDFs
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in '{DATA_DIR}'. Exiting.")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF file(s): {pdf_files}")

    # 3. Extract sections from all PDFs
    all_sections = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        sections = extract_sections_from_pdf(pdf_path)
        all_sections.extend(sections)
    
    if not all_sections:
        logger.warning("No sections extracted. Please check your PDF format.")
        return
    
    logger.info(f"Total sections extracted: {len(all_sections)}")
    
    # Display sample sections
    logger.info("=== Sample Sections ===")
    for i, section in enumerate(all_sections[:3]):
        logger.info(f"\nSection {i+1}:")
        logger.info(f"Metadata: {section.metadata}")
        logger.info(f"Content (first 200 chars): {section.page_content[:200]}...")

    # 4. Initialize Embeddings
    logger.info(f"Loading embedding model: {MODEL_NAME}...")
    try:
        # embeddings = HuggingFaceEmbeddings(
        #     model_name=MODEL_NAME,
        #     model_kwargs={
        #         "trust_remote_code": True,
        #         "revision": MODEL_REVISION,
        #         "device": "cpu"
        #     }
        # )
        embeddings = OllamaEmbeddings(model=MODEL_NAME)
        
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return

    # 5. Create and Save Vector Store
    logger.info(f"Creating FAISS index at '{DB_PATH}' with {len(all_sections)} sections...")
    try:
        # faiss_db = FAISS.from_documents(all_sections, embeddings)
        faiss_db = FAISS.from_documents(all_sections[:50], embeddings)
        for i in range(50, len(all_sections), 50):
            faiss_db.add_documents(all_sections[i:i+50])

        faiss_db.save_local(DB_PATH)
        logger.info(f"✅ Ingestion complete! Vector database saved with {len(all_sections)} sections.")
        logger.info(f"Database path: {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to create/save vector store: {e}")

if __name__ == "__main__":
    main()
