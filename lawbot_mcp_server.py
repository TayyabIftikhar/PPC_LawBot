import os
import re
import logging
import sys
import time
from mcp.server.fastmcp import FastMCP
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ppc_vector_db_final") 
# LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "local_deepseek_model")
# EMBEDDING_MODEL = "./nomic-embed"
LOCAL_MODEL_PATH = "deepseek-r1:1.5b-qwen-distill-q8_0"
EMBEDDING_MODEL = "nomic-embed-text:latest"
EMBEDDING_REVISION = ""
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server_debug.log")


# --- Enhanced Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), 
        logging.StreamHandler(sys.stderr) 
    ]
)
logger = logging.getLogger("LawBotMCP")

class LocalLawEngine:
    def __init__(self):
        self.retriever = None
        self.llm = None
        self.rag_chain = None
        self.is_loaded = False

    def load_resources(self):
        if self.is_loaded:
            return True

        logger.info("⏳ [1/4] Starting heavy resource load...")
        try:
            # Import inside function to prevent startup timeout
            logger.info("   ... Importing LangChain libraries")
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_huggingface import HuggingFacePipeline
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough

            logger.info("⏳ [2/4] Loading Embeddings Model...")
            # embeddings = HuggingFaceEmbeddings(
            #     model_name=EMBEDDING_MODEL,
            #     model_kwargs={"trust_remote_code": True, "revision": EMBEDDING_REVISION, "torch_dtype": "auto"},
            # )
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

            if not os.path.exists(DB_PATH):
                logger.error(f"❌ Vector DB path invalid: {DB_PATH}")
                return False

            logger.info("⏳ [3/4] Loading Vector Database (FAISS)...")
            db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            self.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

            # if not os.path.exists(LOCAL_MODEL_PATH):
            #     logger.error(f"❌ Model path invalid: {LOCAL_MODEL_PATH}")
            #     return False

            logger.info("⏳ [4/4] Loading LLM (This is the slowest part)...")
            start_time = time.time()
            # self.llm = HuggingFacePipeline.from_model_id(
            #     model_id=LOCAL_MODEL_PATH,
            #     task="text-generation",
            #     model_kwargs={"trust_remote_code": True},
            #     pipeline_kwargs={
            #         "max_new_tokens": 512, 
            #         "temperature": 0.3,
            #         "do_sample": True,
            #         "return_full_text": False,
            #     },
            #     device_map="auto", 
            # )
            self.llm = OllamaLLM(
                model=LOCAL_MODEL_PATH,
                temperature=0.2,
                max_tokens=2048,
            )
            logger.info(f"✅ LLM Loaded in {time.time() - start_time:.2f} seconds.")

            # Define Chain
            template = """
            You are an expert legal assistant specializing in the Pakistan Penal Code (PPC).
            Use the context below to answer.

            CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:
            """
            prompt = ChatPromptTemplate.from_template(template)

            self.rag_chain = (
                RunnablePassthrough.assign(context=lambda inp: self._get_context(inp["question"]))
                | prompt
                | self.llm
                | StrOutputParser()
            )

            self.is_loaded = True
            logger.info("✅ All resources ready.")
            return True
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR during load: {e}", exc_info=True)
            return False

    def _get_context(self, query):
        logger.info(f"🔎 Retrieving docs for: {query}")
        docs = self.retriever.invoke(query)
        logger.info(f"   ... Found {len(docs)} documents.")
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def clean_output(self, text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.replace("</think>", "").strip()

    def query(self, question):
        logger.info(f"🚀 Received Query: {question}")
        
        if not self.is_loaded:
            logger.info("⚠️ Resources not loaded yet. Initializing...")
            success = self.load_resources()
            if not success:
                return "Error: Failed to load AI model. Check server_debug.log."
        
        try:
            logger.info("🤖 Generating answer (This may take 30-60s on CPU)...")
            start_time = time.time()
            
            # Run the chain
            raw = self.rag_chain.invoke({"question": question})
            
            duration = time.time() - start_time
            logger.info(f"✅ Generation complete in {duration:.2f} seconds.")
            return self.clean_output(raw)
        except Exception as e:
            logger.error(f"❌ ERROR generating answer: {e}", exc_info=True)
            return f"Error during generation: {str(e)}"

# --- MCP Server Setup ---
mcp = FastMCP("LawBot-Pakistan")
engine = LocalLawEngine()

@mcp.tool()
def ask_pakistan_law(question: str) -> str:
    """
    Ask a question about the Pakistan Penal Code (PPC). 
    """
    return engine.query(question)

if __name__ == "__main__":
    mcp.run()