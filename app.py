import os
import re
import logging
import asyncio
import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# --- Configuration ---
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "ppc_vector_db_ref")
# LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "local_deepseek_model")
# EMBEDDING_MODEL = "./nomic-embed"
LOCAL_MODEL_PATH = "deepseek-r1:1.5b-qwen-distill-q8_0"
EMBEDDING_MODEL = "nomic-embed-text:latest"
EMBEDDING_REVISION = ""
MAX_HISTORY = 4

# --- Logging Setup ---
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "lawbot_app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("LawBotApp")

# --- Page Config ---
st.set_page_config(page_title="LawBot", page_icon="⚖️", layout="wide")
st.title("⚖️ LawBot: Legal Assistant for Pakistan Penal Code")


# ==========================================
# PART 1: LOCAL LAW ENGINE
# ==========================================
class LocalLawEngine:
    def __init__(self):
        self.retriever = None
        self.llm = None
        self.rag_chain = None

    def load_resources(self):
        if self.llm:
            return True

        logger.info("Loading local resources...")
        try:
            # embeddings = HuggingFaceEmbeddings(
            #     model_name=EMBEDDING_MODEL,
            #     model_kwargs={
            #         "trust_remote_code": True,
            #         "revision": EMBEDDING_REVISION,
            #         "device": 0,
            #     },
            # )
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

            if not os.path.exists(DB_PATH):
                msg = f"Vector DB not found at {DB_PATH}"
                logger.error(msg)
                st.error(msg)
                return False

            db = FAISS.load_local(
                DB_PATH, embeddings, allow_dangerous_deserialization=True
            )
            self.retriever = db.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )

            # if not os.path.exists(LOCAL_MODEL_PATH):
            #     msg = f"Model not found at {LOCAL_MODEL_PATH}"
            #     logger.error(msg)
            #     st.error(msg)
            #     return False

            # self.llm = HuggingFacePipeline.from_model_id(
            #     model_id=LOCAL_MODEL_PATH,
            #     task="text-generation",
            #     model_kwargs={"trust_remote_code": True, "torch_dtype": "auto"},
            #     pipeline_kwargs={
            #         "max_new_tokens": 2048,
            #         "temperature": 0.2,
            #         "do_sample": True,
            #         "return_full_text": False,
            #         "top_p": 0.9,
            #     },
            #     device_map="auto",
            # )
            self.llm = OllamaLLM(
                model=LOCAL_MODEL_PATH,
                temperature=0.2,
                max_tokens=2048,
            )

            self._build_chain()
            logger.info("Local resources loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading local resources: {e}")
            st.error(f"Error loading local resources: {e}")
            return False

    def _build_chain(self):
        template = """
            You are an expert legal assistant specializing in the Pakistan Penal Code (PPC).
            Use ONLY the provided context. If a detail is not in the context, say explicitly that it is not available instead of guessing.

            RESPONSE FORMAT (concise and grounded):
            1) SECTION NUMBER & TITLE: [cite the PPC section and title; if not in context, say "Not found in provided context."]
            2) DEFINITION: [plain-English definition; if missing, say "Not provided in context."]
            3) KEY ELEMENTS/INGREDIENTS: [bullet the required elements; if missing, say "Not provided in context."]
            4) PUNISHMENT:
                - Imprisonment: [duration/type or "Not provided in context."]
                - Fine: [amount/terms or "Not provided in context."]
                - Other: [any other penalties or "Not provided in context."]
            5) EXPLANATION (SIMPLE WORDS): [2-3 short paragraphs; if context is thin, state that explicitly.]
            6) IMPORTANT NOTES / LIMITATIONS: [exceptions, caveats, or "No additional notes in context."]

            CRITICAL INSTRUCTIONS:
            - Do NOT invent facts. If something is absent, say it is not in the provided context.
            - Do NOT repeat the entire answer or add filler like "Based on the documents".
            - Stay concise and structured; prefer bullets where natural.
            - If no relevant section is found, respond: "The specific section regarding this query is not available in the current database."

            CONTEXT:
            {context}

            CHAT HISTORY:
            {chat_history}

            QUESTION:
            {question}

            ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(template)

        self.rag_chain = (
            RunnablePassthrough.assign(
                context=lambda inp: self._get_context(inp["question"]),
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _get_context(self, query):
        logger.info(f"Retrieving local context for: {query}")
        docs = self.retriever.invoke(query)
        st.session_state.last_context = docs
        # return "\n\n---\n\n".join(d.page_content for d in docs)
        formatted_chunks = []

        for d in docs:
            page = d.metadata.get("page", "Unknown")
            section = d.metadata.get("section_number", "Unknown")
            formatted_chunks.append(
                f"[Section {section}, Page {page}]\n{d.page_content}"
            )

        return "\n\n---\n\n".join(formatted_chunks)


    def clean_output(self, text):
        """Clean and format the LLM output."""
        # Remove <think> tags
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.replace("</think>", "")

        # Remove duplicate sentences/paragraphs
        lines = text.split("\n")
        unique_lines = []
        seen = set()
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen:
                unique_lines.append(line)
                seen.add(line_stripped)
            elif not line_stripped:
                unique_lines.append(line)  # Keep empty lines for formatting

        text = "\n".join(unique_lines)

        # Remove common "meta" phrases if the model ignores the negative prompt
        text = text.replace("Here is the answer based on the context:", "")
        text = text.replace("Based on the provided context:", "")
        text = text.replace("Here is the response:", "")
        text = text.replace("The answer is:", "")

        return text.strip()

    def _build_references(self):
        docs = st.session_state.get("last_context", [])

        if not docs:
            return (
                "SOURCE REFERENCE:\n"
                "- Document: Pakistan Penal Code.pdf\n"
            )

        # Aggregate sections per page
        page_sections_map = {}
        sources_set = set()

        for d in docs:
            page = d.metadata.get("page", "Unknown")
            section = d.metadata.get("section_number", "Unknown")
            source = d.metadata.get("source", "Pakistan Penal Code.pdf")
            sources_set.add(source)

            if page not in page_sections_map:
                page_sections_map[page] = []
            page_sections_map[page].append(str(section))

        # Build readable string: group sections by page
        sections_per_page_lines = []
        for page in sorted(page_sections_map.keys(), key=lambda x: int(x) if str(x).isdigit() else 9999):
            sections = sorted(page_sections_map[page], key=lambda x: int(x) if str(x).isdigit() else 9999)
            sections_str = ", ".join(sections)
            sections_per_page_lines.append(f"Page {page}: Sections {sections_str}")

        source_str = ", ".join(sorted(sources_set))

        return (
            "SOURCE REFERENCE:\n"
            f"- Document: {source_str}\n"
            + "\n".join(f"- {line}" for line in sections_per_page_lines)
        )


    def query(self, question, history_str):
        if not self.load_resources():
            return "System Error: Resources failed to load."
        
        docs = st.session_state.get("last_context", [])
    
        # If no documents retrieved, ask user to clarify
        if not docs:
            return (
                "I could not find relevant information in the Pakistan Penal Code database. "
                "Please provide more details about your question so I can give you a precise answer."
            )
        
        logger.info(f"Running Local RAG for: {question}")
        raw = self.rag_chain.invoke({"question": question, "chat_history": history_str})
        
        # raw = """
        # 1. The Offense: Qatl-i-Khata (Accidental Killing)
        # If you killed a person by mistake without the intention to cause death or harm (e.g., aiming at an attacker but hitting someone else, or accidental discharge), it falls under Qatl-i-Khata.
        # Section 318 (Qatl-i-Khata): Defines causing death "without any intention to cause death... either by mistake of act or by mistake of fact."
        # Punishment (Section 319):
        # Diyat: You are liable to pay Diyat (financial compensation to the victim's heirs).
        # Imprisonment: If the act was "rash or negligent" (e.g., firing a gun carelessly), you may also face imprisonment of either description for a term which may extend to five years as ta'zir.
        # 2. The Defense: Right of Private Defense
        # If you can prove you were acting in lawful self-defense, your action might not be considered an offense at all.
        # Section 96: "Nothing is an offence which is done in the exercise of the right of private defence."
        # Section 100: This section lists when you are allowed to voluntarily cause death in self-defense, such as when under assault that causes reasonable apprehension of death, grievous hurt, rape, or kidnapping.
        # Section 106 (Risk to Innocent Person): If you were under a deadly assault and had no choice but to fire, running the risk of harming an innocent person, this section protects you. (Example: Firing at a mob attacking you and accidentally hitting a bystander).
        # Summary: You will likely be charged under Section 319 (liable for Diyat and up to 5 years in prison). However, if you prove the act was necessary for self-defense against a deadly threat, Section 96 and Section 106 serve as your complete legal defense to exonerate you.
        # """

        cleaned = self.clean_output(raw)

        references = self._build_references()

        return f"{cleaned}\n\n{references}"
        # return self.clean_output(raw)


if "law_engine" not in st.session_state:
    st.session_state.law_engine = LocalLawEngine()


# ==========================================
# PART 2: MCP CLIENT (Perplexity Connection)
# ==========================================
async def query_perplexity_mcp(query):
    """
    Connects to the Perplexity MCP Server using stdio.
    """
    logger.info(f"Initiating Perplexity MCP connection for: {query}")

    if not PERPLEXITY_API_KEY:
        return "Error: PERPLEXITY_API_KEY is missing. Please set it in server.py."

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@chatmcp/server-perplexity-ask"],
        env={"PERPLEXITY_API_KEY": PERPLEXITY_API_KEY},
    )

    # --- WEB SEARCH PROMPT ---
    strict_legal_prompt = f"""
    You are a legal expert on Pakistani Law. 
    Search query: "{query}"
    
    INSTRUCTIONS:
    1. Answer ONLY according to the **Pakistan Penal Code (PPC)** and relevant Pakistani laws.
    2. Do NOT provide laws from India, UK, or USA.
    3. Cite the specific **Section Number** of the PPC (e.g., Section 302).
    4. Provide the exact punishment (imprisonment term and fine).
    5. Explain the concept in simple, layman terms.
    6. Provide the answer directly without introductory filler.
    """

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                if not tools.tools:
                    logger.error("No tools found in Perplexity MCP server.")
                    return "Error: No tools found in MCP server."

                tool_name = tools.tools[0].name
                logger.info(f"Calling MCP tool: {tool_name}")

                arguments = {
                    "messages": [{"role": "user", "content": strict_legal_prompt}]
                }

                result = await session.call_tool(tool_name, arguments=arguments)

                if result.content and len(result.content) > 0:
                    logger.info("Received response from Perplexity.")
                    return result.content[0].text

                logger.warning("No content in Perplexity response.")
                return "No response from Perplexity."
    except Exception as e:
        logger.error(f"MCP Connection Error: {str(e)}")
        return f"MCP Connection Error: {str(e)}"


# ==========================================
# PART 3: STREAMLIT UI
# ==========================================

with st.sidebar:
    st.header("Mode Selection")
    mode = st.radio(
        "Choose Intelligence Source:",
        ["📚 Local Database (PPC)", "🌐 Web Search (Perplexity)"],
        captions=["Uses local FAISS + DeepSeek", "Uses external MCP Server"],
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        logger.info("Chat cleared by user.")
        st.rerun()

    # if mode == "📚 Local Database (PPC)":
    #     if st.checkbox("Show Retrieved Context"):
            # if "last_context" in st.session_state and st.session_state.last_context:
            #     st.write("### Local Documents:")
            #     for i, doc in enumerate(st.session_state.last_context):
            #         with st.expander(f"Chunk {i+1}"):
            #             st.text(doc.page_content)
            # for i, doc in enumerate(st.session_state.last_context):
            #     page = doc.metadata.get("page", "Unknown")
            #     section = doc.metadata.get("section_number", "Unknown")
            #     with st.expander(f"Section {section} | Page {page}"):
            #         st.text(doc.page_content)


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask a legal question...")

if user_input:
    logger.info(f"User Query ({mode}): {user_input}")

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        with st.spinner(f"Thinking using {mode}..."):

            if mode == "📚 Local Database (PPC)":
                history_msgs = st.session_state.messages[-MAX_HISTORY:]
                history_str = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in history_msgs]
                )
                full_response = st.session_state.law_engine.query(
                    user_input, history_str
                )

            else:
                full_response = asyncio.run(query_perplexity_mcp(user_input))

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
