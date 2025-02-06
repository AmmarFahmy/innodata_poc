import os
import re
import json
import hashlib
import logging
import streamlit as st
import PyPDF2
from raglite import RAGLiteConfig, insert_document, hybrid_search, retrieve_chunks, rerank_chunks, rag
from rerankers import Reranker
from typing import List
from pathlib import Path
import openai
import time
import warnings

from pdf2image import convert_from_bytes

# Setup logging and ignore specific warnings.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Define the system prompt for the RAG assistant.
RAG_SYSTEM_PROMPT = """
You are a friendly and knowledgeable legal assistant that provides complete and insightful answers.
Answer the user's question using only the context provided.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, treat the context as if it were entirely part of your working memory.
""".strip()

# ------------------------------------------
# 1. Predefined Legal Taxonomy
# ------------------------------------------
LEGAL_TAXONOMY_KEYWORDS = [
    # Core Legal Areas
    "contract law", "tort law", "criminal law", "civil law", "constitutional law",
    "property law", "family law", "intellectual property", "corporate law", "tax law",
    "administrative law", "environmental law", "labor law", "immigration law",
    "bankruptcy law", "securities law", "antitrust law", "international law",

    # Legal Processes & Procedures
    "civil procedure", "criminal procedure", "evidence", "jurisdiction", "arbitration",
    "mediation", "litigation", "appeal", "discovery", "pleadings", "injunction",
    "class action", "settlement", "trial", "hearing", "deposition",

    # Legal Concepts & Principles
    "due process", "precedent", "statute", "regulation", "liability", "negligence",
    "damages", "remedy", "standing", "jurisdiction", "venue", "immunity",
    "consideration", "breach", "fraud", "defamation", "estoppel",

    # Rights & Protections
    "civil rights", "human rights", "privacy rights", "discrimination",
    "equal protection", "freedom of speech", "freedom of religion",
    "right to counsel", "miranda rights", "fourth amendment", "fifth amendment",

    # Business & Commercial
    "mergers and acquisitions", "securities regulation", "commercial law",
    "partnership law", "llc law", "agency law", "employment law", "trade law",
    "consumer protection", "unfair competition", "trademark", "patent", "copyright",

    # Property & Real Estate
    "real property", "personal property", "easement", "zoning", "land use",
    "landlord tenant", "mortgage", "title", "deed", "conveyance",

    # Criminal Justice
    "felony", "misdemeanor", "mens rea", "actus reus", "probable cause",
    "search and seizure", "self defense", "double jeopardy", "plea bargain",

    # Specialized Areas
    "healthcare law", "education law", "elder law", "military law", "maritime law",
    "aviation law", "sports law", "entertainment law", "cyber law", "blockchain law",
    "data privacy", "artificial intelligence law", "environmental compliance",

    # Government & Public Law
    "municipal law", "state law", "federal law", "legislative process",
    "executive power", "judicial review", "administrative procedure",
    "public policy", "regulatory compliance", "government contracts",

    # Alternative Dispute Resolution
    "negotiation", "conciliation", "dispute resolution", "binding arbitration",
    "non-binding arbitration", "mediation agreement", "settlement conference"
]

# ------------------------------------------
# 2. Automatic Taxonomy Extraction (Regex-based)
# ------------------------------------------


def extract_taxonomy_keywords_automatic(text: str, taxonomy: list) -> list:
    """
    Return a list of taxonomy keywords that appear in the text using regex matching.
    """
    found_keywords = []
    for keyword in taxonomy:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text, flags=re.IGNORECASE):
            found_keywords.append(keyword)
    return found_keywords

# ------------------------------------------
# 3. Intelligent Taxonomy Extraction (LLM-based)
# ------------------------------------------


def extract_taxonomy_keywords_intelligent(text: str, taxonomy: list) -> tuple:
    """
    Uses GPT-4o-mini to extract taxonomy keywords from the page content.
    The assistant is provided with both the page content and the list of legal taxonomy keywords.
    It returns a tuple (exact_matches, related_keywords) where:
      - exact_matches: a list of keywords that exactly appear in the content (if any)
      - related_keywords: a list of 5 highly relevant taxonomy keywords.
    If no exact matches are found, only related_keywords are provided.
    """
    try:
        # Mimic the fallback function style.
        client = openai.OpenAI(
            api_key=st.session_state.user_env["OPENAI_API_KEY"])
        system_prompt = (
            "You are a legal taxonomy extraction assistant. "
            "Given the following page content and a list of legal taxonomy keywords, "
            "identify all keywords from the list that exactly appear in the page content. "
            "Then, suggest 5 additional legal taxonomy keywords that are highly relevant to the content. "
            "If no exact matches are found, just provide 5 related keywords. "
            "Return your answer as a JSON object with two keys: exact_matches and related_keywords. "
            "Do not include any extra text."
        )
        user_prompt = f"Taxonomy keywords: {', '.join(taxonomy)}\n\nPage content:\n{text}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        result_text = response.choices[0].message.content.strip()
        logger.info("LLM extraction result: " + result_text)
        try:
            data = json.loads(result_text)
        except Exception as parse_error:
            logger.error(
                "JSON parsing error in intelligent extraction: " + str(parse_error))
            logger.error("LLM result was: " + result_text)
            return ([], [])
        exact_matches = data.get("exact_matches", [])
        related_keywords = data.get("related_keywords", [])
        logger.info(f"Exact matches: {exact_matches}")
        logger.info(f"Related keywords: {related_keywords}")
        return (exact_matches, related_keywords)
    except Exception as e:
        logger.error("LLM extraction error: " + str(e))

        return ([], [])

# ------------------------------------------
# 4. Helper Function: Parse Chunk Text for Metadata
# ------------------------------------------


def parse_chunk_text(chunk_text: str):
    """
    Expects the chunk text to be formatted as:

    ===PAGE_INFO===
    Document: <doc_name>
    DocHash: <doc_hash>
    Page: <page_number>
    Taxonomy: <header_line>
    ===CONTENT===
    <actual page content>

    For Intelligent mode, header_line may be formatted as:
    <exact_matches> | Related: <related_keywords>

    Returns a tuple: (doc_name, doc_hash, page_number, taxonomy_info, actual_content)
    taxonomy_info is returned as a string.
    """
    doc_name = "Unknown"
    doc_hash = "Unknown"
    page_num = "Unknown"
    taxonomy_info = ""
    content = chunk_text
    if chunk_text.startswith("===PAGE_INFO==="):
        parts = chunk_text.split("===CONTENT===")
        if len(parts) >= 2:
            header = parts[0]
            content = "===CONTENT===".join(parts[1:]).strip()
            for line in header.splitlines():
                if line.startswith("Document:"):
                    doc_name = line.split("Document:")[1].strip()
                elif line.startswith("DocHash:"):
                    doc_hash = line.split("DocHash:")[1].strip()
                elif line.startswith("Page:"):
                    page_num = line.split("Page:")[1].strip()
                elif line.startswith("Taxonomy:"):
                    taxonomy_info = line.split("Taxonomy:")[1].strip()
    return doc_name, doc_hash, page_num, taxonomy_info, content

# ------------------------------------------
# 5. Configuration Initialization
# ------------------------------------------


def initialize_config(openai_key: str, cohere_key: str, db_url: str) -> RAGLiteConfig:
    try:
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["COHERE_API_KEY"] = cohere_key
        return RAGLiteConfig(
            db_url=db_url,
            llm="gpt-4o",
            embedder="text-embedding-3-large",
            embedder_normalize=True,
            chunk_max_size=8000,
            embedder_sentence_window_size=2,
            reranker=Reranker("cohere", api_key=cohere_key, lang="en")
        )
    except Exception as e:
        raise ValueError(f"Configuration error: {e}")

# ------------------------------------------
# 6. Document Processing: Page-Wise Chunking with Metadata Injection and Progress UI
# ------------------------------------------


def process_document(file_path: str, doc_hash: str, doc_name: str) -> bool:
    try:
        if not st.session_state.get('my_config'):
            raise ValueError("Configuration not initialized")

        # Sanitize document name to avoid encoding issues
        doc_name = doc_name.encode('ascii', 'replace').decode('ascii')

        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            logger.info(f"Processing PDF '{doc_name}' with {num_pages} pages.")
            progress_bar = st.progress(0)
            status_text = st.empty()

            for page_index in range(num_pages):
                status_text.text(
                    f"Processing page {page_index+1} of {num_pages}...")
                with st.spinner(f"Processing page {page_index+1}..."):
                    try:
                        page = pdf_reader.pages[page_index]

                        # Extract text and handle encoding more robustly
                        raw_text = page.extract_text() or ""

                        # Convert text to plain ASCII, replacing non-ASCII characters
                        text = raw_text.encode(
                            'ascii', 'replace').decode('ascii')

                        # Remove any remaining problematic characters
                        text = ''.join(
                            char for char in text if ord(char) < 128)

                        extraction_mode = st.session_state.get(
                            "extraction_mode", "Automatic")

                        if extraction_mode == "Intelligent":
                            exact_matches, related_keywords = extract_taxonomy_keywords_intelligent(
                                text, LEGAL_TAXONOMY_KEYWORDS)
                            logger.info(f"Exact matches: {exact_matches}")
                            logger.info(
                                f"Related keywords: {related_keywords}")
                            if exact_matches:
                                header_line = f"{', '.join(exact_matches)} | Related: {', '.join(related_keywords)}"
                            else:
                                header_line = f"{', '.join(related_keywords)}"
                        else:
                            tax_keywords = extract_taxonomy_keywords_automatic(
                                text, LEGAL_TAXONOMY_KEYWORDS)
                            header_line = f"{', '.join(tax_keywords) if tax_keywords else 'None'}"

                        # Create safe filename for temporary file
                        safe_doc_name = ''.join(
                            c for c in doc_name if c.isalnum() or c in ('-', '_'))
                        temp_page_file = f"temp_page_{safe_doc_name}_{page_index+1}.txt"

                        # Write the temporary file using ASCII encoding
                        with open(temp_page_file, "w", encoding='ascii', errors='replace') as tmp:
                            header = (
                                "===PAGE_INFO===\n"
                                f"Document: {doc_name}\n"
                                f"DocHash: {doc_hash}\n"
                                f"Page: {page_index+1}\n"
                                f"Taxonomy: {header_line}\n"
                                "===CONTENT===\n"
                            )
                            tmp.write(header)
                            tmp.write(text)

                        insert_document(Path(temp_page_file),
                                        config=st.session_state.my_config)
                        os.remove(temp_page_file)
                        progress_bar.progress((page_index + 1) / num_pages)

                    except Exception as page_error:
                        logger.error(
                            f"Error processing page {page_index+1}: {str(page_error)}")
                        continue

            status_text.text("Processing complete!")
        return True

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False

# ------------------------------------------
# 7. Search and Fallback Functions
# ------------------------------------------


def perform_search(query: str) -> List:
    try:
        chunk_ids, scores = hybrid_search(
            query, num_results=10, config=st.session_state.my_config)
        if not chunk_ids:
            return []
        chunks = retrieve_chunks(chunk_ids, config=st.session_state.my_config)
        return rerank_chunks(query, chunks, config=st.session_state.my_config)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []


def handle_fallback(query: str) -> str:
    try:
        client = openai.OpenAI(
            api_key=st.session_state.user_env["OPENAI_API_KEY"])
        system_prompt = (
            "You are a helpful AI assistant. When you don't know something, "
            "be honest about it. Provide clear, concise, and accurate responses. "
            "If the question is not related to any specific document, use your general knowledge to answer."
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Fallback error: {str(e)}")
        st.error(f"Fallback error: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

# ------------------------------------------
# 8. Main Streamlit App
# ------------------------------------------


def main():
    st.set_page_config(page_title="Innodata - Taxonomy RAG POC", layout="wide")
    for state_var in ['chat_history', 'documents_loaded', 'my_config', 'user_env', 'processed_pdf_hashes', 'pdf_files']:
        if state_var not in st.session_state:
            if state_var == 'chat_history':
                st.session_state[state_var] = []
            elif state_var == 'documents_loaded':
                st.session_state[state_var] = False
            elif state_var == 'my_config':
                st.session_state[state_var] = None
            elif state_var == 'user_env':
                st.session_state[state_var] = {}
            elif state_var == 'processed_pdf_hashes':
                st.session_state[state_var] = set()
            elif state_var == 'pdf_files':
                st.session_state[state_var] = {}
    with st.sidebar:
        st.title("Configuration")
        openai_key = st.text_input("OpenAI API Key", value=st.session_state.get(
            'openai_key', ''), type="password", placeholder="sk-...")
        cohere_key = st.text_input("Cohere API Key", value=st.session_state.get(
            'cohere_key', ''), type="password", placeholder="Enter Cohere key")
        db_url = st.text_input("Database URL", value=st.session_state.get(
            'db_url', 'sqlite:///raglite.sqlite'), placeholder="sqlite:///raglite.sqlite")
        if not st.session_state.documents_loaded:
            extraction_mode = st.radio("Select Taxonomy Extraction Mode", options=[
                                       "Automatic", "Intelligent"], index=0)
            st.session_state["extraction_mode"] = extraction_mode
        else:
            st.write("Taxonomy Extraction Mode: " +
                     st.session_state.get("extraction_mode", "Automatic"))
        if st.button("Save Configuration"):
            try:
                if not all([openai_key, cohere_key, db_url]):
                    st.error("All fields are required!")
                    return
                st.session_state['openai_key'] = openai_key
                st.session_state['cohere_key'] = cohere_key
                st.session_state['db_url'] = db_url
                st.session_state.my_config = initialize_config(
                    openai_key=openai_key, cohere_key=cohere_key, db_url=db_url)
                st.session_state.user_env = {"OPENAI_API_KEY": openai_key}
                st.success("Configuration saved successfully!")
            except Exception as e:
                st.error(f"Configuration error: {str(e)}")
    st.title("Innodata - Taxonomy POC - RAG with Hybrid Search")
    if not st.session_state.documents_loaded:
        uploaded_files = st.file_uploader("Upload PDF legal documents", type=[
                                          "pdf"], accept_multiple_files=True, key="pdf_uploader")
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.md5(file_bytes).hexdigest()
                if file_hash in st.session_state.processed_pdf_hashes:
                    st.warning(
                        f"'{uploaded_file.name}' has already been uploaded. Skipping duplicate.")
                    continue
                else:
                    st.session_state.processed_pdf_hashes.add(file_hash)
                    st.session_state.pdf_files[file_hash] = file_bytes
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file_bytes)
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        if process_document(temp_path, file_hash, uploaded_file.name):
                            st.success(
                                f"Successfully processed: {uploaded_file.name}")
                        else:
                            st.error(
                                f"Failed to process: {uploaded_file.name}")
                    os.remove(temp_path)
            st.session_state.documents_loaded = True
            st.success(
                "All documents are ready! You can now ask questions about them.")
    else:
        st.info("Documents already processed. You can ask your questions below.")
    if st.session_state.documents_loaded:
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(msg[0])
            with st.chat_message("assistant"):
                st.write(msg[1])
        user_input = st.chat_input("Ask a question about the documents...")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    reranked_chunks = perform_search(query=user_input)
                    if not reranked_chunks or len(reranked_chunks) == 0:
                        logger.info(
                            "No relevant documents found. Falling back to general LLM.")
                        st.info(
                            "No relevant documents found. Using general knowledge to answer.")
                        full_response = handle_fallback(user_input)
                        message_placeholder.markdown(full_response)
                    else:
                        best_chunk = reranked_chunks[0]
                        raw_text = best_chunk.body
                        doc_name, doc_hash, page_number, taxonomy_info, content_without_header = parse_chunk_text(
                            raw_text)
                        formatted_messages = [
                            {"role": "user" if i %
                                2 == 0 else "assistant", "content": msg}
                            for i, msg in enumerate([m for pair in st.session_state.chat_history for m in pair])
                            if msg
                        ]
                        response_stream = rag(
                            prompt=user_input,
                            system_prompt=RAG_SYSTEM_PROMPT,
                            search=hybrid_search,
                            messages=formatted_messages,
                            max_contexts=5,
                            config=st.session_state.my_config
                        )
                        full_response = ""
                        for chunk in response_stream:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                        with st.expander("Top Matched Source Information:", expanded=False):
                            st.write(f"**Document:** {doc_name}")
                            st.write(f"**Page:** {page_number}")
                            if st.session_state.get("extraction_mode") == "Intelligent" and "|" in taxonomy_info:
                                parts = taxonomy_info.split("|")
                                exact_matches = parts[0].strip()
                                related_keywords = parts[1].replace(
                                    "Related:", "").strip()
                                st.write(f"**Exact Matches:** {exact_matches}")
                                st.write(
                                    f"**Related Keywords:** {related_keywords}")
                            else:
                                st.write(
                                    f"**Taxonomy Keywords:** {taxonomy_info if taxonomy_info else 'None'}")
                            if doc_hash in st.session_state.pdf_files:
                                pdf_bytes = st.session_state.pdf_files[doc_hash]
                                try:
                                    page_num_int = int(page_number)
                                    pages = convert_from_bytes(
                                        pdf_bytes, first_page=page_num_int, last_page=page_num_int)
                                    if pages:
                                        st.image(
                                            pages[0], caption=f"{doc_name} - Page {page_number}")
                                except Exception as e:
                                    st.error(
                                        "Could not convert PDF page to image: " + str(e))
                    st.session_state.chat_history.append(
                        (user_input, full_response))
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        if not st.session_state.my_config:
            st.info("Please configure your API keys to get started.")
        else:
            st.info("Please upload some documents to get started.")


if __name__ == "__main__":
    main()
