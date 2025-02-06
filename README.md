# Legal Document RAG with Taxonomy-Aware Hybrid Search 

A powerful Q&A application for legal documents that leverages Hybrid Search and Retrieval-Augmented Generation (RAG) with built-in legal taxonomy awareness. Built with RAGLite for robust document processing and retrieval and Streamlit for an intuitive chat interface, this system provides intelligent answers to legal queries while maintaining awareness of key legal domain concepts.

## Features

- **Dual-Mode Taxonomy Extraction**:
    - **Automatic Mode**: 
        - Fast regex-based keyword matching
        - Identifies exact matches from robust predefined legal taxonomy
        - Efficient for quick document processing
        - No API calls required
    - **Intelligent Mode**:
        - LLM-powered taxonomy analysis using GPT-4o-mini
        - Identifies both exact matches and semantically related concepts
        - Provides additional context through related keyword suggestions
        - More nuanced understanding of legal concepts

- **Advanced Search and Retrieval**:
    - **Hybrid Search System**:
        - Combines semantic search with traditional keyword matching
        - Uses OpenAI's text-embedding-3-large for semantic understanding
        - Supports up to 10 initial search results (retrieves the top 10 most relevant document chunks before reranking)
        - Optimized chunk size of 8000 tokens with 2-sentence overlapping windows (each document chunk contains 8000 tokens and overlaps with adjacent chunks by 2 sentences to maintain context)
    
    - **Intelligent Reranking**:
        - Powered by Cohere's reranking technology
        - Re-orders search results based on relevance to query
        - Improves context selection for more accurate answers
        - Language-aware reranking with English optimization

    - **Fallback Mechanism**:
        - Graceful degradation to general knowledge when no relevant documents found
        - Uses GPT-4o-mini for general legal knowledge
        - Maintains conversation context

- **Document Processing and UI**:
    - Page-level document chunking with metadata enrichment
    - Visual PDF page display for source verification
    - Progress tracking during document processing
    - Interactive chat interface with conversation history

- **Template-Based Configuration**:
    - The application uses Jinja2 templates for managing prompts and taxonomies, following software engineering best practices:

    - **Separation of Concerns**:
        - Prompts and taxonomies are maintained in separate template files
        - `templates/prompts.j2`: Contains all system prompts (RAG, extraction, fallback)
        - `templates/taxonomy.j2`: Contains the comprehensive legal taxonomy keywords
        
    - **Benefits**:
        - **Maintainability**: Edit prompts and taxonomies without touching application code
        - **Version Control**: Track changes to prompts and taxonomies separately
        - **Environment Flexibility**: Support different prompts/taxonomies per environment
        - **Reusability**: Templates can be shared across multiple applications
        - **Readability**: Clean separation between logic and content


## Prerequisites

You'll need the following API keys:

1. **API Keys**:
   - [OpenAI API key](https://platform.openai.com/api-keys) for:
     - GPT-4o model (chat completions)
     - text-embedding-3-large (embeddings)
     - GPT-4o-mini (intelligent taxonomy extraction)
   - [Cohere API key](https://dashboard.cohere.com/api-keys) for reranking

2. **Database Setup** (Optional):
   - Default: SQLite (no setup required)
   - Alternatively: Use any SQLAlchemy-compatible database

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Required System Dependencies**:
   - install both pypandoc and Pandoc via conda
   ```bash
   conda install -c conda-forge pypandoc pandoc
   ```

## Usage

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Configure the Application**:
   - Enter your OpenAI API key
   - Enter your Cohere API key
   - Configure database URL (optional, defaults to SQLite)
   - Select taxonomy extraction mode (Automatic or Intelligent)
   - Click "Save Configuration"

3. **Upload Documents**:
   - Upload PDF legal documents
   - The system will automatically:
     - Process documents page by page
     - Extract legal taxonomy keywords based on selected mode
     - Create searchable chunks with metadata
     - Display processing progress

4. **Ask Questions**:
   - Ask questions about your legal documents
   - View source information including:
     - Original document and page number
     - Extracted taxonomy keywords (exact matches and related concepts in Intelligent mode)
     - PDF page preview
   - System automatically falls back to general knowledge for non-document questions

## Legal Taxonomy

The system includes built-in recognition for over 100 legal concepts across various categories:
- Core Legal Areas (e.g., contract law, tort law, criminal law)
- Legal Processes & Procedures (e.g., civil procedure, arbitration)
- Legal Concepts & Principles (e.g., due process, liability)
- Rights & Protections (e.g., civil rights, privacy rights)
- Business & Commercial (e.g., securities regulation, intellectual property)
- Property & Real Estate (e.g., zoning, land use)
- Criminal Justice (e.g., felony, probable cause)
- Specialized Areas (e.g., healthcare law, cyber law)
- Government & Public Law (e.g., administrative law, regulatory compliance)
- Alternative Dispute Resolution (e.g., mediation, arbitration)

See the code for the complete list of supported taxonomy keywords.
