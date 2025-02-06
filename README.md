# Legal Document RAG with Taxonomy-Aware Hybrid Search 

A powerful Q&A application for legal documents that leverages Hybrid Search and Retrieval-Augmented Generation (RAG) with built-in legal taxonomy awareness. Built with RAGLite for robust document processing and retrieval and Streamlit for an intuitive chat interface, this system provides intelligent answers to legal queries while maintaining awareness of key legal domain concepts.

## Features

- **Taxonomy-Aware Document Processing**:
    - Automatic legal taxonomy keyword extraction
    - PDF document processing with metadata enrichment
    - Page-level document chunking with taxonomy information
    - Visual PDF page display for source verification

- **Advanced Search and Retrieval**:
    - Hybrid search combining semantic and keyword matching
    - Intelligent reranking for better context selection
    - Taxonomy-aware result presentation
    - Fallback to general knowledge for non-document queries

- **Multi-Model Integration**:
  - GPT-4o models for text generation
  - OpenAI text-embedding-3-large for embeddings
  - Cohere for reranking

## Prerequisites

You'll need the following API keys:

1. **API Keys**:
   - [OpenAI API key](https://platform.openai.com/api-keys) for GPT-4o and embeddings
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
   - Poppler (for PDF processing)
   - Python 3.8+

## Usage

1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Configure the Application**:
   - Enter your OpenAI API key
   - Enter your Cohere API key
   - Configure database URL (optional, defaults to SQLite)
   - Click "Save Configuration"

3. **Upload Documents**:
   - Upload PDF legal documents
   - The system will automatically:
     - Process documents page by page
     - Extract legal taxonomy keywords
     - Create searchable chunks with metadata

4. **Ask Questions**:
   - Ask questions about your legal documents
   - View source information including:
     - Original document and page number
     - Relevant taxonomy keywords
     - PDF page preview
   - System automatically falls back to general knowledge for non-document questions

## Legal Taxonomy

The system includes built-in recognition for key legal concepts including:
- Contract law
- Tort law
- Criminal law
- Civil rights
- Constitutional law
- Property law
- Family law
- And many more (see the code for the full list)
