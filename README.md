# RAG Implementation Project

A full-stack Retrieval-Augmented Generation (RAG) system that allows users to upload documents, generate embeddings, and ask questions to get intelligent answers based on the document content.

## Features

- **Document Upload**: Support for PDF, DOCX, and TXT files
- **Vector Embeddings**: Uses OpenAI's text-embedding-3-small model for generating embeddings
- **Vector Database**: Milvus for efficient similarity search
- **Chat Interface**: Streamlit-based web interface for document Q&A
- **REST API**: FastAPI backend for document processing and querying
- **Docker Support**: Complete containerized setup with Milvus, MinIO, and etcd

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- OpenAI API key
- Milvus Cloud account (or local Milvus setup)

## Usage

### 1. Start the FastAPI Backend
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Streamlit Frontend
```bash
streamlit run streamlit_app.py
```

### 3. Access the Application
- **Frontend**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

## 📖 How to Use

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, or TXT files
2. **Process Documents**: Click "Embed All Files" to generate embeddings
3. **Ask Questions**: Use the chat interface to ask questions about your documents
4. **Get Answers**: Receive intelligent responses based on your document content

## API Endpoints

### POST `/upload`
Upload and process documents for embedding generation.

**Request**: Multipart form data with file
**Response**: 
```json
{
  "filename": "document.pdf",
  "message": "Document processed successfully"
}
```

### POST `/query`
Query the document collection for answers.

**Request**:
```json
{
  "query": "What is the main topic of the document?"
}
```

**Response**:
```json
{
  "response": "The main topic is..."
}
```