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

## How to Use

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


## Configuration

### Milvus Collection
The system automatically creates a collection named `rag_collection` with the following schema:
- `id`: Primary key
- `embedding`: Vector field (1536 dimensions for text-embedding-3-small)
- `text`: Original text chunk

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `MILVUS_URI` | Milvus connection URI | Yes |
| `MILVUS_TOKEN` | Milvus authentication token | Yes |

## Docker Services

The project includes a complete Docker setup with:

- **etcd**: Distributed key-value store for Milvus metadata
- **MinIO**: Object storage for Milvus data
- **Milvus**: Vector database for similarity search

## Security

- API keys are stored in environment variables (not in code)
- `.env` file is excluded from version control
- Runtime data (`volumes/`) is excluded from version control

## 🛠️ Development

### Adding New File Types
To support additional file formats, modify the `extract_text_from_upload()` function in `api.py`.

### Customizing Embeddings
To use different embedding models, update the `generate_embedding()` function.

### Scaling
For production use, consider:
- Using Milvus Cloud instead of local setup
- Implementing proper authentication
- Adding rate limiting
- Using a production-grade web server

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [OpenAI](https://openai.com/) for embedding models
- [Milvus](https://milvus.io/) for vector database
- [Streamlit](https://streamlit.io/) for the web interface
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Note**: Make sure to replace placeholder values in the `.env` file with your actual API keys and credentials before running the application.