from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
import io
from dotenv import load_dotenv
import fitz  # PyMuPDF
import docx
from pymilvus import MilvusClient

# --- 1. Milvus & OpenAI Setup ---
load_dotenv()
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "rag_collection"

if not MILVUS_URI or not MILVUS_TOKEN or not OPENAI_API_KEY:
    raise ValueError("One or more environment variables are missing. Please check your .env file.")

# Initialize clients
milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- 2. Pydantic Models ---
class UploadResponse(BaseModel):
    filename: str
    message: str

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# --- 3. Core RAG Functions ---
def extract_text_from_upload(file: UploadFile) -> str:
    """Extracts text from an uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="File has no name.")
    
    try:
        content = file.file.read()
        # Reset file pointer for potential future reads
        file.file.seek(0)
        
        text = ""
        if file.filename.lower().endswith(".pdf"):
            try:
                with fitz.open(stream=io.BytesIO(content), filetype="pdf") as doc:
                    text = "".join(page.get_text() for page in doc)  # type: ignore
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading PDF file: {str(e)}")
        elif file.filename.lower().endswith(".docx"):
            try:
                doc = docx.Document(io.BytesIO(content))
                text = "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading DOCX file: {str(e)}")
        elif file.filename.lower().endswith(".txt"):
            try:
                # Try UTF-8 first, then fallback to other encodings
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = content.decode("latin-1")
                except UnicodeDecodeError:
                    text = content.decode("cp1252", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .txt, .pdf, or .docx files.")
        
        return text.strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error processing file: {str(e)}")

def generate_chunks(text: str) -> list[str]:
    """Splits text into smaller chunks."""
    # This is a placeholder for more sophisticated chunking
    return [text[i:i + 1000] for i in range(0, len(text), 900)]

def generate_embedding(chunks: list[str]) -> list[dict]:
    """Generates embeddings for a list of text chunks."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    embeddings = [item.embedding for item in response.data]
    return [{"embedding": vec, "text": chunk} for vec, chunk in zip(embeddings, chunks)]

def insert_embeddings_into_milvus(data: list[dict]):
    """Inserts data into the Milvus collection."""
    milvus_client.insert(collection_name=COLLECTION_NAME, data=data)

# --- 4. FastAPI App ---
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Check if the collection exists and create it if not
    if COLLECTION_NAME not in milvus_client.list_collections():
        milvus_client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=1536,  # Dimension for text-embedding-3-small
            primary_field_name="id",
            vector_field_name="embedding",
            auto_id=True
        )
        # Add an index for faster similarity search
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="L2"
        )
        milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
        print(f"Collection '{COLLECTION_NAME}' created and indexed.")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Handles file uploads, extracts text, generates embeddings,
    and stores them in the Milvus database.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name cannot be empty.")

        text = extract_text_from_upload(file)

        if not text.strip():
            raise HTTPException(status_code=400, detail=f"File {file.filename} is empty or contains no text.")

        chunks = generate_chunks(text)
        print(f"Generated {len(chunks)} chunks from {file.filename}.")

        embedded_data = generate_embedding(chunks)
        print(f"Generated embeddings for {file.filename}.")
        
        insert_embeddings_into_milvus(embedded_data)
        print(f"Successfully inserted embeddings for {file.filename} into Milvus.")
        
        return UploadResponse(filename=file.filename, message="File processed and embedded successfully.")
    
    except Exception as e:
        print(f"Error during file upload: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_collection(request: QueryRequest):
    """
    Takes a user's query, embeds it, searches Milvus, and returns a response from an LLM.
    """
    try:
        # 1. Embed the user's query
        query_embedding = generate_embedding([request.query])[0]['embedding']

        # 2. Search for similar vectors in Milvus
        search_results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=3,
            output_fields=["text"]
        )
        
        context = " ".join([hit['entity']['text'] for res in search_results for hit in res])

        # 3. Construct the prompt and get a response from OpenAI
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {request.query}"
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a specialized assistant for answering questions based ONLY on the provided context. Do not use any outside knowledge. If the answer is not found in the context, you MUST state that you 'could not find an answer in the provided documents.'"},
                {"role": "user", "content": prompt}
            ]
        )
        
        final_response = response.choices[0].message.content or "The model did not return a response."
        return QueryResponse(response=final_response)

    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during the query: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 