import os
import io
import fitz  # PyMuPDF
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

UPLOAD_FOLDER = "uploads"
MODEL = "text-embedding-3-small"

def extract_text_from_file(file_name: str, file_content: bytes):
    """
    Extracts plain text from an in-memory file content based on its extension.
    Supports .txt, .pdf, and .docx files.
    """
    if file_name.endswith(".pdf"):
        # Open PDF from bytes
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            # The 'get_text()' method is correct, but let's be explicit with the page iteration
            # to avoid any ambiguity that might confuse linters or older library versions.
            text = ""
            for page in doc:
                text += page.get_text() # type: ignore
            return text
    elif file_name.endswith(".docx"):
        # Open DOCX from bytes
        doc = docx.Document(io.BytesIO(file_content))
        return "\n".join(para.text for para in doc.paragraphs)
    elif file_name.endswith(".txt"):
        # Decode TXT from bytes
        return file_content.decode("utf-8")
    else:
        # Return None for unsupported file types
        return None

def upload_documents(folder=UPLOAD_FOLDER):
    """
    DEPRECATED for API use.
    This function reads documents from a local folder and is kept for local testing.
    """
    docs = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, "rb") as f:
                content_bytes = f.read()
            text = extract_text_from_file(filename, content_bytes)
            if text:
                docs.append({"filename": filename, "content": text.strip()})
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    return docs

def generate_chunks(documents_string):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(documents_string)

    return [{"id": i, "text": chunk} for i, chunk in enumerate(chunks)]

def generate_single_embedding(client, text_chunk: str):
    response = client.embeddings.create(
        model=MODEL,
        input=text_chunk
    )
    return response.data[0].embedding

def generate_embedding(client, chunks):
    """
    Generates embeddings for a list of text chunks by calling the single embedding function.
    This is used for bulk processing during ingestion.
    """
    embedded_data = []

    for chunk in chunks:
        embedding = generate_single_embedding(client, chunk['text'])
        embedded_data.append({
            "id": chunk['id'],
            "text": chunk['text'],
            "embedding": embedding
        })

    return embedded_data