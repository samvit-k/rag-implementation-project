import os
from dotenv import load_dotenv
from openai import OpenAI
from generate_embeddings import upload_documents, generate_chunks, generate_embedding
from milvus_client import get_milvus_client, create_milvus_collection_if_not_exists, insert_embeddings_into_milvus

def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please check your environment variables.")

    openai_client = OpenAI(api_key=api_key)
    milvus_client = get_milvus_client()

    create_milvus_collection_if_not_exists(milvus_client)

    documents_string = ''.join(doc['content'] for doc in upload_documents())
    chunks = generate_chunks(documents_string)

    embedded_data_list = generate_embedding(openai_client, chunks)

    insert_embeddings_into_milvus(milvus_client, embedded_data_list)

if __name__ == "__main__":
    main()