import os 
from dotenv import load_dotenv
from openai import OpenAI
from milvus_client import get_milvus_client, milvus_similarity_search
from generate_embeddings import generate_single_embedding

load_dotenv()

CHAT_MODEL = "gpt-3.5-turbo"
COLLECTION_NAME = "rag_collection"
DIMENSION = 1536
MODEL = "text-embedding-3-small"
TOP_K = 3

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file. Please check your environment variables.")

    openai_client = OpenAI(api_key=api_key)
    milvus_client = get_milvus_client()

    while True: 
        try: 
            question = input("Enter a prompt based on the provided documents: ").strip()
            if question.lower() in ['exit', 'quit']:
                break

            query_embedding = generate_single_embedding(openai_client, question)   

            search_results =  milvus_similarity_search(milvus_client, query_embedding)
            retrieved_chunks = [result['entity']['text'] for result in search_results[0]]

            context = "\n".join(retrieved_chunks)

            prompt = f"""
            You are a helpful assistant. Please answer the user's question based on the
            following context. If the context does not contain the answer, say so.

            Context:
            {context}

            User's Question: {question}

            Answer:
            """

            response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            answer = response.choices[0].message.content

            print(answer)

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()