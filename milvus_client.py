from pymilvus import MilvusClient

COLLECTION_NAME = "rag_collection"
DIMENSION = 1536
TOP_K = 3

def get_milvus_client():
    """Initializes and returns a MilvusClient instance."""
    client = MilvusClient(
        uri="http://localhost:19530"
    )
    return client

def create_milvus_collection_if_not_exists(client: MilvusClient):
    if not client.has_collection(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=DIMENSION,
            primary_field_name="id",
            vector_field_name="embedding",
            text_field_name="text",
            auto_id=True,
            metric_type="L2"
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

def insert_embeddings_into_milvus(client: MilvusClient, embedded_data_list):
    data_to_insert = [
        {"embedding": item["embedding"], "text": item["text"]}
        for item in embedded_data_list
    ]

    result = client.insert(
        collection_name=COLLECTION_NAME,
        data=data_to_insert
    )
    return result

def milvus_similarity_search(client, data):
    res = client.search(
                collection_name=COLLECTION_NAME,
                data=[data],
                limit=TOP_K,
                output_fields=["text"]
            ) 
    return res