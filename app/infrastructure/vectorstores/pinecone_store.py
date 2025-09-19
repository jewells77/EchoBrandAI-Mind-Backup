import os
import uuid
from tqdm.auto import tqdm
from app.infrastructure.vectorstores.pinecone_config import index


def upsert_embeddings(embeddings: list, metadata_list: list, namespace: str = ""):
    """
    Upsert multiple chunk embeddings into Pinecone with dynamic metadata and optional namespace.
    """
    vector_ids = [str(uuid.uuid4()) for _ in embeddings]
    vectors = [
        (vector_id, embedding, metadata)
        for vector_id, embedding, metadata in zip(vector_ids, embeddings, metadata_list)
    ]
    upsert_args = {"vectors": vectors}
    if namespace:
        upsert_args["namespace"] = namespace
    with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
        index.upsert(**upsert_args)
        progress.update(len(embeddings))


def query_embeddings(
    vector: list,
    top_k: int = 3,
    namespace: str = "",
    filter: dict = None,
    include_metadata: bool = True,
    include_values: bool = False,
):
    """
    Query Pinecone for similar vectors with advanced options.
    """
    query_args = {
        "vector": vector.tolist() if hasattr(vector, "tolist") else vector,
        "top_k": top_k,
        "include_metadata": include_metadata,
        "include_values": include_values,
    }
    if namespace:
        query_args["namespace"] = namespace
    if filter:
        query_args["filter"] = filter
    # Pinecone's query does not support 'fields' directly, but you can filter metadata after retrieval
    return index.query(**query_args)
