from app.domain.tools.text_chunker import chunk_text
from app.domain.llm_providers.embedding_factory import get_embedding_provider
from app.infrastructure.vectorstores.pinecone_store import upsert_embeddings


class EmbeddingService:
    def __init__(
        self, embedding_provider_name: str = "huggingface", namespace: str = ""
    ):
        self.embedding_provider = get_embedding_provider(embedding_provider_name)
        self.namespace = namespace

    def process_and_upsert(
        self, text: str, url: str, chunk_size: int = 800, chunk_overlap: int = 100
    ):
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = [self.embedding_provider.get_embedding(chunk) for chunk in chunks]
        metadata_list = [{"url": url, "text": chunk} for chunk in chunks]
        upsert_embeddings(embeddings, metadata_list, namespace=self.namespace)
        return {"chunks": chunks, "embeddings": embeddings, "metadata": metadata_list}
