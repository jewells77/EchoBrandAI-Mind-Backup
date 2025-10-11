from app.domain.tools.text_chunker import chunk_text
from app.domain.llm_providers.embedding_factory import get_embedding_provider
from app.infrastructure.vectorstores.qdrant_store import upsert_points
from qdrant_client.models import PointStruct
from app.infrastructure.vectorstores.qdrant_store import validate_payload_fields
import uuid


class EmbeddingService:
    def __init__(
        self, embedding_provider_name: str = "huggingface", collection_name: str = ""
    ):
        self.embedding_provider = get_embedding_provider(embedding_provider_name)
        self.collection_name = collection_name

    async def process_and_upsert(
        self,
        text: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        metadata: dict = None,
    ):
        metadata = metadata or {}
        # Validate what each chunk's payload will look like
        validate_payload_fields(
            self.collection_name, {**metadata, "text": "just for validation"}
        )
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        metadata_list = [{**metadata, "text": chunk} for chunk in chunks]
        embeddings = [self.embedding_provider.get_embedding(chunk) for chunk in chunks]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=metadata,
            )
            for (embedding, metadata) in zip(embeddings, metadata_list)
        ]
        await upsert_points(self.collection_name, points)
        return {"chunks": chunks, "embeddings": embeddings, "metadata": metadata_list}
