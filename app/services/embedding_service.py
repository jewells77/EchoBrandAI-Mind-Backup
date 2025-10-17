from app.api.exceptions import APIError
from app.domain.tools.text_chunker import chunk_text
from app.domain.llm_providers.embedding_factory import get_embedding_provider
from app.infrastructure.vectorstores.qdrant_store import QdrantStore
from qdrant_client.models import PointStruct
import uuid
from azure.storage.blob import BlobClient
import os
from tempfile import NamedTemporaryFile
from app.config import get_embedding_config, settings
from urllib.parse import urlparse
from app.domain.file_parsers.factory import get_parser_for_file


class EmbeddingService:
    def __init__(self, embedding_provider_name: str = None, collection_name: str = ""):
        provider = get_embedding_config(provider=embedding_provider_name)[0]
        self.embedding_provider = get_embedding_provider(provider)
        self.collection_name = collection_name

    async def process_and_upsert(
        self,
        text: str = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        metadata: dict = None,
        chunks: list[str] = None,
    ):
        """
        Processes text (splitting into chunks if needed), generates embeddings, and upserts into Qdrant.
        If 'chunks' is provided, uses them directly; otherwise, chunks the 'text' argument.
        Args:
            text: Full text to split into chunks (ignored if chunks is given)
            chunk_size: Size for splitting text
            chunk_overlap: Overlap for chunking
            metadata: Metadata to associate with each chunk
            chunks: Precomputed list of text chunks (optional)
        Returns:
            dict with generated chunks, embeddings, and used metadata
        """
        metadata = metadata or {}
        qdrant_store = QdrantStore()
        # Validate what each chunk's payload will look like
        await qdrant_store.validate_payload_fields(
            self.collection_name, {**metadata, "text": "just for validation"}
        )
        if chunks is not None:
            chunks_to_use = chunks
        else:
            if text is None or text == "":
                raise APIError(
                    "Either 'text' or 'chunks' must be provided.", status_code=400
                )
            chunks_to_use = chunk_text(
                text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        metadata_list = [{**metadata, "text": chunk} for chunk in chunks_to_use]
        embeddings = self.embedding_provider.embed_documents(chunks_to_use)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=metadata,
            )
            for (embedding, metadata) in zip(embeddings, metadata_list)
        ]
        await qdrant_store.upsert_points(self.collection_name, points)
        return {
            "chunks": chunks_to_use,
            "embeddings": embeddings,
            "metadata": metadata_list,
        }

    def extract_container_and_blob(self, url: str):
        parsed = urlparse(url)
        path = parsed.path.lstrip("/").split("/", 1)
        container = path[0]
        blob_name = path[1] if len(path) > 1 else ""
        return container, blob_name

    async def embed_file_from_azure_blob(self, user_id: str, url: str):
        """
        Loads a document from Azure Blob Storage, splits/embeds, and upserts to Qdrant with appropriate payload.
        """
        temp_path = None
        try:
            # TODO: ".txt", ".docx" will be added later
            # Check allowed extensions
            allowed_exts = (".pdf",)  # Extendable list
            if not any(url.lower().endswith(ext) for ext in allowed_exts):
                raise APIError(
                    f"File is not a supported type. Supported: {allowed_exts}",
                    status_code=400,
                )
            # Check if the user already has an embedding for this source_url
            qdrant_store = QdrantStore()
            filter_dict = {
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                    {"key": "source_url", "match": {"value": url}},
                ]
            }
            existing = await qdrant_store.get_items_by_filter(
                self.collection_name, filter_dict=filter_dict, limit=1
            )
            if existing:
                raise APIError(
                    "Embeddings for this file and user already exist.", status_code=409
                )
            _container, blob_name = self.extract_container_and_blob(url)
            blob_client = BlobClient.from_connection_string(
                conn_str=settings.AZURE_BLOB_CONNECTION_STRING,
                container_name=settings.AZURE_BLOB_CONTAINER,
                blob_name=blob_name,
            )
            suffix = os.path.splitext(blob_name)[1]
            with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                download_stream = blob_client.download_blob()
                temp_file.write(download_stream.readall())
                temp_path = temp_file.name
            parser = get_parser_for_file(temp_path)
            chunks_text = parser.parse(temp_path)
            payload = {
                "user_id": user_id,
                "source_url": url,
                "file_type": suffix.lower().lstrip("."),
            }
            await self.process_and_upsert(
                metadata=payload,
                chunks=chunks_text,
            )
            return
        except APIError as e:
            raise e
        except Exception as e:
            raise APIError(f"{e}", status_code=500)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    async def delete_file_embedding(self, user_id: str, url: str):
        """
        Deletes embeddings from Qdrant for a specific user and file URL.
        """
        try:
            qdrant_store = QdrantStore()
            filter_dict = {
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                    {"key": "source_url", "match": {"value": url}},
                ]
            }
            result = await qdrant_store.delete_points_by_filter(
                self.collection_name, filter_dict
            )
            # Use attribute access based on Qdrant UpdateResult
            if hasattr(result, "status") and result.status != "completed":
                raise APIError("Failed to delete embeddings.", status_code=500)
            return
        except APIError as e:
            raise e
        except Exception as e:
            raise APIError(f"{e}", status_code=500)
