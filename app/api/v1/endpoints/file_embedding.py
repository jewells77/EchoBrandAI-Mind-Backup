from fastapi import APIRouter
from app.api.exceptions import APIError
from app.api.v1.schemas.content import FileEmbeddingRequest
from app.services.embedding_service import EmbeddingService
from app.config import settings

router = APIRouter()


@router.post(
    "/file-embedding",
    summary="Embed file from Azure Blob Storage",
    description="""Accepts user_id and a file URL in Azure Blob Storage, splits and embeds contents, saves to Qdrant under 'brand-detail' collection.
```json
{
    "user_id": "john123",
    "url": "https://example.com/file.pdf"
}
```
""",
)
async def embed_file(request: FileEmbeddingRequest):
    try:
        embedding_service = EmbeddingService(
            collection_name=settings.QDRANT_BRAND_DETAIL_COLLECTION
        )
        await embedding_service.embed_file_from_azure_blob(
            user_id=request.user_id, url=request.url
        )
        return {"message": "File embedding created successfully"}
    except APIError as e:
        raise e
    except Exception as e:
        raise APIError(
            str(e),
            status_code=500,
            public_message="Something went wrong.",
        )


@router.delete(
    "/file-embedding",
    summary="Delete file embedding from Qdrant",
    description="""
Deletes embeddings in Qdrant for a specific user ID and Azure Blob Storage file URL.
Example Parameters:
```json
{
    "user_id": "john123",
    "url": "https://example.com/file.pdf"
}
```
""",
)
async def delete_file_embedding(request: FileEmbeddingRequest):
    try:
        embedding_service = EmbeddingService(
            collection_name=settings.QDRANT_BRAND_DETAIL_COLLECTION
        )
        await embedding_service.delete_file_embedding(
            user_id=request.user_id, url=request.url
        )
        return {"message": "Embeddings deleted successfully"}
    except APIError as e:
        raise e
    except Exception as e:
        raise APIError(
            str(e),
            status_code=500,
            public_message="Something went wrong.",
        )
