from pydantic import BaseModel, Field, HttpUrl


class CompetitorScrapeRequest(BaseModel):
    """Request model for scraping competitor data."""

    url: HttpUrl = Field(..., description="Single competitor URL to scrape")
    user_id: str = Field(..., description="user identifier for namespacing embeddings")


class DeleteEmbeddingsRequest(BaseModel):
    """
    Request model for deleting embeddings from Pinecone.
    Example:
        metadata_filter = {"url": {"$eq": "https://silverlifegym.in/"}}
    """

    metadata_filter: dict = Field(
        ...,
        description="Metadata filter for deletion, e.g., {'url': {'$eq': 'https://silverlifegym.in/'}}",
    )
    namespace: str = Field(..., description="User ID or namespace for deletion scope.")
