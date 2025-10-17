from pydantic import BaseModel, Field
from typing import Dict


class FinalizedPostRequest(BaseModel):
    """Request model for extracting finalized posts by thread_id."""

    thread_id: str = Field(..., description="Thread ID of the conversation workflow")


class FinalizedPostResponse(BaseModel):
    """Response model for finalized multi-platform post extraction."""

    platforms: Dict[str, str] = Field(
        ..., description="Map of platform name to finalized post content"
    )


class FileEmbeddingRequest(BaseModel):
    """Request model for file embedding from Azure Blob Storage."""

    user_id: str = Field(..., description="User identifier.")
    url: str = Field(..., description="URL of the file in Azure Blob Storage.")
