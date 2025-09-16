from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, Optional

from app.api.v1.schemas.content import BrandDetails


class UnifiedChatRequest(BaseModel):
    """Unified request model for starting or continuing a chat conversation.

    - If thread_id is provided → continuation; other details may be omitted.
    - If thread_id is not provided → starting a new chat; brand details, competitors summary,
      guidelines, and message are required.
    """

    thread_id: Optional[str] = Field(
        None, description="Thread ID to continue an existing conversation"
    )
    message: str = Field(..., description="User message or initial brief")
    brand_details: Optional[BrandDetails] = Field(
        None, description="Details about the brand (required if starting a new chat)"
    )
    competitors_summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed summary of competitors (required if starting a new chat)",
    )
    guidelines: Optional[Dict[str, Any]] = Field(
        None,
        description="Content guidelines including tone, audience, etc. (required if starting a new chat)",
    )

    @model_validator(mode="after")
    def validate_required_fields(
        cls, values: "UnifiedChatRequest"
    ) -> "UnifiedChatRequest":
        # Continuation flow
        if values.thread_id:
            return values

        # New chat flow → require brand_details, competitors_summary, guidelines
        missing_fields = []
        if values.brand_details is None:
            missing_fields.append("brand_details")
        if values.competitors_summary is None:
            missing_fields.append("competitors_summary")
        if values.guidelines is None:
            missing_fields.append("guidelines")
        if missing_fields:
            raise ValueError(
                f"When thread_id is not provided, {', '.join(missing_fields)} must be provided"
            )
        return values


class ChatInitResponse(BaseModel):
    """Response model for chat initialization."""

    thread_id: str = Field(..., description="Thread ID for continuing the conversation")

    content_strategy: Dict[str, Any] = Field(
        ..., description="Content strategy recommendations"
    )
    final_output: str = Field(
        ..., description="Final output including title and content"
    )


class ChatContinueResponse(BaseModel):
    """Response model for continuing a chat conversation."""

    thread_id: str = Field(..., description="Thread ID for the conversation")
    message: Optional[str] = Field(None, description="Conversational response message")
    status: str = Field(..., description="Status of the conversation")
    final_output: str = Field(
        ..., description="Final output including title and content"
    )
