from pydantic import BaseModel, Field, model_validator, ValidationInfo
from typing import List, Optional


class UnifiedChatRequest(BaseModel):
    """Unified request model for starting or continuing a chat conversation.

    Conditional requirements:
    - If thread_id is provided, only thread_id and message are required.
    - If thread_id is not provided, user_id, message, and brand_details are required.
    """

    thread_id: Optional[str] = Field(
        None, description="Thread ID to continue an existing conversation"
    )
    user_id: Optional[str] = Field(
        None, description="User ID for validation (required for new chat)"
    )
    message: str = Field(..., description="User message or initial brief")
    brand_details: Optional[str] = Field(
        None, description="Details about the brand (required if starting a new chat)"
    )

    @model_validator(mode="after")
    def validate_required_fields(self, info: ValidationInfo) -> "UnifiedChatRequest":
        if self.thread_id:
            # In continuation, only message is required (already enforced by type), thread_id must be present
            if not self.message:
                raise ValueError(
                    "Message is required when continuing a chat (thread_id provided)"
                )
            return self
        # New chat flow: thread_id not provided
        missing = []
        if not self.user_id:
            missing.append("user_id")
        if not self.message:
            missing.append("message")
        if not self.brand_details:
            missing.append("brand_details")
        if missing:
            raise ValueError(
                f"When thread_id is not provided, the following fields must be provided: {', '.join(missing)}"
            )
        return self


class ChatContinueResponse(BaseModel):
    """Response model for continuing a chat conversation."""

    thread_id: str = Field(..., description="Thread ID for the conversation")
    message: Optional[str] = Field(None, description="Conversational response message")
    ai_generated_images: Optional[List[str]] = Field(
        None, description="List of AI-generated image URLs"
    )
    status: str = Field(..., description="Status of the conversation")
    final_output: str = Field(
        ..., description="Final output including title and content"
    )
