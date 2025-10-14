from fastapi import APIRouter, HTTPException
from app.api.exceptions import APIError
from typing import Dict, Any

from app.api.v1.schemas.chat import (
    UnifiedChatRequest,
    ChatContinueResponse,
)
from app.services.chat_service import ChatService


router = APIRouter()


@router.post(
    "/",
    response_model=ChatContinueResponse,
    summary="Unified chat endpoint",
    description="""
- If request.thread_id is provided, continue the conversation with the given message.
- Otherwise, start a new conversation
- In the message, we can include the image URL as well (Azure Blob URL)
- **Start chat:**
```json
{
  \"user_id\": \"jack123\",
  \"message\": \"Hey, can you write a 200-word Instagram post about coffee?\",
  \"brand_details\": \"Name: EcoBrand | Description: *(none)* | Industry: Personal Care\"
}
```
- **Continue chat:**
```json
{
  \"thread_id\": \"chat_123\",
  \"message\": \"Awesome! Can you make it a bit crisper?\"
}
```
""",
)
async def chat(
    request: UnifiedChatRequest,
) -> ChatContinueResponse:
    """
    Unified chat endpoint.

    - If request.thread_id is provided, continue the conversation with the given message.
    - Otherwise, start a new conversation
    """
    try:
        chat_service = ChatService()

        # Continuation flow
        if request.thread_id:
            result = await chat_service.continue_chat(
                thread_id=request.thread_id, message=request.message
            )

        # New chat flow
        else:
            result = await chat_service.start_chat(
                brand_details=request.brand_details,
                user_qurey=request.message,
                user_id=request.user_id,
            )

        if result.get("status") == "error":
            raise HTTPException(
                status_code=400, detail=result.get("error", "Unknown error")
            )

        return ChatContinueResponse(
            thread_id=result["thread_id"],
            message=result.get("message", ""),
            ai_generated_images=result.get("ai_generated_images", []),
            status=result.get("status", "completed"),
            final_output=result.get("final_output"),
        )
    except APIError:
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@router.get("/history/{thread_id}", response_model=Dict[str, Any])
async def get_chat_history(thread_id: str) -> Dict[str, Any]:
    """
    Get the conversation history for a given thread.

    This endpoint retrieves the current state and history of a conversation
    thread from MongoDB. Useful for debugging or understanding the current
    context of a conversation.
    """
    try:
        chat_service = ChatService()

        result = await chat_service.get_chat_history(thread_id)

        if result.get("status") == "not_found":
            raise HTTPException(
                status_code=404, detail=result.get("error", "Thread not found")
            )
        elif result.get("status") == "error":
            raise HTTPException(
                status_code=500, detail=result.get("error", "Unknown error")
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting chat history: {str(e)}"
        )
