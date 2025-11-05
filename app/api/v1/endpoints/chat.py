from fastapi import APIRouter, status
from app.api.exceptions import APIError
from typing import Dict, Any

from app.api.v1.schemas.chat import (
    UnifiedChatRequest,
    ChatContinueResponse,
)
from app.services.chat_service import ChatService

from app.api.v1.schemas.pre_question import PreQuestionInput, PreQuestionOutput

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
            raise APIError(result.get("error", "Unknown error"), status_code=400)

        return ChatContinueResponse(
            thread_id=result["thread_id"],
            message=result.get("message", ""),
            ai_generated_images=result.get("ai_generated_images", []),
            status=result.get("status", "completed"),
            final_output=result.get("final_output"),
        )
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Error processing chat: {str(e)}", status_code=500)


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
            raise APIError(result.get("error", "Thread not found"), status_code=404)
        elif result.get("status") == "error":
            raise APIError(result.get("error", "Unknown error"), status_code=500)

        return result

    except APIError:
        raise

    except Exception as e:
        raise APIError(f"Error getting chat history: {str(e)}", status_code=500)


@router.post(
    "/pre-questions",
    response_model=PreQuestionOutput,
    status_code=status.HTTP_200_OK,
    summary="To generate 3 personalized pre-chat questions based on brand details.",
    description="""
Accepts a JSON object with the 'brand_details' key and generates three creator-style questions.
To get questions(example body):
```json
{
    \"brand_details\": \"Name: EcoBrand | Description: *(none)* | Industry: Personal Care\"
}
```
""",
)
async def generate_pre_questions(
    # FastAPI handles the validation of the JSON body against this schema
    input_data: PreQuestionInput,
    
) -> PreQuestionOutput:
    
    try:
        # Instantiate the main chat service
        service = ChatService()
        
        # Call the new method on the chat service
        questions = await service.generate_questions(brand_details=input_data.brand_details)
        
        return questions
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Error generating pre-questions: {str(e)}", status_code=500)