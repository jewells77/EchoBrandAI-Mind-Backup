from fastapi import APIRouter, HTTPException
from app.api.v1.schemas.content import FinalizedPostRequest, FinalizedPostResponse
from app.services.finalized_post_service import FinalizedPostService

router = APIRouter()


@router.post("/finalized-post", response_model=FinalizedPostResponse)
async def get_finalized_post(request: FinalizedPostRequest) -> FinalizedPostResponse:
    """
    Extract finalized social media posts for a given thread_id using FinalizedPostExtractor agent.
    """
    try:
        service = FinalizedPostService()
        result = await service.get_finalized_post(request.thread_id)
        return FinalizedPostResponse(platforms=result["platforms"])
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error extracting finalized post: {str(e)}"
        )
