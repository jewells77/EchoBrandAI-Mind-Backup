from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_scraper
from app.api.v1.schemas.competitor import (
    CompetitorScrapeRequest,
    DeleteWebsiteEmbeddingsRequest,
)
from app.services.competitor_service import CompetitorService
from app.infrastructure.scraping.playwright_client import PlaywrightScraper
from app.infrastructure.vectorstores.qdrant_store import delete_points_by_filter
from app.api.exceptions import APIError
from app.config import settings


router = APIRouter()


@router.post("/website-embeddings")
async def get_competitor_insights(
    request: CompetitorScrapeRequest,
    scraper: PlaywrightScraper = Depends(get_scraper),
):
    """
    Scrape and analyze a single competitor website to extract insights.

    This endpoint can be used to:
    1. Gather competitive intelligence separate from content generation
    2. Pre-scrape competitor data to use in future content generation requests

    Returns a simple success or raises an HTTPException on error.
    """
    try:
        # Initialize the competitor service
        competitor_service = CompetitorService(scraper)

        # Get competitor insights
        await competitor_service.process_competitor_website(
            url=str(request.url), user_id=request.user_id
        )
        return {
            "status": "success",
            "message": "Competitor website processed successfully.",
        }

    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing competitors: {str(e)}"
        )


@router.delete("/website-embeddings")
async def delete_qdrant_embeddings(request: DeleteWebsiteEmbeddingsRequest):
    try:
        # Example Parameters
        # {
        #     "user_id": "john123",
        #     "filter_dict": {
        #         "should": [
        #             {"key": "url", "match": {"value": "https://silverlifegym.in/"}},
        #             {"key": "url", "match": {"value": "https://silverlifegym.in"}},
        #         ]
        #     },
        # }
        collection_name = settings.QDRANT_WEBSITE_CONTENT_COLLECTION
        # Always enforce user_id as a must condition
        must_conditions = [{"key": "user_id", "match": {"value": request.user_id}}]
        if "must" in request.filter_dict:
            must_conditions.extend(request.filter_dict["must"])

        filter_with_user = {"must": must_conditions}
        if "should" in request.filter_dict:
            filter_with_user["should"] = request.filter_dict["should"]
        if "must_not" in request.filter_dict:
            filter_with_user["must_not"] = request.filter_dict["must_not"]

        result = delete_points_by_filter(collection_name, filter_with_user)
        return {"status": "success", "result": str(result)}
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting Qdrant embeddings: {str(e)}"
        )
