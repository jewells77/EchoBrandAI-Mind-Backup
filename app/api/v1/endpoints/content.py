from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any

from app.api.deps import get_llm_provider
from app.api.v1.schemas.content import (
    ContentGenerationRequest,
    ContentGenerationResponse,
)
from app.domain.graphs.content_workflow import (
    ContentCreationWorkflow,
    LangGraphContentWorkflow,
)
from app.domain.llm_providers.base import BaseLLMProvider


router = APIRouter()


@router.post("/generate", response_model=ContentGenerationResponse)
async def generate_content(
    request: ContentGenerationRequest,
    llm_provider: BaseLLMProvider = Depends(get_llm_provider),
) -> ContentGenerationResponse:
    """
    Generate content based on brand details, competitors, and content request.

    This endpoint orchestrates the content generation workflow:
    1. Brand DNA analysis
    2. Competitor intelligence
    3. Content strategy
    4. Content generation
    5. Content refinement

    Returns a complete result with all steps' outputs.
    """
    try:
        # Initialize the workflow (sequential for simplicity in the API)
        workflow = ContentCreationWorkflow(llm_provider)

        # Run the workflow
        result = await workflow.run(
            brand_details=request.brand_details.model_dump(),
            content_request=request.content_request,
            competitors_summary=request.competitors_summary,
            guidelines=request.guidelines,
            thread_id=request.thread_id,  # Pass the thread_id if provided
        )

        # Return the results
        return ContentGenerationResponse(
            brand_profile=result["brand_profile"],
            competitor_insights=result["competitor_insights"],
            content_strategy=result["content_strategy"],
            final_content=result["final_content"],
            job_id=None,  # For synchronous processing
            thread_id=result.get("thread_id"),  # Include thread_id if available
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating content: {str(e)}"
        )


@router.post("/generate/async", response_model=Dict[str, str])
async def generate_content_async(
    request: ContentGenerationRequest,
    background_tasks: BackgroundTasks,
    llm_provider: BaseLLMProvider = Depends(get_llm_provider),
) -> Dict[str, str]:
    """
    Generate content asynchronously using LangGraph workflow.

    This endpoint starts the content generation process in the background:
    1. Brand DNA analysis and Competitor intelligence (in parallel)
    2. Content strategy
    3. Content generation
    4. Content refinement

    Returns a job ID that can be used to fetch results later.
    """
    # TODO: Implement job queue and background processing

    # For now, return a placeholder job ID
    return {"job_id": "task_not_implemented_yet", "status": "queued"}


@router.post("/langgraph", response_model=ContentGenerationResponse)
async def generate_content_with_langgraph(
    request: ContentGenerationRequest,
    llm_provider: BaseLLMProvider = Depends(get_llm_provider),
) -> ContentGenerationResponse:
    """
    Generate content using the LangGraph workflow.

    This endpoint uses LangGraph for parallel processing and advanced orchestration:
    1. Brand DNA analysis and Competitor intelligence (in parallel)
    2. Content strategy
    3. Content generation
    4. Content refinement

    Returns a complete result with all steps' outputs and a thread_id for conversation continuity.
    """
    try:
        # Initialize the LangGraph workflow
        workflow = LangGraphContentWorkflow(llm_provider)

        # Run the workflow with the thread_id if provided
        result = await workflow.run(
            brand_details=request.brand_details.model_dump(),
            content_request=request.content_request,
            competitors_summary=request.competitors_summary,
            guidelines=request.guidelines,
            thread_id=request.thread_id,  # Pass the thread_id if provided
        )

        # Return the results including the thread_id for continuity
        return ContentGenerationResponse(
            brand_profile=result.get("brand_profile", {}),
            competitor_insights=result.get("competitor_insights", {}),
            content_strategy=result.get("content_strategy", {}),
            final_content=result.get("final_content", {}),
            job_id=None,  # For synchronous processing
            thread_id=result.get("thread_id"),  # Include thread_id in response
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating content with LangGraph: {str(e)}"
        )
