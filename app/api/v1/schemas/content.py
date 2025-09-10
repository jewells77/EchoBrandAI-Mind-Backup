from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, Any, List, Optional


class BrandDetails(BaseModel):
    """Brand details for content generation."""

    name: str = Field(..., description="Brand name")
    description: str = Field(..., description="Brand description")
    values: Optional[List[str]] = Field(default=[], description="Brand values")
    industry: str = Field(..., description="Industry the brand operates in")
    mission_statement: Optional[str] = Field(
        None, description="Brand mission statement"
    )
    additional_info: Optional[Dict[str, Any]] = Field(
        None, description="Additional brand information"
    )
