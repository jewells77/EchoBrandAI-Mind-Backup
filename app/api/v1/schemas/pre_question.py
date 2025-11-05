# app/api/v1/schemas/pre_question.py

from pydantic import BaseModel, Field
from typing import List

class PreQuestionInput(BaseModel):
    """
    Schema for the brand details input. 
    It forces the request body to be a JSON object with a single 'brand_details' key 
    whose value is the string passed to the agent.
    """
    brand_details: str = Field(
        ...,
        description="The detailed string containing brand name, description, and industry.",
        example="Name: RetroPulse | Description: Vintage-inspired digital watch brand | Industry: Wearable Tech"
    )

class PreQuestionOutput(BaseModel):
    """
    Schema for the three generated pre-chat questions response.
    It matches the structure returned by the PreQuestionAgent (via its TypedDict).
    """
    questions: List[str] = Field(
        ...,
        description="A list containing exactly three generated pre-chat questions.",
        example=[
            "Which RetroPulse watch model is currently trending on social media?",
            "Do you have any new collections launching this holiday season?",
            "Can you suggest a short, engaging caption for an Instagram reel promoting the 'TimeWarp 2.0' model?"
        ]
    )