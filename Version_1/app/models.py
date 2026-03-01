"""
Pydantic request / response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List


class AnalyzeRequest(BaseModel):
    """Incoming request — a natural-language business-practice description."""
    prompt: str = Field(..., min_length=1, description="Business practice to analyze")


class AnalyzeResponse(BaseModel):
    """Strictly-formatted response expected by the test harness."""
    harmful: bool = Field(..., description="True if the practice violates the CCPA")
    articles: List[str] = Field(
        default_factory=list,
        description="List of violated CCPA section identifiers (empty if not harmful)",
    )


class HealthResponse(BaseModel):
    status: str = "ok"
