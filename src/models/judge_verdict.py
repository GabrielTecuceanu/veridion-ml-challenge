from __future__ import annotations

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    company_id: str
    qualified: bool
    confidence: float = Field(ge=0.0, le=1.0)
    matched_criteria: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    reasoning: str = ""
