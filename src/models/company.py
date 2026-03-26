from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class Company(BaseModel):
    id: str
    name: str
    description: str = ""
    country_code: Optional[str] = None
    is_public: Optional[bool] = None
    employee_count: Optional[int] = None
    revenue: Optional[float] = None
    year_founded: Optional[int] = None
    business_model: list[str] = Field(default_factory=list)
    primary_naics: Optional[dict[str, Any]] = None
    secondary_naics: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)


class CompanyMatch(BaseModel):
    company: Company
    score: float = Field(description="Heuristic weighted score from Stage 3")
    vector_similarity: Optional[float] = None
    naics_alignment: float = 0.0
    keyword_overlap: float = 0.0
    constraint_satisfaction: float = 0.0
    data_completeness: float = 0.0
    missing_data: bool = False
    qualification_path: list[str] = Field(
        default_factory=list,
        description="Stages this company passed through, e.g. ['stage1','stage2','stage3','stage4']",
    )
    matched_criteria: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    reasoning: str = ""
    rank: Optional[int] = None
    confidence: Optional[float] = None
