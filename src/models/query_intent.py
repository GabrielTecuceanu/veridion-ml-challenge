from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class LocationConstraint(BaseModel):
    raw: str = Field(description="Original location string from the query")
    resolved_countries: list[str] = Field(
        default_factory=list,
        description="ISO-2 country codes resolved by the LLM (e.g. 'Scandinavia' -> ['SE','NO','DK','FI'])",
    )


class NumericConstraint(BaseModel):
    field: Literal["employee_count", "revenue", "year_founded"]
    operator: Literal["gt", "gte", "lt", "lte", "eq", "between"]
    value: float
    value2: Optional[float] = Field(
        default=None, description="Upper bound for 'between' operator"
    )


class QueryIntent(BaseModel):
    query_type: Literal["structured", "hybrid", "reasoning"] = Field(
        description="Type A=structured, Type B=hybrid, Type C=reasoning/supply-chain"
    )
    industry_keywords: list[str] = Field(
        default_factory=list,
        description="Industry/domain keywords for keyword overlap scoring",
    )
    naics_codes: list[str] = Field(
        default_factory=list,
        description="LLM-predicted NAICS codes (6-digit preferred)",
    )
    location: Optional[LocationConstraint] = None
    numeric_filters: list[NumericConstraint] = Field(default_factory=list)
    boolean_filters: dict[str, bool] = Field(
        default_factory=dict,
        description="e.g. {'is_public': True}",
    )
    business_model_filter: list[str] = Field(
        default_factory=list,
        description="Full strings: 'Business-to-Business', 'Business-to-Consumer', etc.",
    )
    semantic_criteria: str = Field(
        default="",
        description="Free-text criteria that cannot be structured; forwarded to the LLM Judge",
    )
    ecosystem_role: str = Field(
        default="",
        description="Role in supply chain (e.g. 'supplier', 'manufacturer'); set for Type C queries",
    )
    target_beneficiary: str = Field(
        default="",
        description="Who the supplier serves (e.g. 'cosmetics brands'); set for Type C queries",
    )

    @model_validator(mode="after")
    def enforce_hybrid_when_semantic(self) -> "QueryIntent":
        """Guard: if semantic_criteria is non-empty, force query_type >= 'hybrid'."""
        if self.semantic_criteria and self.query_type == "structured":
            self.query_type = "hybrid"
        return self
