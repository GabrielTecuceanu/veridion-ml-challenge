from __future__ import annotations

import logging
from typing import Any

from src.config import (
    NULL_FIELD_SCORE,
    SCORE_WEIGHT_COMPLETENESS,
    SCORE_WEIGHT_CONSTRAINT,
    SCORE_WEIGHT_KEYWORD,
    SCORE_WEIGHT_NAICS,
    SCORE_WEIGHT_VECTOR,
)
from src.models.company import CompanyMatch
from src.models.query_intent import NumericConstraint, QueryIntent
from src.utils.naics import best_naics_score

logger = logging.getLogger(__name__)


def _keyword_overlap(company: Any, keywords: list[str]) -> float:
    """Fraction of query keywords found anywhere in the company's text fields."""
    if not keywords:
        return NULL_FIELD_SCORE

    text_blob = " ".join(
        filter(
            None,
            [
                company.name or "",
                company.description or "",
                " ".join(
                    [company.primary_naics.get("label", "") if company.primary_naics else ""]
                    + [n.get("label", "") for n in (company.secondary_naics or [])]
                ),
            ],
        )
    ).lower()

    hits = sum(1 for kw in keywords if kw.lower() in text_blob)
    return hits / len(keywords)


def _satisfies_numeric(value: float | None, nc: NumericConstraint) -> bool | None:
    """True/False if value is known; None if value is missing."""
    if value is None:
        return None
    v = float(value)
    if nc.operator == "gt":
        return v > nc.value
    if nc.operator == "gte":
        return v >= nc.value
    if nc.operator == "lt":
        return v < nc.value
    if nc.operator == "lte":
        return v <= nc.value
    if nc.operator == "eq":
        return v == nc.value
    if nc.operator == "between" and nc.value2 is not None:
        return nc.value <= v <= nc.value2
    return None


def _constraint_satisfaction(company: Any, intent: QueryIntent) -> float:
    """Score how well the company satisfies hard numeric + boolean constraints.

    - Missing field -> 0.5 (uncertain, not penalised to 0)
    - Satisfied     -> 1.0
    - Violated      -> 0.0
    - No constraints -> 1.0 (fully satisfied by definition)
    """
    checks: list[float] = []

    for nc in intent.numeric_filters:
        raw_value = getattr(company, nc.field, None)
        result = _satisfies_numeric(raw_value, nc)
        if result is None:
            checks.append(NULL_FIELD_SCORE)
        else:
            checks.append(1.0 if result else 0.0)

    for field, expected in intent.boolean_filters.items():
        actual = getattr(company, field, None)
        if actual is None:
            checks.append(NULL_FIELD_SCORE)
        else:
            checks.append(1.0 if actual == expected else 0.0)

    if intent.location and intent.location.resolved_countries:
        cc = (company.country_code or "").upper()
        if not cc:
            checks.append(NULL_FIELD_SCORE)
        else:
            match = cc in [c.upper() for c in intent.location.resolved_countries]
            checks.append(1.0 if match else 0.0)

    if intent.business_model_filter and company.business_model:
        bm_set = {b.lower() for b in company.business_model}
        expected_set = {b.lower() for b in intent.business_model_filter}
        checks.append(1.0 if bm_set & expected_set else 0.0)
    elif intent.business_model_filter and not company.business_model:
        checks.append(NULL_FIELD_SCORE)

    if not checks:
        return 1.0
    return sum(checks) / len(checks)


def _naics_alignment(company: Any, query_codes: list[str]) -> float:
    if not query_codes:
        return NULL_FIELD_SCORE

    primary_code: str | None = None
    if company.primary_naics:
        primary_code = company.primary_naics.get("code")

    secondary_codes: list[str] = []
    for n in company.secondary_naics or []:
        code = n.get("code")
        if code:
            secondary_codes.append(code)

    return best_naics_score(primary_code, secondary_codes, query_codes)


def score_matches(
    matches: list[CompanyMatch],
    intent: QueryIntent,
) -> list[CompanyMatch]:
    """Stage 3: compute weighted heuristic scores for each CompanyMatch.

    Weights (from config):
        vector_similarity  0.30
        naics_alignment    0.25
        keyword_overlap    0.20
        constraint_sat     0.20
        data_completeness  0.05

    Null fields score NULL_FIELD_SCORE (0.5) - uncertain, not penalised to 0.
    """
    for match in matches:
        company = match.company
        payload: dict = company.raw or {}

        # cosine similarity is already normalized to [0,1]
        vec_sim = match.vector_similarity
        if vec_sim is None:
            vec_sim = NULL_FIELD_SCORE

        naics = _naics_alignment(company, intent.naics_codes)
        kw = _keyword_overlap(company, intent.industry_keywords)
        csat = _constraint_satisfaction(company, intent)
        completeness = payload.get("data_completeness", NULL_FIELD_SCORE)

        score = (
            SCORE_WEIGHT_VECTOR * vec_sim
            + SCORE_WEIGHT_NAICS * naics
            + SCORE_WEIGHT_KEYWORD * kw
            + SCORE_WEIGHT_CONSTRAINT * csat
            + SCORE_WEIGHT_COMPLETENESS * completeness
        )

        match.score = round(score, 4)
        match.vector_similarity = round(vec_sim, 4)
        match.naics_alignment = round(naics, 4)
        match.keyword_overlap = round(kw, 4)
        match.constraint_satisfaction = round(csat, 4)
        match.data_completeness = round(completeness, 4)

        if "stage3" not in match.qualification_path:
            match.qualification_path.append("stage3")

    matches.sort(key=lambda m: m.score, reverse=True)
    for rank, match in enumerate(matches, 1):
        match.rank = rank

    return matches
