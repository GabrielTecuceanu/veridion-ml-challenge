from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.config import (
    JUDGE_AMBIGUOUS_HIGH,
    JUDGE_AMBIGUOUS_LOW,
    JUDGE_BATCH_SIZE,
    JUDGE_TYPE_B_TOP_K,
    JUDGE_TYPE_C_TOP_K,
)
from src.models.company import CompanyMatch
from src.models.judge_verdict import JudgeVerdict
from src.models.query_intent import QueryIntent
from src.utils.llm_client import structured_completion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured response the LLM must return for each batch
# ---------------------------------------------------------------------------

class _BatchVerdict(BaseModel):
    verdicts: list[JudgeVerdict] = Field(
        description="One verdict per company in the batch, in the same order"
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert company qualification analyst.
For each company profile provided, decide whether it qualifies for the
given search query.

Rules:
- Base your verdict ONLY on the information in the company profile.
- Do NOT invent facts that are not in the profile.
- matched_criteria and failed_criteria must reference actual fields or
  statements from the profile.
- confidence: 0.0 = no chance, 1.0 = absolute certainty.
- reasoning: 1–2 sentences explaining the key deciding factor.

Calibration examples:
[TRUE POSITIVE] Query: "B2B SaaS companies in Germany with >100 employees"
  Profile: "Acme GmbH, Berlin, 250 employees, Business-to-Business, Software-as-a-Service"
  → qualified=true, confidence=0.95
  → matched: ["country=DE", "business_model=Business-to-Business", "employee_count=250>100", "SaaS"]

[TRUE NEGATIVE] Query: "B2B SaaS companies in Germany with >100 employees"
  Profile: "Beta Corp, Paris, 300 employees, Business-to-Consumer, e-commerce platform"
  → qualified=false, confidence=0.90
  → failed: ["country=FR≠DE", "business_model=B2C≠B2B", "not SaaS"]

[BORDERLINE] Query: "B2B SaaS companies in Germany with >100 employees"
  Profile: "Gamma AG, Munich, employee_count=unknown, cloud software, B2B"
  → qualified=true, confidence=0.55
  → matched: ["country=DE", "business_model=B2B", "cloud software≈SaaS"]
  → failed: ["employee_count unknown"]
"""


def _format_company(match: CompanyMatch) -> str:
    """Render a company profile as a compact text block for the LLM prompt."""
    c = match.company
    payload: dict[str, Any] = c.raw or {}

    lines: list[str] = [f"### Company ID: {c.id}"]
    lines.append(f"Name: {c.name}")
    if c.description:
        lines.append(f"Description: {c.description[:400]}")
    if c.country_code:
        lines.append(f"Country: {c.country_code}")
    if c.employee_count is not None:
        lines.append(f"Employees: {c.employee_count}")
    if c.revenue is not None:
        lines.append(f"Revenue: ${c.revenue:,.0f}")
    if c.year_founded is not None:
        lines.append(f"Founded: {c.year_founded}")
    if c.is_public is not None:
        lines.append(f"Public: {c.is_public}")
    if c.business_model:
        lines.append(f"Business model: {', '.join(c.business_model)}")
    if c.primary_naics:
        label = c.primary_naics.get("label", "")
        code = c.primary_naics.get("code", "")
        lines.append(f"Primary industry: {label} ({code})")
    target_markets = payload.get("target_markets") or []
    if target_markets:
        lines.append(f"Markets: {', '.join(target_markets[:5])}")
    core_offerings = payload.get("core_offerings") or []
    if core_offerings:
        lines.append(f"Offerings: {'; '.join(core_offerings[:4])}")

    return "\n".join(lines)


def _build_batch_prompt(
    query: str,
    semantic_criteria: str,
    batch: list[CompanyMatch],
) -> str:
    query_section = f"Search query: {query}"
    if semantic_criteria:
        query_section += f"\nAdditional criteria: {semantic_criteria}"

    profiles = "\n\n".join(_format_company(m) for m in batch)

    return (
        f"{query_section}\n\n"
        f"Evaluate the following {len(batch)} company profiles:\n\n"
        f"{profiles}\n\n"
        f"Return a verdict for each company in order."
    )


# ---------------------------------------------------------------------------
# Post-validation
# ---------------------------------------------------------------------------

def _validate_verdict(verdict: JudgeVerdict, match: CompanyMatch) -> JudgeVerdict:
    """Ensure matched_criteria refer to actual profile content.

    This is a lightweight sanity check: if matched_criteria names a field
    value that doesn't appear anywhere in the profile text, drop that criterion.
    We don't hard-fail — the verdict stands but criteria are cleaned up.
    """
    payload: dict[str, Any] = match.company.raw or {}
    profile_blob = " ".join(
        str(v) for v in payload.values() if v is not None
    ).lower()

    cleaned_matched: list[str] = []
    for criterion in verdict.matched_criteria:
        # Very loose check: at least one word from the criterion appears in profile
        words = criterion.lower().replace("=", " ").split()
        if any(w in profile_blob for w in words if len(w) > 3):
            cleaned_matched.append(criterion)
        else:
            logger.debug(
                "Dropped unverifiable criterion %r for company %s",
                criterion,
                match.company.id,
            )

    verdict.matched_criteria = cleaned_matched
    return verdict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _run_batch(
    query: str,
    semantic_criteria: str,
    batch: list[CompanyMatch],
) -> list[JudgeVerdict]:
    """Call the LLM Judge on a single batch and return verdicts."""
    prompt = _build_batch_prompt(query, semantic_criteria, batch)
    response = structured_completion(
        prompt=prompt,
        response_model=_BatchVerdict,
        system=_SYSTEM_PROMPT,
        temperature=0.0,
    )

    verdicts = response.verdicts
    # Align verdict count with batch size
    if len(verdicts) < len(batch):
        logger.warning(
            "Judge returned %d verdicts for batch of %d — padding with disqualified",
            len(verdicts),
            len(batch),
        )
        for i in range(len(verdicts), len(batch)):
            verdicts.append(
                JudgeVerdict(
                    company_id=batch[i].company.id,
                    qualified=False,
                    confidence=0.0,
                    reasoning="No verdict returned by judge.",
                )
            )

    return [_validate_verdict(v, m) for v, m in zip(verdicts, batch)]


def run_judge(
    matches: list[CompanyMatch],
    intent: QueryIntent,
    raw_query: str,
    query_type: str,
) -> list[CompanyMatch]:
    """Stage 4: run LLM Judge on selected matches and apply verdicts.

    Selection logic:
    - Type B: top JUDGE_TYPE_B_TOP_K in the ambiguous 0.4–0.7 score zone
    - Type C: top JUDGE_TYPE_C_TOP_K regardless of score

    Verdicts update the CompanyMatch in-place. Companies the Judge disqualifies
    get their score zeroed and are moved to the bottom of the returned list.
    """
    if query_type == "structured":
        return matches  # Type A skips Judge entirely

    # Select candidates for judging
    if query_type == "hybrid":
        candidates = [
            m for m in matches
            if JUDGE_AMBIGUOUS_LOW <= m.score <= JUDGE_AMBIGUOUS_HIGH
        ][:JUDGE_TYPE_B_TOP_K]
    else:  # reasoning / Type C
        candidates = matches[:JUDGE_TYPE_C_TOP_K]

    if not candidates:
        logger.info("No candidates selected for Judge — skipping")
        return matches

    logger.info(
        "Running Judge on %d candidates (%s mode)", len(candidates), query_type
    )

    # Build an id → match index for fast lookup
    id_to_match: dict[str, CompanyMatch] = {m.company.id: m for m in matches}

    # Process in batches
    semantic = intent.semantic_criteria or ""
    for i in range(0, len(candidates), JUDGE_BATCH_SIZE):
        batch = candidates[i : i + JUDGE_BATCH_SIZE]
        verdicts = _run_batch(raw_query, semantic, batch)

        for verdict in verdicts:
            match = id_to_match.get(verdict.company_id)
            if match is None:
                continue
            match.confidence = verdict.confidence
            match.matched_criteria = verdict.matched_criteria
            match.failed_criteria = verdict.failed_criteria
            match.reasoning = verdict.reasoning
            if "stage4" not in match.qualification_path:
                match.qualification_path.append("stage4")

            if not verdict.qualified:
                match.score = 0.0

    # Re-sort so disqualified companies sink to the bottom
    matches.sort(key=lambda m: (m.score, m.confidence or 0.0), reverse=True)
    for rank, match in enumerate(matches, 1):
        match.rank = rank

    return matches
