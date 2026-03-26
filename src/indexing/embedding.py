from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from src.models.company import Company

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def build_embedding_text(company: Company, raw: dict[str, Any] | None = None) -> str:
    """Build a labeled-prefix text string for embedding.

    Uses labeled field prefixes ("Industry: …", "Markets served: …") rather
    than raw concatenation so the embedding model can distinguish field types.
    """
    parts: list[str] = []

    if company.name:
        parts.append(f"Company: {company.name}")

    if company.primary_naics:
        label = company.primary_naics.get("label", "")
        code = company.primary_naics.get("code", "")
        if label:
            parts.append(f"Industry: {label} ({code})")

    if company.secondary_naics:
        sec_labels = [
            n.get("label", "") for n in company.secondary_naics if n.get("label")
        ]
        if sec_labels:
            parts.append(f"Secondary industries: {', '.join(sec_labels)}")

    if company.description:
        parts.append(f"Description: {company.description}")

    # target_markets and core_offerings live in the raw record
    source = raw or (company.raw if company.raw else {})

    target_markets: list[str] = source.get("target_markets") or []
    if target_markets:
        parts.append(f"Markets served: {', '.join(target_markets)}")

    core_offerings: list[str] = source.get("core_offerings") or []
    if core_offerings:
        parts.append(f"Core offerings: {'; '.join(core_offerings[:8])}")  # cap at 8

    if company.business_model:
        parts.append(f"Business model: {', '.join(company.business_model)}")

    if company.country_code:
        parts.append(f"Country: {company.country_code}")

    return "\n".join(parts)


def encode_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of texts into embeddings using the configured model.

    Returns a 2-D float32 array of shape (len(texts), VECTOR_SIZE).
    """
    model = get_model()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine similarity = dot product after L2-norm
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def encode_companies(companies: list[Company]) -> np.ndarray:
    """Convenience wrapper: build embedding texts then encode all companies."""
    texts = [build_embedding_text(c, c.raw) for c in companies]
    return encode_texts(texts)
