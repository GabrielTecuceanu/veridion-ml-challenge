from __future__ import annotations

# ---------------------------------------------------------------------------
# Qdrant settings
# ---------------------------------------------------------------------------
import os

QDRANT_URL: str = os.getenv("QDRANT_URL", ":memory:")
QDRANT_COLLECTION: str = "companies"
QDRANT_VECTOR_SIZE: int = 1024  # BAAI/bge-m3 output dimension
QDRANT_DISTANCE: str = "Cosine"
QDRANT_HNSW_EF: int = 128       # ef_construct; ef at query time defaults to same

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE: int = 32

# ---------------------------------------------------------------------------
# Ollama / LLM settings
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = "qwen3:8b"
OLLAMA_TEMPERATURE: float = 0.0
OLLAMA_MAX_RETRIES: int = 3
OLLAMA_TIMEOUT: float = 60.0

# ---------------------------------------------------------------------------
# Revenue sanity cap (Stage 0 normalisation)
# ---------------------------------------------------------------------------
REVENUE_SANITY_CAP: float = 10_000_000_000_000.0  # $10 trillion

# ---------------------------------------------------------------------------
# Stage 3 — heuristic scoring weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
SCORE_WEIGHT_VECTOR: float = 0.30
SCORE_WEIGHT_NAICS: float = 0.25
SCORE_WEIGHT_KEYWORD: float = 0.20
SCORE_WEIGHT_CONSTRAINT: float = 0.20
SCORE_WEIGHT_COMPLETENESS: float = 0.05

# NAICS hierarchy partial-match bonuses
NAICS_SCORE_EXACT: float = 1.0
NAICS_SCORE_4DIGIT: float = 0.7
NAICS_SCORE_2DIGIT: float = 0.3
NAICS_SCORE_NONE: float = 0.0

# Null-field uncertainty score (used instead of 0 when data is missing)
NULL_FIELD_SCORE: float = 0.5

# ---------------------------------------------------------------------------
# Stage 2 — vector search
# ---------------------------------------------------------------------------
VECTOR_SEARCH_TOP_K: int = 50

# ---------------------------------------------------------------------------
# Stage 4 — LLM Judge
# ---------------------------------------------------------------------------
JUDGE_BATCH_SIZE: int = 5
JUDGE_TYPE_B_TOP_K: int = 10          # send top-N in ambiguous zone
JUDGE_TYPE_C_TOP_K: int = 25
JUDGE_AMBIGUOUS_LOW: float = 0.40     # borderline zone for Type B
JUDGE_AMBIGUOUS_HIGH: float = 0.70

# ---------------------------------------------------------------------------
# Stage 0.5 — query rewriting (Type C)
# ---------------------------------------------------------------------------
REWRITE_NUM_VARIANTS: int = 4         # how many supplier-perspective queries

# ---------------------------------------------------------------------------
# Business model aliases  (shorthand → canonical full string)
# ---------------------------------------------------------------------------
BUSINESS_MODEL_ALIASES: dict[str, str] = {
    "b2b": "Business-to-Business",
    "b2c": "Business-to-Consumer",
    "b2g": "Business-to-Government",
    "saas": "Software-as-a-Service",
    "marketplace": "Marketplace",
    "wholesale": "Wholesale",
    "retail": "Retail",
    "enterprise": "Enterprise",
    "manufacturing": "Manufacturing",
    "service provider": "Service Provider",
    "service": "Service Provider",
    "d2c": "Business-to-Consumer",
}

# ---------------------------------------------------------------------------
# Indexer settings
# ---------------------------------------------------------------------------
INDEXER_BATCH_UPSERT: int = 64        # companies per Qdrant upsert call
