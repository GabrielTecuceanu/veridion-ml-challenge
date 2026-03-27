from __future__ import annotations

import os

QDRANT_URL: str = os.getenv("QDRANT_URL", ":memory:")
QDRANT_COLLECTION: str = "companies"
QDRANT_VECTOR_SIZE: int = 1024  # intfloat/multilingual-e5-large output dim
QDRANT_DISTANCE: str = "COSINE"
QDRANT_HNSW_EF: int = 128  # ef_construct; ef at query time defaults to same

EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"
EMBEDDING_BATCH_SIZE: int = 32

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = "qwen3:8b"
OLLAMA_TEMPERATURE: float = 0.0
OLLAMA_MAX_RETRIES: int = 3
OLLAMA_TIMEOUT: float = 60.0

REVENUE_SANITY_CAP: float = 10_000_000_000_000.0  # $10 trillion

# Stage 3 scoring weights (must sum to 1.0)
SCORE_WEIGHT_VECTOR: float = 0.30
SCORE_WEIGHT_NAICS: float = 0.25
SCORE_WEIGHT_KEYWORD: float = 0.20
SCORE_WEIGHT_CONSTRAINT: float = 0.20
SCORE_WEIGHT_COMPLETENESS: float = 0.05

NAICS_SCORE_EXACT: float = 1.0
NAICS_SCORE_4DIGIT: float = 0.7
NAICS_SCORE_2DIGIT: float = 0.3
NAICS_SCORE_NONE: float = 0.0

NULL_FIELD_SCORE: float = 0.5  # used when a field is missing rather than 0

VECTOR_SEARCH_TOP_K: int = 50

JUDGE_BATCH_SIZE: int = 5
JUDGE_TYPE_B_TOP_K: int = 10   # top-N from the ambiguous score zone
JUDGE_TYPE_C_TOP_K: int = 25
JUDGE_AMBIGUOUS_LOW: float = 0.40
JUDGE_AMBIGUOUS_HIGH: float = 0.70

REWRITE_NUM_VARIANTS: int = 4

# Business model aliases (shorthand -> canonical)
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

INDEXER_BATCH_UPSERT: int = 64
