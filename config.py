import os
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
VECTORS_DIR = DATA_DIR / "vectors"
DB_PATH = DATA_DIR / "embeddings.db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTORS_DIR.mkdir(exist_ok=True)

# Jina API — multi-vector (ColBERT)
JINA_API_KEY_PATH = PROJECT_DIR / "apiKey.txt"
JINA_API_URL = "https://api.jina.ai/v1/multi-vector"
JINA_MODEL = "jina-colbert-v2"
JINA_DIMENSIONS = 128

# Jina API — single-vector
JINA_SINGLE_VECTOR_URL = "https://api.jina.ai/v1/embeddings"
JINA_SINGLE_VECTOR_MODEL = "jina-embeddings-v3"
JINA_SINGLE_VECTOR_DIMENSIONS = 1024

# Claude CLI (for LLM calls via subprocess)
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", "claude")

# Search
DEFAULT_TOP_K = 5

# Search routing thresholds
SEARCH_SCORE_THRESHOLD_LOW = 0.30       # absolute floor — ignore below this
SEARCH_SCORE_RELATIVE_CUTOFF = 0.70     # must score >= 70% of the top score
SEARCH_MAX_CANDIDATES_PER_LEVEL = 3
SEARCH_LLM_AMBIGUITY_RANGE = 0.10      # if top scores within this range, ask LLM

# Ingestion dedup
DEDUP_SIMILARITY_THRESHOLD = 0.92

# Web/RSS ingestion
RSS_FETCH_DELAY = 1.0
RSS_REQUEST_TIMEOUT = 15
RSS_MAX_ARTICLES_PER_FEED = 20
RSS_MIN_ARTICLE_WORDS = 100


def get_jina_api_key() -> str:
    """Load Jina API key from file."""
    key_path = JINA_API_KEY_PATH
    if not key_path.exists():
        raise FileNotFoundError(f"API key file not found: {key_path}")
    return key_path.read_text().strip()
