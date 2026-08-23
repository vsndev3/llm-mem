import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BINARY_PATH = REPO_ROOT / "target" / "release" / "llm-mem-mcp"
CONFIG_PATH = Path(os.environ.get("LLM_MEM_CONFIG_PATH", str(REPO_ROOT / "benchmark" / "config_flat.toml")))

DATA_DIR = REPO_ROOT / "benchmark-data"
LONGMEMEVAL_DIR = DATA_DIR / "longmemeval"
LOCOMO_DIR = DATA_DIR / "locomo"

DEPS_DIR = REPO_ROOT / "benchmark-deps"
LONGMEMEVAL_DEPS_DIR = DEPS_DIR / "longmemeval"
LOCOMO_DEPS_DIR = DEPS_DIR / "locomo"

OUTPUT_DIR = REPO_ROOT / "benchmark" / "output"
LOGS_DIR = REPO_ROOT / "benchmark" / "logs"

LONGMEMEVAL_S = LONGMEMEVAL_DIR / "longmemeval_s_cleaned.json"
LONGMEMEVAL_M = LONGMEMEVAL_DIR / "longmemeval_m_cleaned.json"
LONGMEMEVAL_ORACLE = LONGMEMEVAL_DIR / "longmemeval_oracle.json"

LOCOMO_10 = LOCOMO_DIR / "locomo10.json"

JUDGE_MODEL = os.environ.get("LLM_MEM_JUDGE_MODEL", "gpt-4o-mini")
JUDGE_BASE_URL = os.environ.get("LLM_MEM_JUDGE_BASE_URL", "https://api.openai.com/v1")
JUDGE_API_KEY = os.environ.get("LLM_MEM_JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

BANK_PREFIX = "bm"
QUERY_K = int(os.environ.get("LLM_MEM_QUERY_K", "10"))
SIMILARITY_THRESHOLD = float(os.environ.get("LLM_MEM_SIM_THRESHOLD", "0.25"))
