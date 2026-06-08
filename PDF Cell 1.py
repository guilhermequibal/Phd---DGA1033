# =============================================================
# CELL 1: CONFIGURATION (PDFs)
# =============================================================
import os

# --- Resource limits (set BEFORE any ML import to take effect) ---
# Caps CPU threads so the server stays usable for other users.
# FIX: configurable via environment with a safe default, instead of a hardcoded
#      magic string. Override with EXPERIMENT_N_THREADS=4 python ...
N_THREADS = int(os.environ.get("EXPERIMENT_N_THREADS", "8"))
THREAD_ENV_VARS = [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
]
for v in THREAD_ENV_VARS:
    os.environ[v] = str(N_THREADS)

import torch
torch.set_num_threads(N_THREADS)

# --- GPU isolation (one GPU per user — violations cause instant kill) ---
# FIX: configurable, still defaults to GPU 0.
GPU_DEVICE = os.environ.get("EXPERIMENT_GPU_DEVICE", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

# FIX: single source of truth for the env-var verification done in Cell 2.5,
#      so the two cells can never drift apart.
REQUIRED_ENV = {v: str(N_THREADS) for v in THREAD_ENV_VARS}
REQUIRED_ENV["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

# --- Paths ---
# FIX: anchor paths to THIS file's directory so the code works regardless of the
#      current working directory (notebooks frequently change CWD). Falls back to
#      CWD/PROJECT_DIR when __file__ is undefined (e.g. interactive paste).
try:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_DIR = os.path.abspath(os.environ.get("PROJECT_DIR", "."))

ENVISION_INDEX_PATH = os.path.join(PROJECT_DIR, "envision_index.json")
SYSTEM_PROMPT_PATH = os.path.join(PROJECT_DIR, "rag_system_prompt.txt")
USER_GUIDE_PDF_PATH = os.path.join(
    PROJECT_DIR, "ISI Envision Manual_v3_EN_bookmarked-amendments.pdf"
)

# List of your PDF files (rename them if necessary)
PDF_FILES = [
    os.path.join(PROJECT_DIR, "sample_project.pdf"),
]

# --- Model settings ---
MODEL_NAME = os.environ.get("EXPERIMENT_MODEL", "llama3.3:70b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
TEMPERATURE = float(os.environ.get("EXPERIMENT_TEMPERATURE", "0.1"))
CONTEXT_WINDOW = 128000
MAX_OUTPUT_TOKENS = 8192
# FIX: NUM_RUNS configurable; >=5 is recommended if you want statistical claims.
NUM_RUNS = int(os.environ.get("EXPERIMENT_NUM_RUNS", "3"))
# FIX: timeout shortened from 7200s (2h) and made configurable; retries added.
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "3600"))  # seconds
OLLAMA_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "2"))

# --- RAG settings ---
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# FIX: make the embedding context budget explicit so the chunker can size chunks
#      to FIT it. MiniLM-L6 silently truncates inputs at 256 tokens; the old
#      1500-char chunks were partly lost at embedding time.
EMBEDDING_MAX_TOKENS = int(os.environ.get("EMBEDDING_MAX_TOKENS", "256"))
CHARS_PER_TOKEN = 4  # rough English heuristic used only to size chunks
CHUNK_SIZE = int(EMBEDDING_MAX_TOKENS * CHARS_PER_TOKEN * 0.85)  # ~870 chars, safe margin
CHUNK_OVERLAP = 150
CHROMA_DB_PATH = os.path.join(PROJECT_DIR, "chroma_db")
RETRIEVAL_TOP_K = 15
# FIX: allow reusing an existing index instead of always re-embedding everything.
#      "auto" rebuilds only when sources/params changed; "always"/"never" force it.
REBUILD_INDEX = os.environ.get("REBUILD_INDEX", "auto")

# --- Results directory ---
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("✅ Configuration loaded for the PDF experiment.")
print(f"   Model: {MODEL_NAME}")
print(f"   PDF files: {len(PDF_FILES)}")
print(f"   User guide: {os.path.basename(USER_GUIDE_PDF_PATH)}")
print(f"   CPU threads: {N_THREADS} | GPU: CUDA_VISIBLE_DEVICES={GPU_DEVICE}")
print(f"   Chunk size: {CHUNK_SIZE} chars (~{EMBEDDING_MAX_TOKENS} tok budget), "
      f"overlap {CHUNK_OVERLAP}")
print(f"   Project dir: {PROJECT_DIR}")
