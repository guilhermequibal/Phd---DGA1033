# =============================================================
# CELL 1: CONFIGURATION (PDFs)
#
# STRATEGY:
# This cell centralizes every experiment parameter in a single location
# and exposes each one as an environment variable override. This design
# ensures full reproducibility: any parameter that can influence results
# (model, temperature, chunk size, number of runs) has exactly one value
# that propagates through all subsequent cells. Controlled variation is
# achieved by changing environment variables at launch time, not by
# editing the code.
#
# Resource caps (CPU threads, GPU) are set here, before any ML library
# is imported, because libraries such as PyTorch and OpenBLAS read these
# variables at import time. Setting them later has no effect.
#
# Chunk size is derived from the embedding model's token limit rather
# than an arbitrary character count. MiniLM-L6 silently truncates inputs
# that exceed 256 tokens; oversized chunks are partly discarded at
# embedding time, degrading retrieval quality without any visible error.
# =============================================================
import os

# --- Resource limits ---
# Caps CPU threads to keep the shared server usable for other users.
# Must be applied before any ML import to take effect.
N_THREADS = int(os.environ.get("EXPERIMENT_N_THREADS", "8"))
THREAD_ENV_VARS = [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
]
for v in THREAD_ENV_VARS:
    os.environ[v] = str(N_THREADS)

import torch
torch.set_num_threads(N_THREADS)

# --- GPU isolation ---
# One GPU per user; violations cause immediate process termination on this server.
# Default to GPU 2: on Bool servers, GPUs 0-1 are commonly occupied by other users.
# Override at launch time with EXPERIMENT_GPU_DEVICE=<id> if the available GPU differs.
GPU_DEVICE = os.environ.get("EXPERIMENT_GPU_DEVICE", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

# Single source of truth for the environment verification performed in Cell 2.5.
REQUIRED_ENV = {v: str(N_THREADS) for v in THREAD_ENV_VARS}
REQUIRED_ENV["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE

# --- Paths ---
# Anchored to this file's own directory so the code works regardless of the
# current working directory (notebooks frequently change CWD at runtime).
try:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_DIR = os.path.abspath(os.environ.get("PROJECT_DIR", "."))

ENVISION_INDEX_PATH = os.path.join(PROJECT_DIR, "envision_index.json")
SYSTEM_PROMPT_PATH = os.path.join(PROJECT_DIR, "rag_system_prompt.txt")
USER_GUIDE_PDF_PATH = os.path.join(
    PROJECT_DIR, "ISI Envision Manual_v3_EN_bookmarked-amendments.pdf"
)

# List of project PDF files to evaluate
PDF_FILES = [
    os.path.join(PROJECT_DIR, "sample_project.pdf"),
]

# --- Model settings ---
MODEL_NAME = os.environ.get("EXPERIMENT_MODEL", "llama3.3:70b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
TEMPERATURE = float(os.environ.get("EXPERIMENT_TEMPERATURE", "0.1"))
# Context window sized so that all layers of llama3.3:70b Q4_K_M fit on a single
# 48 GB A6000. With 32K context the KV cache is ~10 GB; model weights are ~42 GB;
# total ~52 GB leaves only ~4–5 GB for CPU offload (~9 out of 80 layers), giving
# ~5–10 t/s. At 128K the KV cache alone is ~41 GB, pushing ~66 layers to CPU and
# reducing throughput to ~0.46 t/s — making multi-hour calls infeasible on one GPU.
CONTEXT_WINDOW = 32000
# 4000 output tokens gives ~267 tokens per credit for a 15-credit category,
# enough for structured JSON with applicability, score, points, and justification.
MAX_OUTPUT_TOKENS = 4000
# At least 5 runs are recommended to support statistical claims about the results.
NUM_RUNS = int(os.environ.get("EXPERIMENT_NUM_RUNS", "3"))
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "7200"))  # seconds
OLLAMA_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "2"))

# --- RAG settings ---
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
# Chunk size is derived from the embedding model's token limit to guarantee that
# every chunk fits within the model's context window and is fully embedded.
EMBEDDING_MAX_TOKENS = int(os.environ.get("EMBEDDING_MAX_TOKENS", "256"))
CHARS_PER_TOKEN = 4  # rough English heuristic used only to size chunks
CHUNK_SIZE = int(EMBEDDING_MAX_TOKENS * CHARS_PER_TOKEN * 0.85)  # ~870 chars, safe margin
CHUNK_OVERLAP = 150
CHROMA_DB_PATH = os.path.join(PROJECT_DIR, "chroma_db")
RETRIEVAL_TOP_K = 15
# "auto" rebuilds the index only when source files or parameters change,
# avoiding redundant re-embedding on repeated runs.
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
