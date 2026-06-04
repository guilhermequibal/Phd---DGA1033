# =============================================================
# CELL 1: CONFIGURATION (PDFs)
# =============================================================

import os

# --- Resource limits (set BEFORE any ML import to take effect) ---
# Caps CPU threads so the server stays usable for other users.
N = "8"
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[v] = N

import torch
torch.set_num_threads(8)

# --- GPU isolation (one GPU per user — violations cause instant kill) ---
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# --- Paths ---
PROJECT_DIR = '.'  # A relatively safe route!
ENVISION_INDEX_PATH = os.path.join(PROJECT_DIR, 'envision_index.json')
SYSTEM_PROMPT_PATH  = os.path.join(PROJECT_DIR, 'rag_system_prompt.txt')
USER_GUIDE_PDF_PATH = os.path.join(
    PROJECT_DIR, 'ISI Envision Manual_v3_EN_bookmarked-amendments.pdf'
)

# List of your PDF files (rename them if necessary)
PDF_FILES = [
    os.path.join(PROJECT_DIR, 'sample_project.pdf'),
]

# --- Model settings ---
MODEL_NAME        = 'llama3.3:70b'
OLLAMA_URL        = 'http://localhost:11434'
TEMPERATURE       = 0.1
CONTEXT_WINDOW    = 128000
MAX_OUTPUT_TOKENS = 8192
NUM_RUNS          = 3  # Testing three times for reliability

# --- RAG settings ---
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHROMA_DB_PATH  = os.path.join(PROJECT_DIR, 'chroma_db')
RETRIEVAL_TOP_K = 15

# --- Results directory ---
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print('✅ Configuration loaded for the PDF experiment.')
print(f'   Model:      {MODEL_NAME}')
print(f'   PDF files:  {len(PDF_FILES)}')
print(f'   User guide: {os.path.basename(USER_GUIDE_PDF_PATH)}')
print(f'   CPU threads: {N} | GPU: CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}')
