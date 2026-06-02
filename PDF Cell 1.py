# =============================================================
# CELL 1: CONFIGURATION (PDFs)
# =============================================================

import os

# --- Paths ---
PROJECT_DIR = '.' # A relatively safe route!
ENVISION_INDEX_PATH = os.path.join(PROJECT_DIR, 'envision_index.json')
SYSTEM_PROMPT_PATH = os.path.join(PROJECT_DIR, 'rag_system_prompt.txt')

# List of your PDF files (rename them if necessary)
PDF_FILES = [ 
    os.path.join(PROJECT_DIR, 'sample_project.pdf'),
]

# --- Model settings ---
MODEL_NAME = 'llama3.3:70b'
OLLAMA_URL = 'http://localhost:11434'
TEMPERATURE = 0.1 
CONTEXT_WINDOW = 128000 
MAX_OUTPUT_TOKENS = 8192 
NUM_RUNS = 3 # Testing three times for reliability

# --- RAG settings ---
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHROMA_DB_PATH = os.path.join(PROJECT_DIR, 'chroma_db')
RETRIEVAL_TOP_K = 15

# --- Thread limits (server security) ---
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_MAX_THREADS'] = '4'

# --- Results directory ---
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print('✅ Configuration loaded for the PDF experiment.')
print(f'   Model: {MODEL_NAME}')
print(f'   PDF files: {len(PDF_FILES)}')