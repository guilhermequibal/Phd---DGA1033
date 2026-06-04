# =============================================================
# CELL 2: HELPER FUNCTIONS (PDFs)
# =============================================================

import json, time, datetime, traceback, hashlib
import requests
import pandas as pd
import fitz  # PyMuPDF para ler PDFs

# ---------- PDF EXTRACTION (project document) ----------
def extract_pdf_metadata(filepath):
    """Extract all text from a PDF file using PyMuPDF."""
    text_content = ""
    try:
        doc = fitz.open(filepath)
        for page in doc:
            text_content += page.get_text("text") + "\n"
        doc.close()
    except Exception as e:
        print(f"Erro ao ler o PDF {filepath}: {e}")

    return {
        'source_file': os.path.basename(filepath),
        'extracted_text': text_content
    }

# ---------- PDF CHUNKING (for RAG indexing) ----------
def chunk_pdf_for_rag(filepath, chunk_size=1500, overlap=200):
    """Extract and chunk a PDF into overlapping text segments for RAG indexing.

    Returns a list of dicts with keys: text, chunk_idx, source_file, page_approx.
    chunk_size and overlap are measured in characters.
    """
    chunks = []
    try:
        doc = fitz.open(filepath)
        # Build a single text string and track where each page starts
        full_text = ""
        page_offsets = [0]  # page_offsets[i] = char offset of page i start
        for page in doc:
            full_text += page.get_text("text") + "\n"
            page_offsets.append(len(full_text))
        doc.close()

        start = 0
        chunk_idx = 0
        while start < len(full_text):
            end = min(start + chunk_size, len(full_text))
            chunk_text = full_text[start:end].strip()
            if chunk_text:
                # Approximate page: last page that started at or before `start`
                page_num = next(
                    (i for i in range(len(page_offsets) - 1, -1, -1)
                     if page_offsets[i] <= start),
                    0
                ) + 1  # 1-indexed
                chunks.append({
                    'text': chunk_text,
                    'chunk_idx': chunk_idx,
                    'source_file': os.path.basename(filepath),
                    'page_approx': page_num,
                })
                chunk_idx += 1
            start += chunk_size - overlap

    except Exception as e:
        print(f"Error chunking PDF {filepath}: {e}")

    return chunks

# ---------- OLLAMA API ----------
def call_ollama(system_prompt, user_message, timeout=7200):  # Timeout, safe!
    """Sends the prompt to the Ollama API and returns the response."""
    start = time.time()
    try:
        r = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json={
                'model': MODEL_NAME,
                'system': system_prompt,
                'prompt': user_message,
                'stream': False,
                'options': {
                    'num_ctx': CONTEXT_WINDOW,
                    'temperature': TEMPERATURE,
                    'num_predict': MAX_OUTPUT_TOKENS,
                }
            },
            timeout=timeout
        )
        elapsed = time.time() - start
        result = r.json()
        return {
            'response': result.get('response', ''),
            'elapsed_seconds': round(elapsed, 2),
            'eval_count': result.get('eval_count', 0),
            'prompt_eval_count': result.get('prompt_eval_count', 0),
            'success': True
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'response': f'ERROR: {str(e)}',
            'elapsed_seconds': round(elapsed, 2),
            'eval_count': 0,
            'prompt_eval_count': 0,
            'success': False
        }

# ---------- ENVISION DATA LOADER ----------
def load_envision_by_category():
    """Loads the Envision JSON data, sorted by category."""
    with open(ENVISION_INDEX_PATH) as f:
        data = json.load(f)
    categories = {}
    for cid, credit in data['credits'].items():
        cat = credit['sheet']
        if cat not in categories:
            categories[cat] = {}
        categories[cat][cid] = credit
    return categories, data['metadata']

# ---------- RESULT SAVING ----------
def save_result(result_dict, filepath):
    """Save the dictionaries as JSON files."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f'   Salvo: {filepath}')

def timestamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

print('✅ Helper functions (PyMuPDF & Ollama) loaded successfully.')
