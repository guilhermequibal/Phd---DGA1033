# =============================================================
# CELL 2: HELPER FUNCTIONS (PDFs)
# =============================================================

import json, time, datetime, traceback, hashlib
import requests
import pandas as pd
import fitz  # PyMuPDF para ler PDFs

# ---------- PDF EXTRACTION ----------
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

# ---------- OLLAMA API ----------
def call_ollama(system_prompt, user_message, timeout=7200): # Timeout, safe!
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