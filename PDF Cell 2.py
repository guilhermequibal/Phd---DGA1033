# =============================================================
# CELL 2: HELPER FUNCTIONS (PDFs)
# =============================================================
import os
import json, time, datetime, re
import requests
import pandas as pd
import fitz  # PyMuPDF para ler PDFs


# ---------- JSON VALIDATION ----------
def _looks_like_json(text):
    """Best-effort check that the model output contains a parseable JSON object.

    FIX: lets callers know whether the model actually returned usable JSON.
    HTTP success != valid output.
    """
    if not text:
        return False
    s = text.strip()
    # strip markdown code fences if present
    s = re.sub(r"```(?:json)?", "", s).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        return False
    try:
        json.loads(s[start:end + 1])
        return True
    except json.JSONDecodeError:
        return False


# ---------- PDF EXTRACTION (project document) ----------
def extract_pdf_metadata(filepath):
    """Extract all text from a PDF file using PyMuPDF.

    FIX: returns explicit diagnostic fields (extraction_ok, n_pages, n_chars,
    error) instead of silently swallowing exceptions and returning empty text.
    Also flags empty extractions, which usually mean a scanned PDF that needs OCR.
    """
    text_content = ""
    n_pages = 0
    error = None
    try:
        doc = fitz.open(filepath)
        n_pages = doc.page_count
        parts = [page.get_text("text") for page in doc]
        text_content = "\n".join(parts)
        doc.close()
    except (fitz.FileDataError, FileNotFoundError, RuntimeError) as e:
        error = f"{type(e).__name__}: {e}"
        print(f"❌ Erro ao ler o PDF {filepath}: {error}")

    stripped = text_content.strip()
    extraction_ok = error is None and len(stripped) > 0
    if error is None and not stripped:
        print(f"⚠️  PDF {os.path.basename(filepath)} produced no text "
              f"(scanned document? OCR may be required).")

    return {
        "source_file": os.path.basename(filepath),
        "extracted_text": text_content,
        "n_pages": n_pages,
        "n_chars": len(stripped),
        "extraction_ok": extraction_ok,
        "error": error,
    }


# ---------- PDF CHUNKING (for RAG indexing) ----------
def _estimate_tokens(text, chars_per_token=4):
    return max(1, len(text) // chars_per_token)


def _split_sentences(text):
    """Lightweight sentence splitter (keeps it dependency-free)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def chunk_pdf_for_rag(filepath, chunk_size=850, overlap=150, chars_per_token=4):
    """Extract and chunk a PDF into overlapping segments for RAG indexing.

    FIX: the old version sliced blindly every `chunk_size` characters, which
    (a) cut words/sentences in half and (b) produced chunks LARGER than the
    embedding model's token window, so part of every chunk was truncated and
    never embedded. This version:
      * packs whole sentences into each chunk,
      * keeps each chunk under `chunk_size` chars (sized in Cell 1 to fit the
        embedding token limit, so nothing is silently truncated),
      * carries a sentence-level overlap for context continuity,
      * tracks the approximate source page for citations.
    Returns dicts with: text, chunk_idx, source_file, page_approx, n_tokens_est.
    """
    chunks = []
    try:
        doc = fitz.open(filepath)
        page_texts = []
        page_offsets = [0]  # page_offsets[i] = char offset where page i starts
        for page in doc:
            t = page.get_text("text") + "\n"
            page_texts.append(t)
            page_offsets.append(page_offsets[-1] + len(t))
        doc.close()
        full_text = "".join(page_texts)
    except (fitz.FileDataError, FileNotFoundError, RuntimeError) as e:
        print(f"❌ Error chunking PDF {filepath}: {type(e).__name__}: {e}")
        return chunks

    def page_of(char_pos):
        # last page that started at or before char_pos (1-indexed)
        for i in range(len(page_offsets) - 1, -1, -1):
            if page_offsets[i] <= char_pos:
                return i + 1
        return 1

    # Build sentence units, each carrying its character position in full_text.
    sentences = []
    cursor = 0
    for line in full_text.split("\n"):
        line = line.strip()
        if not line:
            cursor += 1
            continue
        start = full_text.find(line, cursor)
        if start == -1:
            start = cursor
        for sent in _split_sentences(line):
            sentences.append((sent, start))
        cursor = start + len(line)

    chunk_idx = 0
    i = 0
    while i < len(sentences):
        buf, buf_len = [], 0
        chunk_start_pos = sentences[i][1]
        j = i
        while j < len(sentences) and buf_len + len(sentences[j][0]) + 1 <= chunk_size:
            buf.append(sentences[j][0])
            buf_len += len(sentences[j][0]) + 1
            j += 1
        if not buf:  # a single oversized sentence: hard-split it once
            buf = [sentences[i][0][:chunk_size]]
            j = i + 1
        chunk_text = " ".join(buf).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "chunk_idx": chunk_idx,
                "source_file": os.path.basename(filepath),
                "page_approx": page_of(chunk_start_pos),
                "n_tokens_est": _estimate_tokens(chunk_text, chars_per_token),
            })
            chunk_idx += 1
        # step back a few sentences to create the overlap
        overlap_chars, step_back, k = 0, 0, j - 1
        while k > i and overlap_chars < overlap:
            overlap_chars += len(sentences[k][0])
            step_back += 1
            k -= 1
        i = max(i + 1, j - step_back)  # always make progress (guaranteed terminate)

    return chunks


# ---------- OLLAMA API ----------
def call_ollama(system_prompt, user_message, timeout=None, max_retries=None):
    """Send the prompt to the Ollama API and return the response + metrics.

    FIX: configurable (and shorter) timeout, automatic retries with exponential
    backoff on transient failures, narrower exception handling, and a
    `valid_json` flag so the caller knows whether the output is parseable.
    """
    timeout = OLLAMA_TIMEOUT if timeout is None else timeout
    max_retries = OLLAMA_MAX_RETRIES if max_retries is None else max_retries

    start = time.time()
    last_err = None
    for attempt in range(1, max_retries + 2):  # initial attempt + retries
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "system": system_prompt,
                    "prompt": user_message,
                    "stream": False,
                    "options": {
                        "num_ctx": CONTEXT_WINDOW,
                        "temperature": TEMPERATURE,
                        "num_predict": MAX_OUTPUT_TOKENS,
                    },
                },
                timeout=timeout,
            )
            r.raise_for_status()
            result = r.json()
            text = result.get("response", "")
            return {
                "response": text,
                "valid_json": _looks_like_json(text),
                "elapsed_seconds": round(time.time() - start, 2),
                "eval_count": result.get("eval_count", 0),
                "prompt_eval_count": result.get("prompt_eval_count", 0),
                "attempts": attempt,
                "success": True,
            }
        except (requests.RequestException, ValueError) as e:
            last_err = e
            if attempt <= max_retries:
                backoff = 2 ** attempt
                print(f"   ⚠️  Ollama call failed ({e}); "
                      f"retry {attempt}/{max_retries} in {backoff}s")
                time.sleep(backoff)

    return {
        "response": f"ERROR: {last_err}",
        "valid_json": False,
        "elapsed_seconds": round(time.time() - start, 2),
        "eval_count": 0,
        "prompt_eval_count": 0,
        "attempts": max_retries + 1,
        "success": False,
    }


# ---------- ENVISION DATA LOADER ----------
def load_envision_by_category():
    """Load the Envision JSON data, grouped by category."""
    with open(ENVISION_INDEX_PATH) as f:
        data = json.load(f)
    categories = {}
    for cid, credit in data["credits"].items():
        cat = credit["sheet"]
        categories.setdefault(cat, {})[cid] = credit
    return categories, data["metadata"]


# ---------- RESULT SAVING ----------
def save_result(result_dict, filepath):
    """Save a dict as a JSON file (creates parent dirs)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f"   Salvo: {filepath}")


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


print("✅ Helper functions (PyMuPDF & Ollama) loaded successfully.")
