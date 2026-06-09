# =============================================================
# CELL 4: RUNNER (PDFs)
#
# STRATEGY:
# This cell runs the core comparative experiment: Zero-Shot vs. RAG.
# Both arms evaluate each PDF against the full set of Envision credits,
# but differ in what reference material the model receives:
#
#   Arm A — Zero-Shot: the model receives the complete Envision rubric
#     for a category plus the extracted PDF text. No retrieval step.
#     This is the baseline, representing how well the model performs
#     with only the structured workbook data and no external guidance.
#
#   Arm B — RAG: a short, focused retrieval query (category name +
#     brief project summary) fetches the top-K most relevant passages
#     from the vector index, which contains both workbook entries and
#     guidance manual chunks. Those passages plus the PDF text are then
#     passed to generation through the same call_ollama() function used
#     by Arm A.
#
# Using the same generation path for both arms is a deliberate design
# choice: it ensures that latency, token counts, and output validity
# are directly comparable between the two arms. The only controlled
# variable is whether retrieval-augmented context is provided — exactly
# the effect this study aims to measure.
#
# The retrieval query intentionally omits the full PDF text. Embedding
# models like MiniLM-L6 truncate inputs at 256 tokens; a long document
# used as a query would be heavily truncated and degrade retrieval
# quality. A short, semantically focused query produces better recall.
# =============================================================
print("=" * 70)
print(" ENVISION PDF EXPERIMENT RUNNER")
print(f" Started: {datetime.datetime.now()}")
print(f" Model: {MODEL_NAME}")
print(f" Files: {len(PDF_FILES)} PDF file(s)")
print(f" Runs: {NUM_RUNS} per experiment")
print("=" * 70)

# Load the base prompt
with open(SYSTEM_PROMPT_PATH) as f:
    system_prompt = f.read()

# Load the Envision categories
categories, meta = load_envision_by_category()
category_names = list(categories.keys())

# Initialize the retriever once and reuse it across all files and runs.
retriever = rag_index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)

timing_log = []

for file_idx, pdf_path in enumerate(PDF_FILES):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    # Prefix with index so two PDFs sharing a basename cannot produce colliding output paths.
    file_label = f"{file_idx:02d}_{base}"
    print(f'\n{"=" * 70}')
    print(f"FILE {file_idx + 1}/{len(PDF_FILES)}: {file_label}")
    print(f'{"=" * 70}')

    # --- PDF Extraction ---
    print("\n Extracting text from a PDF...")
    pdf_data = extract_pdf_metadata(pdf_path)
    if not pdf_data["extraction_ok"]:
        print(f" ❌ Skipping {file_label}: extraction failed "
              f"(error={pdf_data['error']}, chars={pdf_data['n_chars']}).")
        continue
    print(f" ✅ Text extracted ({pdf_data['n_pages']} pages, {pdf_data['n_chars']} chars).")

    # Cap extracted_text at 18 000 characters (~4 500 tokens) when building the
    # LLM prompt. The largest category JSON is ~13 K tokens and the system prompt
    # ~7 K tokens, so the total input already reaches ~24 K tokens before the PDF
    # text is added. With CONTEXT_WINDOW=32 000 and MAX_OUTPUT_TOKENS=4 000 this
    # leaves ~4 000 tokens for the PDF snippet, comfortably within budget.
    # The original pdf_data dict (with full text) is saved to disk unchanged.
    PDF_TEXT_BUDGET = 18000
    pdf_data_for_llm = {**pdf_data}
    if len(pdf_data_for_llm.get("extracted_text", "")) > PDF_TEXT_BUDGET:
        pdf_data_for_llm["extracted_text"] = pdf_data_for_llm["extracted_text"][:PDF_TEXT_BUDGET]
    pdf_json = json.dumps(pdf_data_for_llm, indent=2, default=str)

    # Short project summary used only to steer the retrieval query.
    # Kept under the embedding token window so it is not truncated.
    project_summary = pdf_data["extracted_text"].strip().replace("\n", " ")[:800]

    save_result(pdf_data, os.path.join(RESULTS_DIR, file_label, "pdf_metadata.json"))

    # =====================================================
    # EXPERIMENT A: ZERO-SHOT
    # =====================================================
    for run_num in range(1, NUM_RUNS + 1):
        run_dir = os.path.join(RESULTS_DIR, file_label, "zero_shot", f"run_{run_num}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n --- ZERO-SHOT | Run {run_num}/{NUM_RUNS} ---")
        run_start = time.time()
        all_category_results = {}

        for cat_name, cat_credits in categories.items():
            print(f"   Category: {cat_name} ({len(cat_credits)} credits)...", end=" ")
            cat_json = json.dumps(cat_credits, indent=2, default=str)
            user_msg = f"""## Envision Reference Data — {cat_name} Category
{cat_json}

## Project PDF Metadata
{pdf_json}

Task: Evaluate this project against all {cat_name} credits shown above. For each \
credit, determine applicability, evaluate questions against the PDF metadata, \
determine the achievable level, calculate points, and identify documentation \
gaps. Return your assessment as JSON following the credit_assessments format in \
your system instructions."""

            result = call_ollama(system_prompt, user_msg)
            cat_safe = cat_name.replace(" ", "_").lower()
            save_result(result, os.path.join(run_dir, f"{cat_safe}.json"))
            all_category_results[cat_name] = result
            status = "✅" if result["success"] else "❌"
            json_flag = "" if result["valid_json"] else " ⚠️ non-JSON"
            print(f'{status} ({result["elapsed_seconds"]}s{json_flag})')

        run_elapsed = round(time.time() - run_start, 2)
        summary = {
            "experiment": "zero_shot", "file": file_label, "run": run_num,
            "model": MODEL_NAME, "temperature": TEMPERATURE,
            "total_elapsed_seconds": run_elapsed,
            "categories": {
                k: {
                    "elapsed_seconds": v["elapsed_seconds"],
                    "output_tokens": v["eval_count"],
                    "input_tokens": v["prompt_eval_count"],
                    "valid_json": v["valid_json"],
                    "success": v["success"],
                } for k, v in all_category_results.items()
            },
            "timestamp": timestamp(),
        }
        save_result(summary, os.path.join(run_dir, "_summary.json"))
        timing_log.append({"experiment": "zero_shot", "file": file_label,
                           "run": run_num, "seconds": run_elapsed})

    # =====================================================
    # EXPERIMENT B: RAG
    # =====================================================
    for run_num in range(1, NUM_RUNS + 1):
        run_dir = os.path.join(RESULTS_DIR, file_label, "rag", f"run_{run_num}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n --- RAG | Run {run_num}/{NUM_RUNS} ---")
        run_start = time.time()
        all_category_results = {}

        for cat_name in category_names:
            n_credits = len(categories[cat_name])
            print(f"   Category: {cat_name} ({n_credits} credits)...", end=" ")

            # Step 1: Retrieve with a short, focused query rather than the full PDF.
            # A concise query targeting the category and project context yields
            # better semantic recall from the embedding model.
            retrieval_query = (
                f"Envision {cat_name} category: credit applicability, evaluation "
                f"criteria, scoring levels and points. Project context: {project_summary}"
            )
            retr_start = time.time()
            try:
                nodes = retriever.retrieve(retrieval_query)
                retr_elapsed = round(time.time() - retr_start, 2)

                sources, context_blocks = [], []
                for node in nodes:
                    st = node.metadata.get("source_type", "unknown")
                    if st == "envision_workbook":
                        sid = node.metadata.get("credit_id", "?")
                        loc = node.metadata.get("category", "?")
                    else:  # guidance_manual chunk
                        sid = f"p{node.metadata.get('page_approx', '?')}"
                        loc = node.metadata.get("source_file", "?")
                    sources.append({
                        "source_type": st, "source_id": sid, "location": loc,
                        "score": round(node.score, 4) if node.score is not None else None,
                    })
                    context_blocks.append(f"[{st}:{sid}]\n{node.get_content()}")

                retrieved_context = "\n\n".join(context_blocks)

                # Step 2: Generate through the same call_ollama() path as zero-shot,
                # so token counts and latency metrics are directly comparable between arms.
                user_msg = f"""## Retrieved Envision Reference (top-{RETRIEVAL_TOP_K})
{retrieved_context}

## Project PDF Metadata
{pdf_json}

Task: Evaluate this project against all {cat_name} credits. Use ONLY the \
retrieved reference above plus the PDF metadata. For each credit, determine \
applicability, evaluate questions, determine the achievable level, calculate \
points, and identify documentation gaps. Return JSON following the \
credit_assessments format."""

                gen = call_ollama(system_prompt, user_msg)
                n_wb = sum(1 for s in sources if s["source_type"] == "envision_workbook")
                n_mn = sum(1 for s in sources if s["source_type"] == "guidance_manual")
                result = {
                    "response": gen["response"],
                    "valid_json": gen["valid_json"],
                    "retrieval_seconds": retr_elapsed,
                    "generation_seconds": gen["elapsed_seconds"],
                    "elapsed_seconds": round(retr_elapsed + gen["elapsed_seconds"], 2),
                    "output_tokens": gen["eval_count"],
                    "input_tokens": gen["prompt_eval_count"],
                    "retrieved_sources": sources,
                    "num_sources": len(sources),
                    "num_workbook_hits": n_wb,
                    "num_manual_hits": n_mn,
                    "success": gen["success"],
                }
                status = "✅" if gen["success"] else "❌"
            except Exception as e:
                result = {
                    "response": f"ERROR: {e}",
                    "valid_json": False,
                    "retrieval_seconds": round(time.time() - retr_start, 2),
                    "generation_seconds": 0,
                    "elapsed_seconds": round(time.time() - retr_start, 2),
                    "output_tokens": 0, "input_tokens": 0,
                    "retrieved_sources": [], "num_sources": 0,
                    "num_workbook_hits": 0, "num_manual_hits": 0,
                    "success": False,
                }
                status = "❌"

            cat_safe = cat_name.replace(" ", "_").lower()
            save_result(result, os.path.join(run_dir, f"{cat_safe}.json"))
            all_category_results[cat_name] = result
            json_flag = "" if result["valid_json"] else " ⚠️ non-JSON"
            print(f'{status} ({result["elapsed_seconds"]}s{json_flag}, '
                  f'{result["num_sources"]} sources: '
                  f'{result["num_workbook_hits"]} workbook + '
                  f'{result["num_manual_hits"]} manual)')

        run_elapsed = round(time.time() - run_start, 2)
        summary = {
            "experiment": "rag", "file": file_label, "run": run_num,
            "model": MODEL_NAME, "temperature": TEMPERATURE,
            "retrieval_top_k": RETRIEVAL_TOP_K, "embedding_model": EMBEDDING_MODEL,
            "index_sources": ["envision_workbook", "guidance_manual"],
            "total_elapsed_seconds": run_elapsed,
            "categories": {
                k: {
                    "elapsed_seconds": v["elapsed_seconds"],
                    "retrieval_seconds": v["retrieval_seconds"],
                    "generation_seconds": v["generation_seconds"],
                    "output_tokens": v["output_tokens"],
                    "input_tokens": v["input_tokens"],
                    "num_sources": v["num_sources"],
                    "num_workbook_hits": v["num_workbook_hits"],
                    "num_manual_hits": v["num_manual_hits"],
                    "valid_json": v["valid_json"],
                    "success": v["success"],
                } for k, v in all_category_results.items()
            },
            "timestamp": timestamp(),
        }
        save_result(summary, os.path.join(run_dir, "_summary.json"))
        timing_log.append({"experiment": "rag", "file": file_label,
                           "run": run_num, "seconds": run_elapsed})

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n" + "=" * 70)
print(" ALL EXPERIMENTS COMPLETE")
print(f" Finished: {datetime.datetime.now()}")
print("=" * 70)

timing_df = pd.DataFrame(timing_log)
timing_df.to_csv(os.path.join(RESULTS_DIR, "timing_summary.csv"), index=False)
print("\nTiming summary:")
print(timing_df.to_string(index=False))

# =========================================================
# FLAT CSV EXPORT
# Parses every per-category JSON file and writes one row per
# (arm, run, category, credit), making it easy to load the
# results directly into pandas, Excel, or any statistical tool
# without needing to traverse the nested directory structure.
#
# Responses truncated by MAX_OUTPUT_TOKENS produce incomplete
# JSON; the extractor captures every credit object that was
# fully written before the cutoff and marks the file as
# TRUNCATED so the analyst can filter or handle those rows.
# =========================================================
import csv as _csv
import re as _re
import glob as _glob

_FLAT_CSV = os.path.join(RESULTS_DIR, "credit_assessments_flat.csv")

_FIELDNAMES = [
    "arm", "run", "file", "category",
    "credit_id", "credit_name", "max_points",
    "applicability", "estimated_level", "estimated_points",
    "justification", "questions_summary",
    "level_guide_reference", "concept_phase_actions", "documentation_needed",
    "input_tokens", "output_tokens", "elapsed_seconds",
    "valid_json", "truncated", "success",
]


def _extract_credits(response_text):
    """Return (list[dict], truncated:bool) from a raw LLM response string.

    Handles three cases:
    - Clean JSON inside optional markdown fences (normal case).
    - Truncated JSON where MAX_OUTPUT_TOKENS cut the response mid-object:
      regex-extracts every fully-closed credit object written before the cut.
    - Unrecoverable garbage: returns ([], True).
    """
    text = response_text.strip()
    text = _re.sub(r"^```json?\n?", "", text)
    text = _re.sub(r"\n?```\s*$", "", text)
    start = text.find("{")
    if start == -1:
        return [], True

    # Try the full string first (fast path for valid responses).
    for end in range(len(text) - 1, start, -1):
        if text[end] == "}":
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed.get("credit_assessments", []), False
            except json.JSONDecodeError:
                continue

    # Full parse failed — truncated response.  Extract complete credit
    # objects using a bracket-depth scanner so partial objects are skipped.
    credits, depth, obj_start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                fragment = text[obj_start : i + 1]
                try:
                    obj = json.loads(fragment)
                    if "credit_id" in obj:
                        credits.append(obj)
                except json.JSONDecodeError:
                    pass
    return credits, True  # partial list, mark as truncated


def _qs_summary(questions_evaluated):
    """Flatten questions_evaluated (dict or list) to a readable string."""
    if isinstance(questions_evaluated, dict):
        items = questions_evaluated.items()
    elif isinstance(questions_evaluated, list):
        # Some responses emit a list of {question, status, evidence} dicts.
        items = ((q.get("question", q.get("letter", "?")), q) for q in questions_evaluated)
    else:
        return ""
    parts = []
    for q, v in items:
        if not isinstance(v, dict):
            parts.append(f"Q{q}:{v}")
            continue
        status = v.get("status", "?")
        ev = v.get("evidence") or v.get("gap") or ""
        parts.append(f"Q{q}:{status}" + (f"({ev[:80]})" if ev else ""))
    return " | ".join(parts)


flat_rows = []
_pattern = os.path.join(RESULTS_DIR, "**", "run_*", "*.json")
for _fpath in sorted(_glob.glob(_pattern, recursive=True)):
    _fname = os.path.basename(_fpath)
    if _fname.startswith("_") or "metadata" in _fname:
        continue

    _parts = _fpath.replace(RESULTS_DIR, "").strip("/").split("/")
    if len(_parts) < 4:
        continue
    _file_label, _arm, _run_str, _cat_file = _parts[0], _parts[1], _parts[2], _parts[3]
    _run_num  = int(_run_str.replace("run_", ""))
    _category = _cat_file.replace(".json", "").replace("_", " ").title()

    with open(_fpath) as _f:
        _d = json.load(_f)

    _credits, _truncated = _extract_credits(_d.get("response", ""))

    _base = {
        "arm":            _arm,
        "run":            _run_num,
        "file":           _file_label,
        "category":       _category,
        "input_tokens":   _d.get("prompt_eval_count", _d.get("input_tokens", "")),
        "output_tokens":  _d.get("eval_count", _d.get("output_tokens", "")),
        "elapsed_seconds": _d.get("elapsed_seconds", ""),
        "valid_json":     _d.get("valid_json", False),
        "truncated":      _truncated,
        "success":        _d.get("success", False),
    }

    if not _credits:
        flat_rows.append({**_base, **{k: "" for k in _FIELDNAMES if k not in _base}})
        continue

    for _c in _credits:
        flat_rows.append({
            **_base,
            "credit_id":             _c.get("credit_id", ""),
            "credit_name":           _c.get("credit_name", ""),
            "max_points":            _c.get("max_points", ""),
            "applicability":         _c.get("applicability", ""),
            "estimated_level":       _c.get("estimated_level", ""),
            "estimated_points":      _c.get("estimated_points", ""),
            "justification":         _c.get("justification", ""),
            "questions_summary":     _qs_summary(_c.get("questions_evaluated", {})),
            "level_guide_reference": _c.get("level_guide_reference", ""),
            "concept_phase_actions": "; ".join(_c.get("concept_phase_actions", [])),
            "documentation_needed":  "; ".join(_c.get("documentation_needed", [])),
        })

with open(_FLAT_CSV, "w", newline="", encoding="utf-8") as _f:
    _writer = _csv.DictWriter(_f, fieldnames=_FIELDNAMES, extrasaction="ignore")
    _writer.writeheader()
    _writer.writerows(flat_rows)

print(f"\n✅ Flat CSV saved → {_FLAT_CSV}")
print(f"   {len(flat_rows)} rows | "
      f"{len([r for r in flat_rows if not r['truncated']])} complete | "
      f"{len([r for r in flat_rows if r['truncated']])} from truncated responses")
