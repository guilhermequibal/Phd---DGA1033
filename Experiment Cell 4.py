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
    pdf_json = json.dumps(pdf_data, indent=2, default=str)

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
