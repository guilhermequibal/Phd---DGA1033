# =============================================================
# CELL 4: RUNNER (PDFs)
# =============================================================

print('=' * 70)
print(' ENVISION PDF EXPERIMENT RUNNER')
print(f' Started:  {datetime.datetime.now()}')
print(f' Model:    {MODEL_NAME}')
print(f' Files:    {len(PDF_FILES)} PDF file(s)')
print(f' Runs:     {NUM_RUNS} per experiment')
print('=' * 70)

# Load the base prompt
with open(SYSTEM_PROMPT_PATH) as f:
    system_prompt = f.read()

# Load the Envision categories
categories, meta = load_envision_by_category()
category_names = list(categories.keys())

# RAG Engine
rag_query_engine = rag_index.as_query_engine(
    similarity_top_k=RETRIEVAL_TOP_K,
    system_prompt=system_prompt
)

timing_log = []

for file_idx, pdf_path in enumerate(PDF_FILES):
    file_label = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f'\n{"=" * 70}')
    print(f'FILE {file_idx+1}/{len(PDF_FILES)}: {file_label}')
    print(f'{"=" * 70}')

    # --- PDF Extraction ---
    print('\n Extracting text from a PDF...')
    try:
        pdf_data = extract_pdf_metadata(pdf_path)
        pdf_json = json.dumps(pdf_data, indent=2, default=str)
        print(f' ✅ Text successfully extracted from the PDF.')
    except Exception as e:
        print(f' ❌ Error extracting the PDF: {e}')
        print(f' Skipping this file.')
        continue

    # Save the extracted text for future reference
    save_result(pdf_data, os.path.join(RESULTS_DIR, file_label, 'pdf_metadata.json'))

    # =====================================================
    # EXPERIMENT A: ZERO-SHOT
    # =====================================================
    for run_num in range(1, NUM_RUNS + 1):
        run_dir = os.path.join(RESULTS_DIR, file_label, 'zero_shot', f'run_{run_num}')
        os.makedirs(run_dir, exist_ok=True)
        print(f'\n --- ZERO-SHOT | Run {run_num}/{NUM_RUNS} ---')

        run_start = time.time()
        all_category_results = {}

        for cat_name, cat_credits in categories.items():
            print(f'   Category: {cat_name} ({len(cat_credits)} credits)...', end=' ')

            cat_json = json.dumps(cat_credits, indent=2, default=str)
            user_msg = f"""## Envision Reference Data — {cat_name} Category
{cat_json}

## Project PDF Metadata
{pdf_json}

Task: Evaluate this project against all {cat_name} credits shown above. For each credit, determine applicability, evaluate questions against the PDF metadata, determine the achievable level, calculate points, and identify documentation gaps. Return your assessment as JSON following the credit_assessments format in your system instructions."""

            result = call_ollama(system_prompt, user_msg, timeout=7200)

            cat_safe = cat_name.replace(' ', '_').lower()
            save_result(result, os.path.join(run_dir, f'{cat_safe}.json'))
            all_category_results[cat_name] = result

            status = '✅' if result['success'] else '❌'
            print(f'{status} ({result["elapsed_seconds"]}s)')

        run_elapsed = round(time.time() - run_start, 2)
        summary = {
            'experiment': 'zero_shot', 'file': file_label, 'run': run_num,
            'model': MODEL_NAME, 'temperature': TEMPERATURE,
            'total_elapsed_seconds': run_elapsed,
            'categories': {
                k: {
                    'elapsed_seconds': v['elapsed_seconds'],
                    'output_tokens': v['eval_count'],
                    'input_tokens': v['prompt_eval_count'],
                    'success': v['success'],
                } for k, v in all_category_results.items()
            },
            'timestamp': timestamp()
        }
        save_result(summary, os.path.join(run_dir, '_summary.json'))
        timing_log.append({'experiment': 'zero_shot', 'file': file_label,
                           'run': run_num, 'seconds': run_elapsed})

    # =====================================================
    # EXPERIMENT B: RAG
    # =====================================================
    for run_num in range(1, NUM_RUNS + 1):
        run_dir = os.path.join(RESULTS_DIR, file_label, 'rag', f'run_{run_num}')
        os.makedirs(run_dir, exist_ok=True)
        print(f'\n --- RAG | Run {run_num}/{NUM_RUNS} ---')

        run_start = time.time()
        all_category_results = {}

        for cat_name in category_names:
            n_credits = len(categories[cat_name])
            print(f'   Category: {cat_name} ({n_credits} credits)...', end=' ')

            query_text = f"""Evaluate this infrastructure project against all Envision credits in the {cat_name} category.
Project PDF Metadata: {pdf_json}
For each credit in {cat_name}, determine applicability, evaluate questions, determine the achievable level, calculate points, and identify documentation gaps. Return JSON following the credit_assessments format."""

            rag_start = time.time()
            try:
                rag_response = rag_query_engine.query(query_text)
                rag_elapsed  = round(time.time() - rag_start, 2)

                sources = []
                if hasattr(rag_response, 'source_nodes'):
                    for node in rag_response.source_nodes:
                        source_type = node.metadata.get('source_type', 'unknown')
                        if source_type == 'envision_workbook':
                            source_id = node.metadata.get('credit_id', '?')
                            location  = node.metadata.get('category', '?')
                        else:
                            # guidance_manual chunk
                            source_id = f"p{node.metadata.get('page_approx', '?')}"
                            location  = node.metadata.get('source_file', '?')
                        sources.append({
                            'source_type': source_type,
                            'source_id':   source_id,
                            'location':    location,
                            'score': round(node.score, 4) if node.score else None,
                        })

                n_workbook_hits = sum(1 for s in sources if s['source_type'] == 'envision_workbook')
                n_manual_hits   = sum(1 for s in sources if s['source_type'] == 'guidance_manual')

                result = {
                    'response': str(rag_response),
                    'elapsed_seconds': rag_elapsed,
                    'retrieved_sources': sources,
                    'num_sources': len(sources),
                    'num_workbook_hits': n_workbook_hits,
                    'num_manual_hits': n_manual_hits,
                    'success': True
                }
                status = '✅'

            except Exception as e:
                rag_elapsed = round(time.time() - rag_start, 2)
                result = {
                    'response': f'ERROR: {str(e)}',
                    'elapsed_seconds': rag_elapsed,
                    'retrieved_sources': [],
                    'num_sources': 0,
                    'num_workbook_hits': 0,
                    'num_manual_hits': 0,
                    'success': False
                }
                status = '❌'

            cat_safe = cat_name.replace(' ', '_').lower()
            save_result(result, os.path.join(run_dir, f'{cat_safe}.json'))
            all_category_results[cat_name] = result

            print(
                f'{status} ({result["elapsed_seconds"]}s, '
                f'{result["num_sources"]} sources: '
                f'{result["num_workbook_hits"]} workbook + '
                f'{result["num_manual_hits"]} manual)'
            )

        run_elapsed = round(time.time() - run_start, 2)
        summary = {
            'experiment': 'rag', 'file': file_label, 'run': run_num,
            'model': MODEL_NAME, 'temperature': TEMPERATURE,
            'retrieval_top_k': RETRIEVAL_TOP_K, 'embedding_model': EMBEDDING_MODEL,
            'index_sources': ['envision_workbook', 'guidance_manual'],
            'total_elapsed_seconds': run_elapsed,
            'categories': {
                k: {
                    'elapsed_seconds': v['elapsed_seconds'],
                    'num_sources': v['num_sources'],
                    'num_workbook_hits': v['num_workbook_hits'],
                    'num_manual_hits': v['num_manual_hits'],
                    'success': v['success'],
                } for k, v in all_category_results.items()
            },
            'timestamp': timestamp()
        }
        save_result(summary, os.path.join(run_dir, '_summary.json'))
        timing_log.append({'experiment': 'rag', 'file': file_label,
                           'run': run_num, 'seconds': run_elapsed})

# =========================================================
# FINAL SUMMARY
# =========================================================
print('\n' + '=' * 70)
print(' ALL EXPERIMENTS COMPLETE')
print(f' Finished: {datetime.datetime.now()}')
print('=' * 70)

timing_df = pd.DataFrame(timing_log)
timing_df.to_csv(os.path.join(RESULTS_DIR, 'timing_summary.csv'), index=False)
print('\nTiming summary:')
print(timing_df.to_string(index=False))
