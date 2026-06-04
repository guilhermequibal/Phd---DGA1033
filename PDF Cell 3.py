# =============================================================
# CELL 3: BUILD RAG INDEX
# =============================================================
# Sources indexed:
#   1. envision_index.json  — workbook (scoring state, questions, rubrics)
#   2. ISI Envision Manual  — guidance PDF (intent, applicability, criteria text)
# Both are stored in the same ChromaDB collection with a `source_type` field
# so the retriever can transparently pull from either when answering a query.
# =============================================================

import chromadb
import llama_index.core
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

Settings.llm = Ollama(
    model=MODEL_NAME,
    request_timeout=7200,
    temperature=TEMPERATURE
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL
)

print('Building the vector index from envision_index.json AND user guide PDF...')

documents = []

# ------------------------------------------------------------------
# SOURCE 1 — Envision workbook (JSON)
# Covers: credit IDs, scoring rubrics, questions, lookup thresholds,
#         applicability answers, and tabulation data.
# ------------------------------------------------------------------
with open(ENVISION_INDEX_PATH) as f:
    envision_raw = json.load(f)

for credit_id, credit in envision_raw['credits'].items():
    text = f"""Credit: {credit['credit_id']} - {credit['credit_name']}
Category: {credit['sheet']} / {credit['category']}
Points: {credit['points_display']}
{credit['intent']}
{credit['metric']}
{credit['applicability_description']}
Questions:"""
    for q in credit.get('questions', []):
        text += f"\n  {q['letter']}: {q['text']}"

    if credit.get('lookup_criteria'):
        text += '\n\nLookup Criteria:'
        for lc in credit['lookup_criteria']:
            text += f"\n  {lc['criterion']}: {lc['description']}"

    tab = credit.get('tabulation', {})
    if tab:
        text += '\n\nLevel Guides:'
        for lvl in ['Improved', 'Enhanced', 'Superior', 'Conserving']:
            g = tab.get(f'LevelGuide_{lvl}', '')
            if g:
                text += f"\n  {lvl}: {g}"

    pts = credit.get('points', {})
    if pts:
        text += f"""\n\nPoints Rubric:
No Level={pts.get('No_Level', 0)},
Improved={pts.get('Improved', 0)},
Enhanced={pts.get('Enhanced', 0)},
Superior={pts.get('Superior', 0)},
Conserving={pts.get('Conserving', 0)},
Restorative={pts.get('Restorative', 'N/A')}"""

    documents.append(Document(
        text=text,
        metadata={
            'source_type': 'envision_workbook',
            'credit_id': credit_id,
            'credit_name': credit['credit_name'],
            'category': credit['sheet'],
            'subcategory': credit['category'],
            'max_points': pts.get('Total', 0),
            'has_lookup': bool(credit.get('lookup_criteria')),
        }
    ))

n_workbook = len(documents)
print(f'  ✅ Workbook: {n_workbook} credit documents from envision_index.json')

# ------------------------------------------------------------------
# SOURCE 2 — ISI Envision Guidance Manual (PDF)
# Covers: full credit descriptions, applicability statements,
#         performance-improvement guidance, evaluation criteria text,
#         documentation guidance, and related-credits sections.
# These are the passages the system prompt (§2.1) requires for Phase 1
# applicability checks and Phase 2 criteria evaluation.
# ------------------------------------------------------------------
print(f'  Chunking user guide: {os.path.basename(USER_GUIDE_PDF_PATH)} ...')
guide_chunks = chunk_pdf_for_rag(USER_GUIDE_PDF_PATH, chunk_size=1500, overlap=200)

for chunk in guide_chunks:
    documents.append(Document(
        text=chunk['text'],
        metadata={
            'source_type': 'guidance_manual',
            'source_file': chunk['source_file'],
            'chunk_idx': chunk['chunk_idx'],
            'page_approx': chunk['page_approx'],
        }
    ))

n_manual = len(guide_chunks)
print(f'  ✅ Manual: {n_manual} chunks from user guide PDF')
print(f'  Total documents in index: {len(documents)}')

# ------------------------------------------------------------------
# Build ChromaDB vector index
# ------------------------------------------------------------------
db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    db_client.delete_collection('envision_credits')
except Exception:
    pass

chroma_collection = db_client.create_collection('envision_credits')
vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context   = StorageContext.from_defaults(vector_store=vector_store)

rag_index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=Settings.embed_model,
    show_progress=True
)

print(f'\n✅ RAG index built with {len(documents)} total documents')
print(f'   Workbook entries: {n_workbook} | Manual chunks: {n_manual}')
print(f'   Saved to: {CHROMA_DB_PATH}')
