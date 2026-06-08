# =============================================================
# CELL 3: BUILD RAG INDEX
#
# STRATEGY:
# This cell constructs the vector index that powers the RAG experimental
# arm. Two complementary knowledge sources are indexed together:
#
#   1. envision_index.json — structured workbook data (credit IDs, scoring
#      rubrics, applicability questions, level guides, point tables).
#   2. ISI Envision Guidance Manual (PDF) — unstructured narrative text
#      (design intent, applicability criteria, measurement guidance).
#
# Storing both in the same ChromaDB collection with a `source_type` field
# lets the retriever draw relevant context from either source in a single
# query, without the caller needing to manage two separate indexes.
#
# To avoid redundant computation on repeated runs, a SHA-256 fingerprint
# of the source files and chunking/embedding parameters is compared
# against the stored fingerprint before deciding whether to rebuild.
# If nothing changed, the existing index is reused directly.
# =============================================================
import os, json, hashlib

import chromadb
from llama_index.core import (
    Document, VectorStoreIndex, StorageContext, Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(
    model=MODEL_NAME,
    request_timeout=OLLAMA_TIMEOUT,
    temperature=TEMPERATURE,
)
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

COLLECTION_NAME = "envision_credits"
FINGERPRINT_PATH = os.path.join(CHROMA_DB_PATH, "_fingerprint.json")


def format_credit_text(credit):
    """Serialize a credit entry to a plain-text document for embedding.

    Uses .get() with fallbacks on every field so that a missing or renamed
    key in the JSON schema degrades gracefully rather than raising a KeyError
    that would abort the entire index build.
    """
    text = (
        f"Credit: {credit.get('credit_id', '?')} - {credit.get('credit_name', '?')}\n"
        f"Category: {credit.get('sheet', '?')} / {credit.get('category', '?')}\n"
        f"Points: {credit.get('points_display', '?')}\n"
        f"{credit.get('intent', '')}\n"
        f"{credit.get('metric', '')}\n"
        f"{credit.get('applicability_description', '')}\n"
        "Questions:"
    )
    for q in credit.get("questions", []):
        text += f"\n  {q.get('letter', '?')}: {q.get('text', '')}"

    if credit.get("lookup_criteria"):
        text += "\n\nLookup Criteria:"
        for lc in credit["lookup_criteria"]:
            text += f"\n  {lc.get('criterion', '?')}: {lc.get('description', '')}"

    tab = credit.get("tabulation", {})
    if tab:
        text += "\n\nLevel Guides:"
        for lvl in ["Improved", "Enhanced", "Superior", "Conserving"]:
            g = tab.get(f"LevelGuide_{lvl}", "")
            if g:
                text += f"\n  {lvl}: {g}"

    pts = credit.get("points", {})
    if pts:
        text += (
            "\n\nPoints Rubric:"
            f"\n  No Level={pts.get('No_Level', 0)},"
            f"\n  Improved={pts.get('Improved', 0)},"
            f"\n  Enhanced={pts.get('Enhanced', 0)},"
            f"\n  Superior={pts.get('Superior', 0)},"
            f"\n  Conserving={pts.get('Conserving', 0)},"
            f"\n  Restorative={pts.get('Restorative', 'N/A')}"
        )
    return text


def _source_fingerprint():
    """Compute a hash over source file metadata and indexing parameters.

    Any change to a source file or to EMBEDDING_MODEL, CHUNK_SIZE, or
    CHUNK_OVERLAP invalidates this fingerprint and triggers a rebuild.
    """
    h = hashlib.sha256()
    for p in [ENVISION_INDEX_PATH, USER_GUIDE_PDF_PATH]:
        h.update(p.encode())
        if os.path.exists(p):
            st = os.stat(p)
            h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    h.update(f"{EMBEDDING_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    return h.hexdigest()


def _existing_fingerprint():
    try:
        with open(FINGERPRINT_PATH) as f:
            return json.load(f).get("fingerprint")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _build_documents():
    documents = []

    # --- SOURCE 1 — Envision workbook (JSON) ---
    with open(ENVISION_INDEX_PATH) as f:
        envision_raw = json.load(f)
    for credit_id, credit in envision_raw["credits"].items():
        pts = credit.get("points", {})
        documents.append(Document(
            text=format_credit_text(credit),
            metadata={
                "source_type": "envision_workbook",
                "credit_id": credit_id,
                "credit_name": credit.get("credit_name", "?"),
                "category": credit.get("sheet", "?"),
                "subcategory": credit.get("category", "?"),
                "max_points": pts.get("Total", 0),
                "has_lookup": bool(credit.get("lookup_criteria")),
            },
        ))
    n_workbook = len(documents)
    print(f"   ✅ Workbook: {n_workbook} credit documents from envision_index.json")

    # --- SOURCE 2 — ISI Envision Guidance Manual (PDF) ---
    print(f"   Chunking user guide: {os.path.basename(USER_GUIDE_PDF_PATH)} ...")
    guide_chunks = chunk_pdf_for_rag(
        USER_GUIDE_PDF_PATH, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP
    )
    for chunk in guide_chunks:
        documents.append(Document(
            text=chunk["text"],
            metadata={
                "source_type": "guidance_manual",
                "source_file": chunk["source_file"],
                "chunk_idx": chunk["chunk_idx"],
                "page_approx": chunk["page_approx"],
            },
        ))
    n_manual = len(guide_chunks)
    print(f"   ✅ Manual: {n_manual} chunks from user guide PDF")
    return documents, n_workbook, n_manual


# ------------------------------------------------------------------
# Decide whether to rebuild or reuse the existing index.
# Reusing an unchanged index avoids re-embedding the entire corpus,
# which can take several minutes on first build.
# ------------------------------------------------------------------
fingerprint = _source_fingerprint()
db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

reuse = False
if REBUILD_INDEX == "never":
    reuse = True
elif REBUILD_INDEX == "auto":
    reuse = (_existing_fingerprint() == fingerprint)

rag_index = None
if reuse:
    try:
        chroma_collection = db_client.get_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        rag_index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=Settings.embed_model
        )
        print(f"♻️  Reusing existing index ({chroma_collection.count()} vectors) "
              f"— sources unchanged.")
    except Exception as e:  # chromadb raises various types; any failure triggers rebuild
        print(f"   (could not reuse existing index: {e}) — rebuilding.")
        reuse = False

if not reuse:
    print("Building the vector index from envision_index.json AND user guide PDF...")
    documents, n_workbook, n_manual = _build_documents()
    print(f"   Total documents in index: {len(documents)}")

    try:
        db_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    chroma_collection = db_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    rag_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
        show_progress=True,
    )

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    with open(FINGERPRINT_PATH, "w") as f:
        json.dump({"fingerprint": fingerprint,
                   "n_workbook": n_workbook,
                   "n_manual": n_manual}, f)

    print(f"\n✅ RAG index built with {len(documents)} total documents")
    print(f"   Workbook entries: {n_workbook} | Manual chunks: {n_manual}")
    print(f"   Saved to: {CHROMA_DB_PATH}")
