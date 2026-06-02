# =============================================================
# CELL 3: BUILD RAG INDEX 
# =============================================================

import chromadb
import llama_index.core
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# Secure timeout on LlamaIndex!
Settings.llm = Ollama(
    model=MODEL_NAME, 
    request_timeout=7200, 
    temperature=TEMPERATURE
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL
)

print('Building the vector index from envision_index.json...')

with open(ENVISION_INDEX_PATH) as f:
    envision_raw = json.load(f)

documents = []
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
No Level={pts.get('No_Level',0)}, 
Improved={pts.get('Improved',0)}, 
Enhanced={pts.get('Enhanced',0)}, 
Superior={pts.get('Superior',0)}, 
Conserving={pts.get('Conserving',0)}, 
Restorative={pts.get('Restorative','N/A')}"""
        
    doc = Document(
        text=text,
        metadata={
            'credit_id': credit_id,
            'credit_name': credit['credit_name'],
            'category': credit['sheet'],
            'subcategory': credit['category'],
            'max_points': pts.get('Total', 0),
            'has_lookup': bool(credit.get('lookup_criteria')),
        }
    )
    documents.append(doc)

db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    db_client.delete_collection('envision_credits')
except:
    pass

chroma_collection = db_client.create_collection('envision_credits')
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

rag_index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context, 
    embed_model=Settings.embed_model,
    show_progress=True
)

print(f'\n✅ RAG index successfully built and saved in {CHROMA_DB_PATH}')