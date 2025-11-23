import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
EMBED_MODEL = "text-embedding-3-large"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Get ANY USB products and see what their metadata looks like
print("=== SEARCHING FOR USB PRODUCTS (NO FILTER) ===")
docs = vector_store.similarity_search("usb cable", k=15)
print(f"Found {len(docs)} total docs\n")

print("=== ACTUAL SUBCATEGORY VALUES IN PINECONE ===")
subcats_seen = set()
for i, doc in enumerate(docs):
    meta = doc.metadata or {}
    subcat = meta.get('subcategory', 'N/A')
    cat = meta.get('category', 'N/A')
    pn = meta.get('product_number', 'N/A')
    if subcat != 'N/A':
        subcats_seen.add(subcat)
    print(f"{i+1}. {pn}")
    print(f"   category: '{cat}'")
    print(f"   subcategory: '{subcat}'")
    print()

print("\n=== UNIQUE SUBCATEGORIES FOUND ===")
for s in sorted(subcats_seen):
    print(f"  - '{s}'")