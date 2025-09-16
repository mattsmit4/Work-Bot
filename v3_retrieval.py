# Import basics
import os
import time
import re
from dotenv import load_dotenv

# Import Pinecone and LangChain tools
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize Pinecone connection
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Initialize embedding model and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ---------- Metadata filter logic ----------

# Maps user synonyms to metadata field names
metadata_field_keywords = {
    "totalports": ["total ports", "number of ports", "ports total", "how many ports"],
    "dockusbports": ["usb ports", "number of usb ports"],
    "weightofproduct": ["weight of product", "product weight", "how heavy", "weight"],
    "shipping(package)weight": ["shipping weight"],
    "drivesize": ["drive size", "how big drive", "disk size"],
    "kvmsupport": ["kvm", "kvm support"],
    # Add more if needed
}

# Try to extract numeric metadata filter from the query
def extract_filter_from_query(query):
    query_lower = query.lower()
    for field, keywords in metadata_field_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                match = re.search(r'(\d+(\.\d+)?)', query_lower)
                if match:
                    value = float(match.group(1))
                    return {field: int(value) if value.is_integer() else value}
    return None

# ---------- Query section ----------

query = "Do you have any products that have 15 total ports?"
filter_dict = extract_filter_from_query(query)

# Try up to 3 times in case of connection issues
for attempt in range(3):
    try:
        results = vector_store.similarity_search_with_score(
            query,
            k=5,
            filter=filter_dict
        )
        threshold = 0.2 if filter_dict else 0.5
        results = [r for r in results if r[1] >= threshold]
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(2)
else:
    raise RuntimeError("All retries failed.")

# ---------- Output section ----------
if results:
    print(f"✅ Found {len(results)} relevant result(s):\n" + "-" * 60)
    for i, (doc, score) in enumerate(results, 1):
        sku = doc.metadata.get("product_number", "Unknown SKU")
        print(f"{i}. SKU: {sku} (Score: {score:.4f})\n---\n{doc.page_content[:3000]}...\n")
else:
    print("❌ No relevant results found.")
