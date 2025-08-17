import streamlit as st
import os, json, re, string
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from difflib import get_close_matches

load_dotenv()
st.title("StarTech.com Products Chatbot")

# Fallback trigger phrases
fallback_keywords = [
    "what color","what colours","what colour","what type","what kind","how big","how small","how long",
    "how wide","how tall","how thick","how heavy","how many","how much","what size","what sizes","does it",
    "is it","is it compatible","are they","specs","details","specifications","tech specs","technical details",
    "what ports","which ports","what connectors","which connectors","what inputs","what outputs",
    "how fast","what speed","what resolution","is this compatible","is this supported","will this work", "tell me more",
    "can i use this","can it","is there","do they","what’s included","what is included","what do you get",
    "included accessories","in the box","what’s in the box","what comes with","do i need","will it help",
    "does it require","will it fit","will it keep","what version","any differences","any difference","difference between"
]
farewell_keywords = ["thank you","thanks","appreciate it","cheers","bye","goodbye","see you","you’ve been helpful","you have been helpful","that’s all","that is all"]

# numeric metadata keywords
metadata_field_keywords = {
    "ports": ["total ports","ports total","ports"],
    "packqty": ["in the pack","in the package","package quantity","pack qty","in a pack","in a package"],
    "displays": ["displays","monitors","screens","number of displays"],
    "numharddrive": ["hard drive","hard drives"],
    "kvmports": ["kvm ports","ports kvm"],
    "cablelength": ["cable length","length of cable","cablelength","cord length"]
}

# ---------- cached resources ----------
@st.cache_resource
def load_vector_store():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"])
    return PineconeVectorStore(index=index, embedding=embeddings)

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0.7)

@st.cache_resource
def load_categorical_values():
    path = "documents/categorical_values.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("category", [])
            data.setdefault("subcategory", [])
            data.setdefault("material", [])
            data.setdefault("fiberduplex", [])
            data.setdefault("fibertype", [])
            data.setdefault("material_tags", [])
            data.setdefault("color", [])
            data.setdefault("wireless", [])
            return data
    return {
        "category": [],
        "subcategory": [],
        "material": [],
        "fiberduplex": [],
        "fibertype": [],
        "material_tags": [],
        "color": [],
        "wireless": [],
    }

vector_store = load_vector_store()
llm = load_llm()
categorical_values = load_categorical_values()

# ---------- utils ----------
def norm_text(s):
    return s.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def extract_product_number(text):
    m = re.search(r"\b(?=[A-Z0-9-]{6,}\b)(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b", text)
    return m.group(0) if m else None

def is_vague_follow_up(prompt):
    if extract_product_number(prompt):  # explicit product ref => not vague
        return False
    clean = norm_text(prompt)
    return len(clean.split()) <= 6 or any(k in clean for k in fallback_keywords)

def is_farewell(prompt):
    p = norm_text(prompt)
    return any(k in p for k in farewell_keywords)

def try_match_categorical(meta_key: str, prompt_norm: str):
    values = categorical_values.get(meta_key, [])
    for v in values:
        v_raw = str(v).strip().lower()
        v_norm = norm_text(v_raw)
        if v_norm and v_norm in prompt_norm:
            return v_raw
    matches = get_close_matches(prompt_norm, [str(v).strip().lower() for v in values], n=1, cutoff=0.92)
    return matches[0] if matches else None

# NEW: infer yes/no for wireless from natural phrasing
def infer_wireless_from_prompt(prompt_norm: str):
    mentions = any(tok in prompt_norm for tok in ["wireless", "wifi", "wi fi"])
    if not mentions:
        return None
    negatives = ["no wireless", "without wireless", "not wireless", "no wifi", "without wifi", "wired only"]
    if any(neg in prompt_norm for neg in negatives):
        return "no"
    positives = ["must be wireless", "need wireless", "with wireless", "wireless only", "wireless required"]
    if any(pos in prompt_norm for pos in positives) or "wireless" in prompt_norm or "wifi" in prompt_norm or "wi fi" in prompt_norm:
        return "yes"
    return None

# --- numeric helpers (decimals + ranges) ---
def find_number_near_keywords(prompt_norm: str, keywords: list[str]):
    for kw in keywords:
        kw_esc = re.escape(kw)
        m = re.search(rf'(\d+(?:\.\d+)?)\s+{kw_esc}\b', prompt_norm)
        if m:
            return float(m.group(1))
        m = re.search(rf'\b{kw_esc}\s+(\d+(?:\.\d+)?)', prompt_norm)
        if m:
            return float(m.group(1))
    return None

def parse_global_range(prompt_norm: str):
    m = re.search(r'(?:between|from)\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)', prompt_norm)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return {"$gte": lo, "$lte": hi}
    m = re.search(r'(?:<=|less than or equal to|at most|no more than|up to)\s*(\d+(?:\.\d+)?)', prompt_norm)
    if m: return {"$lte": float(m.group(1))}
    m = re.search(r'(?:<|less than|under|below)\s*(\d+(?:\.\d+)?)', prompt_norm)
    if m: return {"$lt": float(m.group(1))}
    m = re.search(r'(?:>=|greater than or equal to|at least|no less than|minimum of)\s*(\d+(?:\.\d+)?)', prompt_norm)
    if m: return {"$gte": float(m.group(1))}
    m = re.search(r'(?:>|greater than|over|more than)\s*(\d+(?:\.\d+)?)', prompt_norm)
    if m: return {"$gt": float(m.group(1))}
    return None

def extract_filter_from_prompt(prompt):
    prompt_norm = norm_text(prompt)
    filters = {}

    numeric_hits = [field for field, keywords in metadata_field_keywords.items() if any(kw in prompt_norm for kw in keywords)]

    if len(numeric_hits) == 1:
        field = numeric_hits[0]
        rng = parse_global_range(prompt_norm)
        if rng:
            filters[field] = rng
        else:
            val = find_number_near_keywords(prompt_norm, metadata_field_keywords[field])
            if val is not None:
                filters[field] = val
    else:
        for field, keywords in metadata_field_keywords.items():
            if any(kw in prompt_norm for kw in keywords):
                val = find_number_near_keywords(prompt_norm, keywords)
                if val is not None:
                    filters[field] = val

    for meta_key in ("category", "subcategory", "material", "fiberduplex", "fibertype", "color", "wireless"):
        val = try_match_categorical(meta_key, prompt_norm)
        if val:
            filters[meta_key] = val.strip().lower()

    if "wireless" not in filters:
        w = infer_wireless_from_prompt(prompt_norm)
        if w:
            filters["wireless"] = w

    tag_hits = [t for t in categorical_values.get("material_tags", []) if t and t in prompt_norm]
    if tag_hits:
        for t in sorted(set(tag_hits)):
            filters[f"mtag_{t}"] = True
        filters.pop("material", None)

    return filters or None

def show_response(reply):
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append(AIMessage(reply))

# ---------- handlers ----------
def handle_greeting(prompt):
    greeting_keywords = ["hello","hi","hey","good morning","good afternoon","good evening"]
    wc = len(prompt.strip().split())
    if not st.session_state.greeted_user and wc <= 4 and any(re.search(rf"\b{g}\b", prompt.lower()) for g in greeting_keywords):
        show_response("Hi there! 👋 I'm here to help you find the right StarTech.com product.\n\nYou can ask me about product specs, features, or tell me what you're trying to do and I’ll recommend something that fits.")
        st.session_state.greeted_user = True
        return True
    return False

def handle_farewell(prompt):
    if is_farewell(prompt):
        show_response("Thank you for reaching out! If you have any more questions or need further assistance, feel free to ask. I'm here to help!")
        return True
    return False

def handle_explicit_product(prompt):
    pn = extract_product_number(prompt)
    if pn:
        docs = vector_store.similarity_search(query="product spec", k=5, filter={"product_number": pn.upper()})
        if docs:
            st.session_state.last_product_number = pn.upper()
            st.session_state.last_context = docs[0].page_content
            st.session_state.last_score = 1.0
            st.session_state.last_metadata = docs[0].metadata or {}
        else:
            st.session_state.last_product_number = ""
            st.session_state.last_context = ""
            st.session_state.last_score = 0.0
            st.session_state.last_metadata = {}
        return True
    return False

def handle_vague_followup(prompt):
    if is_vague_follow_up(prompt):
        if not st.session_state.last_context:
            show_response(
                "Could you tell me a bit more about what you're looking for? For example:\n"
                "- Is there a specific problem you are trying to solve?\n"
                "- What specs in a product are important to you?\n"
                "- What do you want the product to connect to or be compatible with?\n\n"
                "You can also mention a product number if you already have one in mind."
            )
            return True
        st.session_state.last_score = 1.0
        messages = (
            st.session_state.messages +
            [SystemMessage(f"This is the specification for product {st.session_state.last_product_number}:\n{st.session_state.last_context}")] +
            [HumanMessage(prompt)]
        )
        reply = llm.invoke(messages).content
        show_response(reply)
        return True
    return False

def handle_descriptive_query(prompt):
    f = extract_filter_from_prompt(prompt)
    try:
        results = vector_store.similarity_search_with_relevance_scores(prompt, k=5, filter=f)
        results = [(doc, score) for doc, score in results if score >= 0.35]
        if results: return use_top_result(results)
    except Exception as e:
        print(f"Metadata-filtered search failed: {e}")

    results = vector_store.similarity_search_with_relevance_scores(prompt, k=5)
    results = [(doc, score) for doc, score in results if score >= 0.35]
    if results: return use_top_result(results)
    return False

def use_top_result(results):
    top_doc, score = results[0]
    st.session_state.last_product_number = top_doc.metadata.get("product_number", "")
    st.session_state.last_context = top_doc.page_content
    st.session_state.last_score = score
    st.session_state.last_metadata = top_doc.metadata or {}
    return True

# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(
        "You are a StarTech.com assistant. Always be friendly and professional. "
        "You only answer questions about StarTech.com products using the provided context. "
        "Do not mention or recommend products from any other company, supplier, or brand — only StarTech.com. "
        "Do not make up information. If you’re unsure, ask for clarification. "
        "Stay helpful, polite, and focused on StarTech.com solutions."
    )]
if "last_product_number" not in st.session_state: st.session_state.last_product_number = ""
if "last_context" not in st.session_state: st.session_state.last_context = ""
if "last_score" not in st.session_state: st.session_state.last_score = None
if "greeted_user" not in st.session_state: st.session_state.greeted_user = False
if "last_metadata" not in st.session_state: st.session_state.last_metadata = {}

# ---------- chat flow ----------
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage): continue
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

prompt = st.chat_input("Ask about a StarTech.com product")
if prompt:
    st.session_state.messages.append(HumanMessage(prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    if handle_greeting(prompt): st.stop()
    if handle_farewell(prompt): st.stop()
    if handle_explicit_product(prompt): pass
    elif handle_vague_followup(prompt): st.stop()
    else: handle_descriptive_query(prompt)

    if not st.session_state.last_context:
        show_response(
            "Could you tell me a bit more about what you're looking for? For example:\n"
            "- Is there a specific problem you are trying to solve?\n"
            "- What specs in a product are important to you?\n"
            "- What do you want the product to connect to or be compatible with?\n\n"
            "You can also mention a product number if you already have one in mind."
        )
        st.stop()

    filters = extract_filter_from_prompt(prompt)

    print("\n--- Debug Info ---")
    print(f"Active Filters: {filters}")
    print(f"User Prompt: {prompt}")
    print(f"Product Number: {st.session_state.last_product_number}")
    print(f"Similarity Score: {st.session_state.last_score if st.session_state.last_score is not None else 'N/A'}")

    md = st.session_state.get("last_metadata") or {}
    interesting_keys = (
        "category", "subcategory", "material", "material_tags",
        "fiberduplex", "fibertype", "ports", "displays", "color", "cablelength", "wireless"
    )

    print("Resolved Metadata (top doc):")
    for k in interesting_keys:
        if k in md:
            print(f"- {k}: {md[k]}")

    print("Context Retrieved:")
    print(st.session_state.last_context)
    print("-------------------\n")

    messages = (
        st.session_state.messages +
        [SystemMessage(f"This is the specification for product {st.session_state.last_product_number}:\n{st.session_state.last_context}")] +
        [HumanMessage(prompt)]
    )
    reply = llm.invoke(messages).content

    pnums = re.findall(r"\b(?=[A-Z0-9-]{6,}\b)(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b", reply)
    if len(set(pnums)) > 1:
        reply += "\n\n**Which product would you like to know more about?**"

    show_response(reply)
