# v3_Work_Chatbot.py

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

# ---- env helpers (consistent, safe) ----
def env_required(name: str, hint: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing environment variable '{name}'. {hint}")
    return v

def env_optional(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or str(v).strip() == "" else v

# Required
PINECONE_API_KEY = env_required("PINECONE_API_KEY")
OPENAI_API_KEY = env_required("OPENAI_API_KEY")
INDEX_NAME = env_required("PINECONE_INDEX_NAME")

# Optional tunables
CHAT_MODEL = env_optional("OPENAI_CHAT_MODEL", "gpt-4o")
TEMP = float(env_optional("OPENAI_TEMPERATURE", "0.7"))
EMBED_MODEL = env_optional("EMBED_MODEL", "text-embedding-3-large")

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
farewell_keywords = ["thank you","thanks","appreciate it","cheers","bye","goodbye","see you","you’ve been helpful","you have been helpful","that’s all","that is all", "cool"]

# Step 6: compare/disambiguate triggers
compare_keywords = ["compare", "difference between", "differences", "which one", "vs", "versus"]

# NEW: installation / troubleshooting intents to block
install_keywords = [
    "install", "installation", "set up", "setup", "configure", "configuration",
    "how do i connect", "wiring", "mount it", "mounting steps", "pair", "pairing",
    "firmware", "driver install", "troubleshoot", "troubleshooting", "fix", "repair",
    "update firmware", "how to use", "step by step", "steps"
]

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
def _mtime(path:str)->float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0

@st.cache_resource
def load_vector_store(index_name:str, embed_model:str, pinecone_key:str, openai_key:str):
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model=embed_model, api_key=openai_key)
    return PineconeVectorStore(index=index, embedding=embeddings)

@st.cache_resource
def load_llm(chat_model:str, temperature:float, openai_key:str):
    os.environ["OPENAI_API_KEY"] = openai_key  # be explicit
    return ChatOpenAI(model=chat_model, temperature=temperature)

@st.cache_resource
def load_categorical_values(mtime:float):
    path = "documents/categorical_values.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for k in ["category","subcategory","material","fiberduplex","fibertype","material_tags","color","wireless",
                      "interface","mounting_options"]:
                data.setdefault(k, [])
            return data
    return {
        "category": [], "subcategory": [], "material": [], "fiberduplex": [], "fibertype": [],
        "material_tags": [], "color": [], "wireless": [], "interface": [], "mounting_options": []
    }

@st.cache_resource
def load_sku_vocab(mtime:float):
    path = "documents/sku_vocab.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            skus = [s.upper() for s in data.get("skus", [])]
            sku_set = set(skus)
            sku_map_nohyphen = {}
            for s in skus:
                sku_map_nohyphen.setdefault(s.replace("-", ""), s)
            return sku_set, sku_map_nohyphen
    except Exception:
        return set(), {}

vector_store = load_vector_store(INDEX_NAME, EMBED_MODEL, PINECONE_API_KEY, OPENAI_API_KEY)
llm = load_llm(CHAT_MODEL, TEMP, OPENAI_API_KEY)
categorical_values = load_categorical_values(_mtime("documents/categorical_values.json"))
sku_set, sku_map_nohyphen = load_sku_vocab(_mtime("documents/sku_vocab.json"))

# ---------- utils ----------
def norm_text(s):
    return s.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def extract_product_numbers(text):
    """Return all SKU mentions using vocab-first, hyphen-insensitive matching."""
    if not text: return []
    txt = text.upper()
    cands = re.findall(r"[A-Z0-9-]{3,}", txt)
    out, seen = [], set()
    for cand in cands:
        match = None
        if cand in sku_set:
            match = cand
        else:
            ch = cand.replace("-", "")
            match = sku_map_nohyphen.get(ch)
        if match and match not in seen:
            seen.add(match)
            out.append(match)
    return out

def extract_product_number(text):
    arr = extract_product_numbers(text)
    return arr[0] if arr else None

def is_vague_follow_up(prompt):
    if extract_product_number(prompt):  # explicit product ref => not vague
        return False
    clean = norm_text(prompt)
    return len(clean.split()) <= 6 or any(k in clean for k in fallback_keywords)

def is_farewell(prompt):
    p = norm_text(prompt)
    return any(k in p for k in farewell_keywords)

def is_install_request(prompt):
    p = norm_text(prompt)
    return any(k in p for k in install_keywords)

def try_match_categorical(meta_key: str, prompt_norm: str):
    values = categorical_values.get(meta_key, [])
    for v in values:
        v_raw = str(v).strip().lower()
        v_norm = norm_text(v_raw)
        if v_norm and v_norm in prompt_norm:
            return v_raw
    matches = get_close_matches(prompt_norm, [str(v).strip().lower() for v in values], n=1, cutoff=0.85)
    return matches[0] if matches else None

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

# --------- Step 6: compare/disambiguation helpers ----------
def _summarize_doc(doc):
    md = doc.metadata or {}
    return {
        "Product Number": md.get("product_number", ""),
        "Category": md.get("category", ""),
        "Subcategory": md.get("subcategory", ""),
        "Material": md.get("material", ""),
        "Color": md.get("color", ""),
        "Ports": md.get("ports", ""),
        "Displays": md.get("displays", ""),
        "Ethernet Speed": md.get("ethernet speed", ""),
        "Max Distance": md.get("max distance", ""),
        "Fiber Duplex": md.get("fiberduplex", ""),
        "Fiber Type": md.get("fibertype", ""),
        "Wireless": md.get("wireless", ""),
        "Interface": md.get("interface", ""),
        "Mounting": md.get("mounting_options", ""),
    }

def _markdown_table(rows:list[dict], cols:list[str])->str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"]*len(cols)) + " |"
    body_lines = []
    for r in rows:
        body_lines.append("| " + " | ".join(str(r.get(c, "") or "") for c in cols) + " |")
    return "\n".join([header, sep] + body_lines)

def show_compare_for_products(pnums:list[str], prompt_hint:str=""):
    docs = []
    for pn in pnums:
        try:
            res = vector_store.similarity_search(query="product spec", k=1, filter={"product_number": pn})
            if res:
                docs.append(res[0])
        except Exception as e:
            print(f"Compare fetch failed for {pn}: {e}")
    if not docs:
        return False
    rows = [_summarize_doc(d) for d in docs]
    cols = ["Product Number","Category","Subcategory","Ports","Displays","Ethernet Speed","Max Distance","Material","Color","Interface","Mounting"]
    table_md = _markdown_table(rows, cols)
    with st.chat_message("assistant"):
        st.markdown("I found multiple relevant products. Here’s a quick compare:")
        st.markdown(table_md)
        st.markdown("**Reply with a product number** to dive deeper.")
    st.session_state.messages.append(AIMessage(f"Compare candidates shown: {', '.join([r['Product Number'] for r in rows])}. {prompt_hint}"))
    # mark that we are waiting for a user's SKU pick
    st.session_state.pending_compare = True
    return True

# --------- Step 7: structured answer rendering + safety ---------
def render_structured_answer(prompt_text:str)->str:
    pn = st.session_state.last_product_number or "(unknown)"
    messages = (
        st.session_state.messages +
        [SystemMessage(
            "SCOPE: You are a StarTech.com product assistant. "
            "You ONLY provide product information, specs, compatibility, and high-level recommendations. "
            "You DO NOT provide installation, configuration, wiring, firmware, or troubleshooting steps. "
            "If asked for those, politely decline and recommend contacting StarTech.com Technical Support."
        )] +
        [SystemMessage(
            f"CONTENT SAFETY: Answer ONLY about the single StarTech.com product number {pn}. "
            "Do not mention, invent, or guess other product numbers or product names. "
            "Use ONLY facts present in the SPECIFICATION block below; if a fact is missing, say: "
            "'That information isn't in the spec I have.'"
        )] +
        [SystemMessage(
            "FORMAT: Return Markdown. Start each heading on its own line:\n"
            "Product: <PRODUCT NUMBER>\n"
            "### Overview\n"
            "### Key specs\n"
            "### In the box (only if present)\n"
            "### Notes (caveats/limits if present)\n"
            "Do NOT reference non-StarTech brands."
        )] +
        [SystemMessage(f"SPECIFICATION:\n{st.session_state.last_context}")] +
        [HumanMessage(prompt_text)]
    )
    return llm.invoke(messages).content

# --- Markdown tidy so '###' never appears inline ---
def fix_markdown_headings(md: str) -> str:
    md = re.sub(r'(Product:\s*[^\n]+)\s+###\s*', r'\1\n\n### ', md, flags=re.IGNORECASE)
    md = re.sub(r'(?<!\n)###\s+', r'\n\n### ', md)
    return md.strip()

def sanitize_reply(reply:str)->str:
    return reply

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

def handle_install_block(prompt):
    if is_install_request(prompt):
        pn = st.session_state.last_product_number
        base = ("I can help with product selection, specs, and compatibility. "
                "For installation, configuration, or troubleshooting, "
                "please refer to the official documentation or contact StarTech.com Technical Support.")
        if pn:
            show_response(f"{base}\n\nIf you share more about your setup, I can confirm whether **{pn}** fits your needs.")
        else:
            show_response(base)
        return True
    return False

def handle_explicit_product(prompt):
    pnums = extract_product_numbers(prompt)
    if len(pnums) >= 2:
        print(f"Compare mode (explicit): {pnums[:3]}")
        show_compare_for_products(pnums[:3], prompt_hint="Explicit compare.")
        # Do NOT pick a product; clear context and wait for user SKU
        st.session_state.last_product_number = ""
        st.session_state.last_context = ""
        st.session_state.last_score = None
        st.session_state.last_metadata = {}
        return "compare"
    elif len(pnums) == 1:
        pn = pnums[0]
        docs = vector_store.similarity_search(query="product spec", k=5, filter={"product_number": pn})
        if docs:
            st.session_state.last_product_number = pn
            st.session_state.last_context = docs[0].page_content
            st.session_state.last_score = 1.0
            st.session_state.last_metadata = docs[0].metadata or {}
        else:
            st.session_state.last_product_number = ""
            st.session_state.last_context = ""
            st.session_state.last_score = 0.0
            st.session_state.last_metadata = {}
        st.session_state.pending_compare = False
        return "single"
    return False

def handle_vague_follow_up(prompt):
    """Respond to short/ambiguous follow-ups using the last retrieved product context."""
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
    reply = render_structured_answer(prompt)
    reply = sanitize_reply(reply)
    reply = fix_markdown_headings(reply)
    show_response(reply)
    return True

def handle_descriptive_query(prompt):
    prompt_norm = norm_text(prompt)
    f = extract_filter_from_prompt(prompt)
    try:
        results = vector_store.similarity_search_with_relevance_scores(prompt, k=5, filter=f)
        results = [(doc, score) for doc, score in results if score >= 0.35]
        if results:
            uniq = []
            seen = set()
            for d, s in results:
                pn = d.metadata.get("product_number","")
                if pn and pn not in seen:
                    seen.add(pn); uniq.append((d, s))
            if len(uniq) >= 2 and (any(k in prompt_norm for k in compare_keywords) or abs(uniq[0][1] - uniq[1][1]) <= 0.06):
                candidates = [d.metadata.get("product_number","") for d, _ in uniq[:3]]
                print(f"Compare mode (tie/intent): {candidates}")
                show_compare_for_products(candidates, prompt_hint="Close scores/compare intent.")
                # Clear context; wait for user's SKU pick
                st.session_state.last_product_number = ""
                st.session_state.last_context = ""
                st.session_state.last_score = None
                st.session_state.last_metadata = {}
                st.session_state.pending_compare = True
                return "compare"
            return use_top_result(results)
    except Exception as e:
        print(f"Metadata-filtered search failed: {e}")

    results = vector_store.similarity_search_with_relevance_scores(prompt, k=5)
    results = [(doc, score) for doc, score in results if score >= 0.35]
    if results:
        return use_top_result(results)
    return False

def use_top_result(results):
    top_doc, score = results[0]
    st.session_state.last_product_number = top_doc.metadata.get("product_number", "")
    st.session_state.last_context = top_doc.page_content
    st.session_state.last_score = score
    st.session_state.last_metadata = top_doc.metadata or {}
    st.session_state.pending_compare = False
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
if "pending_compare" not in st.session_state: st.session_state.pending_compare = False

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

    # hard guard first: no installation / troubleshooting
    if handle_install_block(prompt): st.stop()

    # cleaned control flow
    if handle_greeting(prompt) or handle_farewell(prompt):
        st.stop()

    handled = handle_explicit_product(prompt)
    if handled == "compare":
        # compare shown; wait for user to pick a product number
        st.stop()

    if not handled:
        if is_vague_follow_up(prompt):
            if handle_vague_follow_up(prompt):
                st.stop()
        else:
            status = handle_descriptive_query(prompt)
            if status == "compare":
                st.stop()

    # If we are pending a compare choice, don't show generic fallback or render anything
    if st.session_state.pending_compare:
        st.stop()

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

    # SKU debug (prompt)
    cand_prompt = re.findall(r"[A-Z0-9-]{3,}", prompt.upper())
    unrec_prompt = []
    for c in cand_prompt:
        if c in sku_set: continue
        ch = c.replace("-", "")
        if ch not in sku_map_nohyphen:
            unrec_prompt.append(c)
    if unrec_prompt:
        print("Unrecognized SKU-like tokens (prompt):", sorted(set(unrec_prompt))[:20])

    md = st.session_state.get("last_metadata") or {}
    interesting_keys = (
        "category", "subcategory", "material", "material_tags",
        "fiberduplex", "fibertype", "ports", "displays", "color", "cablelength", "wireless",
        "interface", "mounting_options"
    )

    print("Resolved Metadata (top doc):")
    for k in interesting_keys:
        if k in md:
            print(f"- {k}: {md[k]}")

    print("Context Retrieved:")
    print(st.session_state.last_context)
    print("-------------------\n")

    # As a safety net, don't render if we're somehow still pending a compare
    if st.session_state.pending_compare:
        st.stop()

    # Structured answer + heading fix
    reply = render_structured_answer(prompt)
    reply = sanitize_reply(reply)
    reply = fix_markdown_headings(reply)
    show_response(reply)
