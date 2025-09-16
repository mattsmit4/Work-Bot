import streamlit as st
import os, json, re, string
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from difflib import get_close_matches, SequenceMatcher

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
    "how fast","what speed","what resolution","is this compatible","is this supported","will this work","tell me more",
    "can i use this","can it","is there","do they","what’s included","what is included","what do you get",
    "included accessories","in the box","what’s in the box","what comes with","do i need","will it help",
    "does it require","will it fit","will it keep","what version","any differences","any difference","difference between"
]
farewell_keywords = ["thank you","thanks","appreciate it","cheers","bye","goodbye","see you","you’ve been helpful","you have been helpful","that’s all","that is all","cool"]

# NEW: installation / troubleshooting intents to block
install_keywords = [
    "install", "installation", "set up", "setup", "configure", "configuration",
    "how do i connect", "wiring", "mount it", "mounting steps", "pair", "pairing",
    "firmware", "driver install", "troubleshoot", "troubleshooting", "fix", "repair",
    "update firmware", "how to use", "step by step", "steps"
]

# numeric metadata keywords (exact-value only now)
metadata_field_keywords = {
    # widened ports synonyms
    "ports": [
        "total ports","ports total","ports",
        "number of ports","num of ports","num ports",
        "port count","ports count"
    ],
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

# ---- Length formatting helper (mm -> friendly) ----
# Tightened regex to avoid matching normal words (e.g., 'docking').
_LEN_UNIT = r'\b(?:ft|feet|foot|in(?:ch(?:es)?)?|cm|centimeter(?:s)?|centimetre(?:s)?|m|meter(?:s)|metre(?:s)?)\b'
# Number + unit detector (e.g., "6 ft", "1m", "12 inches")
_NUM_UNIT = rf'\b\d+(?:\.\d+)?\s*{_LEN_UNIT}'

# Detect common comparator phrases (incl. typos) OR raw symbols.
_COMPARATOR_RE = re.compile(
    r'(?:'
    r'\bunder\b|\bbelow\b|'
    r'\bless\s+(?:than|then)\b|'
    r'\bno\s+less\s+(?:than|then)\b|\bnot\s+less\s+(?:than|then)\b|'
    r'\bat\s*least\b|\batleast\b|\bmin(?:imum)?(?:\s+of)?\b|\bmin\b|'
    r'\bgreater\s+(?:than|then)\b|\bmore\s+(?:than|then)\b|\bover\b|\babove\b|'
    r'\bat\s*most\b|\batmost\b|\bno\s+more\s+(?:than|then)\b|\bnot\s+more\s+(?:than|then)\b|'
    r'\bmax(?:imum)?(?:\s+of)?\b|\bmax\b|\bup\s*to\b|\bupto\b|'
    r'\bbetween\b|\bfrom\b|\bthrough\b|\bthru\b|'
    r'\bexact(?:ly)?\b|\bequal(?:s)?(?:\s+to)?\b'
    r'|[<>]=?'
    r')'
)

def is_vague_follow_up(prompt):
    if extract_product_number(prompt):
        return False
    p = prompt.lower()
    # Any number, real number+unit, or comparator phrase => not vague
    if (re.search(r'\d', p) or
        re.search(_NUM_UNIT, p) or
        _COMPARATOR_RE.search(p)):
        return False
    clean = norm_text(prompt)
    return len(clean.split()) <= 6 or any(k in clean for k in fallback_keywords)

def is_farewell(prompt):
    p = norm_text(prompt)
    return any(k in p for k in farewell_keywords)

def is_install_request(prompt):
    p = norm_text(prompt)
    return any(k in p for k in install_keywords)

# ---------- fuzzy categorical matching (tightened) ----------
_FUZZY_KEYS = {"category", "subcategory"}
_STOPWORDS = {
    "a","an","the","do","does","did","you","your","yours","have","has","had","any","that","this","these","those",
    "can","could","may","might","will","would","shall","should","more","most","then","than","of","for","to","in",
    "on","with","without","and","or","but","if","it","its","is","are","was","were","be","being","been","at","by",
    "about","from","as","up","over","under","between"
}

def _lemma(tok: str) -> str:
    tok = tok.lower()
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok

def _tokset(s: str):
    return { _lemma(w) for w in norm_text(s).split() if w and w not in _STOPWORDS }

def _meaningful_ngrams(prompt_norm: str):
    words = [w for w in prompt_norm.split() if w and w not in _STOPWORDS]
    grams = []
    # multi-word n-grams (2..4)
    for n in range(2, min(4, len(words)) + 1):
        for i in range(len(words) - n + 1):
            grams.append(" ".join(words[i:i+n]))
    # single words of length >= 4
    grams.extend([w for w in words if len(w) >= 4])
    return grams

def _fuzzy_pick_for_category(values: list[str], prompt_norm: str):
    grams = _meaningful_ngrams(prompt_norm)
    if not grams:
        return None
    best_val, best_score = None, 0.0

    for v in values:
        v_norm = norm_text(v)
        vset = _tokset(v_norm)
        if not vset:
            continue
        for g in grams:
            gset = _tokset(g)
            if not gset:
                continue
            overlap = len(vset & gset)
            if overlap == 0:
                continue
            cover_v = overlap / len(vset)          # how much of candidate covered by prompt
            cover_g = overlap / len(gset)          # how much of n-gram covered by candidate
            char_sim = SequenceMatcher(None, v_norm, g).ratio()

            # tight rule: 80% of candidate tokens present AND (60% of gram OR strong char sim)
            if cover_v >= 0.80 and (cover_g >= 0.60 or char_sim >= 0.86):
                score = 0.6*cover_v + 0.2*cover_g + 0.2*char_sim
                if score > best_score:
                    best_score, best_val = score, v
    return best_val

def try_match_categorical(meta_key: str, prompt_norm: str):
    values = [str(v).strip() for v in categorical_values.get(meta_key, []) if str(v).strip()]
    if not values:
        return None

    # Tight fuzzy only for category/subcategory
    if meta_key in _FUZZY_KEYS:
        hit = _fuzzy_pick_for_category(values, prompt_norm)
        if hit:
            return hit

    # Strict behavior for others (and fallback if fuzzy failed)
    for v in values:
        v_norm = norm_text(v)
        if v_norm and v_norm in prompt_norm:
            return v
    matches = get_close_matches(prompt_norm, [norm_text(v) for v in values], n=1, cutoff=0.85)
    if matches:
        idx = [norm_text(v) for v in values].index(matches[0])
        return values[idx]
    return None

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

def _to_mm(value: float, unit: str | None) -> float:
    if not unit:
        unit = "ft"  # sensible default for cable length
    unit = unit.lower()
    if unit in ("ft", "feet", "foot"):
        return value * 304.8
    if unit in ("in", "inch", "inches"):
        return value * 25.4
    if unit in ("cm", "centimeter", "centimeters", "centimetre", "centimetres"):
        return value * 10.0
    if unit in ("m", "meter", "meters", "metre", "metres"):
        return value * 1000.0
    return value  # assume already mm

def _pretty_mm(mm: float) -> str:
    try:
        mm = float(mm)
    except Exception:
        return str(mm)
    if mm <= 300:
        inches = mm / 25.4
        cm = mm / 10.0
        val = round(inches, 1)
        return (str(int(val)) if val.is_integer() else str(val)) + f" in [{int(round(cm))} cm]"
    feet = mm / 304.8
    meters = mm / 1000.0
    f = round(feet, 1)
    m = round(meters, 1)
    f_s = str(int(f)) if f.is_integer() else str(f)
    m_s = str(int(m)) if m.is_integer() else str(m)
    return f"{f_s} ft [{m_s} m]"

def _len_tol(mm: float) -> float:
    """Tolerance for cable length matching (in mm). Use the larger of ±25mm or ±2%."""
    try:
        mm = float(mm)
    except Exception:
        return 0.0
    return max(25.0, mm * 0.02)

def _satisfies_numeric(meta: dict, flt: dict) -> bool:
    """Client-side check that a doc meets all numeric constraints (ranges + equality)."""
    def _to_num(x):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            m = re.search(r'[-+]?\d+(?:\.\d+)?', x)
            if m:
                try:
                    return float(m.group(0))
                except:
                    return None
        return None

    for field, cond in (flt or {}).items():
        v = meta.get(field)
        v_num = _to_num(v)
        if v_num is None:
            return False

        if isinstance(cond, dict):
            if "$eq" in cond:
                target = _to_num(cond["$eq"])
                if target is None or v_num != target:
                    return False
                continue

            if "$lt" in cond:
                t = _to_num(cond["$lt"])
                if t is None or not (v_num < t):
                    return False
            if "$lte" in cond:
                t = _to_num(cond["$lte"])
                if t is None or not (v_num <= t):
                    return False
            if "$gt" in cond:
                t = _to_num(cond["$gt"])
                if t is None or not (v_num > t):
                    return False
            if "$gte" in cond:
                t = _to_num(cond["$gte"])
                if t is None or not (v_num >= t):
                    return False

            if not any(k in cond for k in ("$lt", "$lte", "$gt", "$gte", "$eq")):
                return False

        else:
            target = _to_num(cond)
            if target is None or v_num != target:
                return False

    return True

def parse_global_range(prompt_norm: str):
    """Parse general range language into Pinecone range ops for unit-less numerics."""
    # between / from ... to|through|thru|-
    m = re.search(r'(?:between|from)\s*(\d+(?:\.\d+)?)\s*(?:and|to|through|thru|-)\s*(\d+(?:\.\d+)?)', prompt_norm)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return {"$gte": lo, "$lte": hi}

    # <= family
    m = re.search(
        r'(?:<=|=<|less\s+than\s+or\s+equal\s+to|at\s*most|atmost|no\s+more\s+(?:than|then)|'
        r'not\s+more\s+(?:than|then)|up\s*to|upto|no\s+greater\s+(?:than|then)|'
        r'max(?:imum)?(?:\s+of)?|max)\s*(\d+(?:\.\d+)?)', prompt_norm)
    if m:
        return {"$lte": float(m.group(1))}

    # < family
    m = re.search(r'(?:<|less\s+(?:than|then)|under|below)\s*(\d+(?:\.\d+)?)', prompt_norm)
    if m:
        return {"$lt": float(m.group(1))}

    # >= family
    m = re.search(
        r'(?:>=|=>|greater\s+than\s+or\s+equal\s+to|at\s*least|atleast|no\s+less\s+(?:than|then)|'
        r'not\s+less\s+(?:than|then)|min(?:imum)?(?:\s+of)?|min)\s*(\d+(?:\.\d+)?)', prompt_norm)
    if m:
        return {"$gte": float(m.group(1))}

    # > family
    m = re.search(r'(?:>|greater\s+(?:than|then)|more\s+(?:than|then)|over|above)\s*(\d+(?:\.\d+)?)', prompt_norm)
    if m:
        return {"$gt": float(m.group(1))}

    return None

# ---- drop-in length filter ----
def _parse_length_filter(prompt: str):
    s = prompt.lower()
    m = re.search(
        rf'(?:between|from)\s*'
        rf'(\d+(?:\.\d+)?)\s*({_LEN_UNIT})?\s*'
        rf'(?:and|to|through|thru|[-–—])\s*'
        rf'(\d+(?:\.\d+)?)\s*({_LEN_UNIT})?',
        s
    )
    if m:
        a = float(m.group(1)); unit_a = m.group(2)
        b = float(m.group(3)); unit_b = m.group(4)
        unit = unit_a or unit_b
        if not unit:
            return None
        lo = _to_mm(min(a, b), unit)
        hi = _to_mm(max(a, b), unit)
        return {"$gte": lo, "$lte": hi}

    m = re.search(
        rf'(\d+(?:\.\d+)?)\s*({_LEN_UNIT})?\s*'
        rf'(?:[-–—]|to|and)\s*'
        rf'(\d+(?:\.\d+)?)\s*({_LEN_UNIT})?',
        s
    )
    if m:
        a = float(m.group(1)); unit_a = m.group(2)
        b = float(m.group(3)); unit_b = m.group(4)
        unit = unit_a or unit_b
        if not unit:
            return None
        lo = _to_mm(min(a, b), unit)
        hi = _to_mm(max(a, b), unit)
        return {"$gte": lo, "$lte": hi}

    m = re.search(rf'(?:<=|less than or equal to|at\s*most|no more than|up to)\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})', s)
    if m:
        return {"$lte": _to_mm(float(m.group(1)), m.group(2))}
    m = re.search(rf'(?:<|less than|under|below)\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})', s)
    if m:
        return {"$lt": _to_mm(float(m.group(1)), m.group(2))}
    m = re.search(rf'(?:>=|greater than or equal to|at\s*least|no less than|minimum of)\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})', s)
    if m:
        return {"$gte": _to_mm(float(m.group(1)), m.group(2))}
    m = re.search(rf'(?:>|greater than|over|more than)\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})', s)
    if m:
        return {"$gt": _to_mm(float(m.group(1)), m.group(2))}
    return None

# --- numeric helper (first number near keywords) ---
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

def extract_filter_from_prompt(prompt):
    """Build filters from the prompt:
       - cablelength: ranges via unit parsing; else exact-with-tolerance (only if length keywords present)
       - other numerics: support 'between/at least/under/over' etc., else exact equality ($eq)
       - categorical: fuzzy only for category/subcategory (80% candidate token coverage)
    """
    prompt_norm = norm_text(prompt)
    filters = {}

    # --- cable length ---
    has_len_kw = any(kw in prompt_norm for kw in metadata_field_keywords["cablelength"])
    mentions_length = has_len_kw or re.search(_NUM_UNIT, prompt.lower()) is not None

    if mentions_length:
        len_cond = _parse_length_filter(prompt)  # range with number+unit if present
        if len_cond:
            filters["cablelength"] = len_cond
        elif has_len_kw:
            m2 = re.search(r'\b(\d+(?:\.\d+)?)\b', prompt_norm)
            if m2:
                mm = _to_mm(float(m2.group(1)), "ft")
                tol = _len_tol(mm)
                filters["cablelength"] = {"$gte": mm - tol, "$lte": mm + tol}

    # --- other numeric fields ---
    rng_any = parse_global_range(prompt_norm)  # one parse covers common phrasing in the prompt
    for field, keywords in metadata_field_keywords.items():
        if field == "cablelength":
            continue
        if any(kw in prompt_norm for kw in keywords):
            if rng_any:
                filters[field] = rng_any
            else:
                val = find_number_near_keywords(prompt_norm, keywords)
                if val is not None:
                    filters[field] = {"$eq": val}

    # --- categorical ---
    for meta_key in ("category", "subcategory", "material", "fiberduplex", "fibertype", "color", "wireless"):
        val = try_match_categorical(meta_key, prompt_norm)
        if val:
            filters[meta_key] = norm_text(val)

    if "wireless" not in filters:
        w = infer_wireless_from_prompt(prompt_norm)
        if w:
            filters["wireless"] = w

    # material tags
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

# --------- Conversational answer rendering + safety ---------
def render_conversational_answer(prompt_text:str)->str:
    pn = st.session_state.last_product_number or "(unknown)"
    messages = (
        st.session_state.messages
        + [SystemMessage(
            "SCOPE: You are a StarTech.com product assistant. "
            "You ONLY provide product information, specs, compatibility, and high-level recommendations. "
            "You DO NOT provide installation, configuration, wiring, firmware, or troubleshooting steps. "
            "If asked for those, politely decline and recommend contacting StarTech.com Technical Support."
        )]
        + [SystemMessage(
            f"CONTENT SAFETY: Answer ONLY about the single StarTech.com product number {pn}. "
            "Do not mention, invent, or guess other product numbers or product names. "
            "Use ONLY facts present in the SPECIFICATION block below; if a fact is missing, say: "
            "'That detail isn’t in the spec I have.'"
        )]
        + [SystemMessage(
            "STYLE: Be conversational and concise like a knowledgeable product specialist. "
            "No section headings or tables. Prefer 3–6 sentences. "
            "Start with: 'For <PRODUCT NUMBER>:' once, then explain. "
            "Answer the user’s question directly using the spec. "
            "If a quick list helps, you may include up to 3–5 short bullets, but only when the user asked for 'specs', 'what’s included', or similar. "
            "If 'Included in Package' exists AND the user asks what's in the box, summarize it inline as 'In the box: ...'. "
            "Avoid emojis. End with a short offer to check another detail if needed."
        )]
        + [SystemMessage(f"SPECIFICATION:\n{st.session_state.last_context}")]
        + [HumanMessage(prompt_text)]
    )
    return llm.invoke(messages).content

def sanitize_reply(reply:str)->str:
    return reply

# ---------- handlers ----------
def handle_greeting(prompt):
    greeting_keywords = ["hello","hi","hey","good morning","good afternoon","good evening"]
    wc = len(prompt.strip().split())
    if not st.session_state.greeted_user and wc <= 4 and any(re.search(rf"\b{g}\b", prompt.lower()) for g in greeting_keywords):
        show_response("Hi there! I’m here to help you find the right StarTech.com product. Tell me what you’re trying to do or ask about a specific product number.")
        st.session_state.greeted_user = True
        return True
    return False

def handle_farewell(prompt):
    if is_farewell(prompt):
        show_response("Thanks for chatting! If you need anything else about StarTech.com products, just ask.")
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
    """If user mentions SKUs explicitly, pick the first one and answer for it."""
    pnums = extract_product_numbers(prompt)
    if len(pnums) >= 1:
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
    reply = render_conversational_answer(prompt)
    reply = sanitize_reply(reply)
    show_response(reply)
    return True

def handle_descriptive_query(prompt, f_override=None):
    """Free-text need. If metadata filters are present, prefer filtered results; otherwise fall back to semantic search.
       When falling back, still enforce numeric constraints client-side."""
    f = f_override if f_override is not None else extract_filter_from_prompt(prompt)

    # 1) Use metadata-filtered search first (use similarity_search here)
    try:
        if f:
            # Filter-only search: the text query doesn't matter much when filters are tight
            docs = vector_store.similarity_search("product spec", k=50, filter=f)
            if docs:
                # wrap in (doc, score) pairs for the existing helper
                return use_top_result([(docs[0], 1.0)])
    except Exception as e:
        print(f"Metadata-filtered search failed: {e}")

    # 2) Unfiltered fallback with a relevance threshold
    results = vector_store.similarity_search_with_relevance_scores(prompt, k=12)
    results = [(doc, score) for doc, score in results if score >= 0.35]
    if not results:
        return False

    # If the user implied numeric constraints, try to honor them client-side
    if f:
        for d, s in results:
            if _satisfies_numeric(d.metadata or {}, f):
                return use_top_result([(d, s)])

        show_response("I couldn’t find a product that meets that requirement. If you can adjust it, I’ll try again.")
        return "no-match"

    # No filter → just take the top result
    return use_top_result(results)

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

    # hard guard first: no installation / troubleshooting
    if handle_install_block(prompt): st.stop()

    if handle_greeting(prompt) or handle_farewell(prompt):
        st.stop()

    handled = handle_explicit_product(prompt)
    if not handled:
        if is_vague_follow_up(prompt):
            if handle_vague_follow_up(prompt):
                st.stop()
        else:
            # --- ALWAYS compute filters up front for debugging ---
            active_filters = extract_filter_from_prompt(prompt)

            status = handle_descriptive_query(prompt, f_override=active_filters)

            # --- ALWAYS print debug, even on no-match/false ---
            print("\n--- Debug Info ---")
            print(f"Active Filters: {active_filters}")
            print(f"User Prompt: {prompt}")
            print(f"Product Number: {st.session_state.get('last_product_number') or ''}")
            print(f"Similarity Score: {st.session_state.get('last_score') if st.session_state.get('last_score') is not None else 'N/A'}")

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
            print(st.session_state.get("last_context") or "(none)")
            print("-------------------\n")

            # Now handle status outcomes
            if status == "no-match":
                st.stop()
            elif status:
                pass
            else:
                show_response(
                    "Could you tell me a bit more about what you're looking for? For example:\n"
                    "- Is there a specific problem you are trying to solve?\n"
                    "- What specs in a product are important to you?\n"
                    "- What do you want the product to connect to or be compatible with?\n\n"
                    "You can also mention a product number if you already have one in mind."
                )
                st.stop()

    if not st.session_state.last_context:
        show_response(
            "I couldn’t find a specific match yet. If you share a product number or more detail (e.g., interface, length, color), I’ll pull the exact specs."
        )
        st.stop()

    # (We keep a final debug print here too, for successful paths)
    final_filters = extract_filter_from_prompt(prompt)
    print("\n--- Debug Info ---")
    print(f"Active Filters: {final_filters}")
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

    reply = render_conversational_answer(prompt)
    reply = sanitize_reply(reply)
    show_response(reply)
