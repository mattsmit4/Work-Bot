# app.py
import streamlit as st
import os, json, re, string, csv
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from difflib import get_close_matches, SequenceMatcher

load_dotenv()
st.title("StarTech.com Products Chatbot")

# --- hydrate env from Streamlit Cloud Secrets (without overwriting local .env) ---
try:
    for k, v in st.secrets.items():
        os.environ.setdefault(str(k), str(v))
except Exception:
    pass

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

# -------------------- NEW: Conversation Logging Function --------------------
def log_to_csv(user_message, bot_response, products_shown=None):
    """Enhanced logging with quality indicators"""
    try:
        import uuid
        
        file_exists = os.path.exists('conversations.csv')
        
        with open('conversations.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow([
                    'Timestamp',
                    'Session ID',
                    'Query Number',
                    'User Message',
                    'Bot Response',
                    'Product Number',
                    'Category',
                    'Subcategory',
                    'Similarity Score',
                    'Response Type',
                    'Match Status',
                    'Filters Applied',
                    'Results Count'
                ])
            
            # Session tracking
            if 'session_id' not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())[:8]  # Short ID
            if 'query_count' not in st.session_state:
                st.session_state.query_count = 0
            if 'last_query_time' not in st.session_state:
                st.session_state.last_query_time = datetime.now()
            
            # Increment query count
            st.session_state.query_count += 1
            
            # Check if this is a new session (more than 30 min since last query)
            time_diff = (datetime.now() - st.session_state.last_query_time).total_seconds()
            if time_diff > 1800:  # 30 minutes
                st.session_state.session_id = str(uuid.uuid4())[:8]
                st.session_state.query_count = 1
            
            st.session_state.last_query_time = datetime.now()
            
            # Determine match status
            if st.session_state.get('last_product_number'):
                match_status = 'success'
            elif 'no match' in bot_response.lower() or 'couldn\'t find' in bot_response.lower():
                match_status = 'no-match'
            else:
                match_status = 'other'
            
            # Get filters and results count
            filters_applied = json.dumps(st.session_state.get('last_filters_applied', {}))
            results_count = st.session_state.get('results_count', 0)
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Write row(s)
            if products_shown and isinstance(products_shown, list):
                for product in products_shown:
                    writer.writerow([
                        timestamp,
                        st.session_state.session_id,
                        st.session_state.query_count,
                        user_message,
                        bot_response,
                        product.get('product_number', ''),
                        product.get('category', ''),
                        product.get('subcategory', ''),
                        product.get('score', ''),
                        'multi-product',
                        match_status,
                        filters_applied,
                        len(products_shown)
                    ])
            else:
                writer.writerow([
                    timestamp,
                    st.session_state.session_id,
                    st.session_state.query_count,
                    user_message,
                    bot_response,
                    st.session_state.get('last_product_number', ''),
                    st.session_state.get('last_metadata', {}).get('category', ''),
                    st.session_state.get('last_metadata', {}).get('subcategory', ''),
                    st.session_state.get('last_score', ''),
                    'single-product' if st.session_state.get('last_product_number') else 'no-product',
                    match_status,
                    filters_applied,
                    results_count
                ])
            
    except Exception as e:
        print(f"Error logging conversation: {e}")

# -------------------- keywords --------------------
fallback_keywords = [
    "what color","what colours","what colour","what type","what kind","how big","how small","how long",
    "how wide","how tall","how thick","how heavy","how many","how much","what size","what sizes","does it",
    "is it","is it compatible","are they","specs","details","specifications","tech specs","technical details",
    "what ports","which ports","what connectors","which connectors","what inputs","what outputs",
    "how fast","what speed","what resolution","is this compatible","is this supported","will this work","tell me more",
    "can i use this","can it","is there","do they","what's included","what is included","what do you get",
    "included accessories","in the box","what's in the box","what comes with","do i need","will it help",
    "does it require","will it fit","will it keep","what version","any differences","any difference","difference between"
]
farewell_keywords = ["thank you","thanks","appreciate it","cheers","bye","goodbye","see you","you've been helpful","you have been helpful","that's all","that is all","cool"]

install_keywords = [
    "install", "installation", "set up", "setup", "configure", "configuration",
    "how do i connect", "wiring", "mount it", "mounting steps", "pair", "pairing",
    "firmware", "driver install", "troubleshoot", "troubleshooting", "fix", "repair",
    "update firmware", "how to use", "step by step", "steps"
]

# -------------------- Synonym & Abbreviation Expansion --------------------
SYNONYMS = {
    # Connector abbreviations
    "dp": "displayport",
    "tb": "thunderbolt",
    "tb3": "thunderbolt 3",
    "tb4": "thunderbolt 4",
    
    # Cable types
    "cat5": "category 5 ethernet",
    "cat5e": "category 5e ethernet",
    "cat6": "category 6 ethernet",
    "cat6a": "category 6a ethernet",
    "cat7": "category 7 ethernet",
    
    # Video standards
    "4k": "3840x2160",
    "4k60": "4k 60hz",
    "1080p": "1920x1080",
    "720p": "1280x720",
    
    # Common phrases
    "charger": "charging cable",
    "power cord": "power cable",
    "monitor cable": "display cable",
    "laptop dock": "docking station",
    
    # Material variations
    "braided": "nylon braided",
    "aluminum": "aluminium",
    
    # Feature abbreviations
    "poe": "power over ethernet",
    "4k support": "4k display support",
}

def expand_synonyms(text: str) -> str:
    """
    Expand common abbreviations and synonyms in user queries.
    Returns the expanded text with synonyms replaced.
    """
    text_lower = text.lower()
    expanded = text_lower
    
    # Sort by length (longest first) to handle multi-word synonyms
    for abbr, full in sorted(SYNONYMS.items(), key=lambda x: -len(x[0])):
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(abbr) + r'\b'
        expanded = re.sub(pattern, full, expanded)
    
    return expanded

# Treat "kvm ports" as the canonical Ports field; add per-connector derived fields too
metadata_field_keywords = {
    "ports": [
        "total ports","ports total","ports",
        "number of ports","num of ports","num ports",
        "port count","ports count",
        "kvm ports","ports kvm"
    ],
    "packqty": ["in the pack","in the package","package quantity","pack qty","in a pack","in a package"],
    "displays": ["displays","monitors","screens","number of displays"],
    "numharddrive": ["hard drive","hard drives"],
    "cablelength": ["cable length","length of cable","cablelength","cord length"],

    # NEW: per-connector numeric filters (populated by ingestion metadata)
    "usb_c_ports": ["usb-c ports","usbc ports","usb c ports","type-c ports","type c ports"],
    "usb_a_ports": ["usb-a ports","usb a ports","usb ports","type-a ports","type a ports"],
    "hdmi_ports":  ["hdmi port","hdmi ports"],
    "dp_ports":    ["displayport","display port","dp port","dp ports"],
    "vga_ports":   ["vga port","vga ports"]
}

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
    return ChatOpenAI(model=chat_model, temperature=temperature, api_key=openai_key)

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

def norm_text(s):
    return s.lower().translate(str.maketrans('', '', string.punctuation)).strip()

def extract_product_numbers(text):
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

# ---- Length patterns ----
_LEN_UNIT = r'\b(?:ft|feet|foot|in(?:ch(?:es)?)?|cm|centimeter(?:s)?|centimetre(?:s)?|m|meter(?:s)|metre(?:s)?)\b'
_NUM_UNIT = rf'\b\d+(?:\.\d+)?\s*{_LEN_UNIT}'
_LEN_UNIT_NOB = r'(?:ft|feet|foot|in(?:ch(?:es)?)?|cm|centimeter(?:s)?|centimetre(?:s)?|m|meter(?:s)|metre(?:s)?)'
_NUM_UNIT_NOB = rf'\d+(?:\.\d+)?\s*{_LEN_UNIT_NOB}'

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
    r'|[<>]=?|\u2264|\u2265'
    r')'
)

_NUM_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12
}
_NUM_WORDS_RE = re.compile(r'\b(' + "|".join(map(re.escape,_NUM_WORDS.keys())) + r')\b', flags=re.I)

def word_to_int(w: str):
    w = w.lower().strip()
    return _NUM_WORDS.get(w)

_DOMAIN_SEEK_TOKENS = {
    "adapter","adapters","hdmi","displayport","dp","dvi","vga","dock","docking","enclosure","kvm","switch","cable","hub",
    "two-pack","2-pack","three-pack","3-pack","pack","bundle"
}

_FUZZY_KEYS = {"category", "subcategory", "interface"}
_STOPWORDS = {
    "a","an","the","do","does","did","you","your","yours","have","has","had","any","that","this","these","those",
    "can","could","may","might","will","would","shall","should","more","most","then","than","of","for","to","in",
    "on","with","without","and","or","but","if","it","its","is","are","was","were","be","being","been","at","by",
    "about","from","as","up","over","under","between", "other","others","detail","details"
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
    for n in range(2, min(4, len(words)) + 1):
        for i in range(len(words) - n + 1):
            grams.append(" ".join(words[i:i+n]))
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
            cover_v = overlap / len(vset)
            cover_g = overlap / len(gset)
            char_sim = SequenceMatcher(None, v_norm, g).ratio()
            if cover_v >= 0.80 and (cover_g >= 0.60 or char_sim >= 0.86):
                score = 0.6*cover_v + 0.2*cover_g + 0.2*char_sim
                if score > best_score:
                    best_score, best_val = score, v
    return best_val

def try_match_categorical(meta_key: str, prompt_norm: str):
    values = [str(v).strip() for v in categorical_values.get(meta_key, []) if str(v).strip()]
    if not values:
        return None
    if meta_key in _FUZZY_KEYS:
        hit = _fuzzy_pick_for_category(values, prompt_norm)
        if hit:
            return hit
    for v in values:
        v_norm = norm_text(v)
        if v_norm and v_norm in prompt_norm:
            return v
    matches = get_close_matches(prompt_norm, [norm_text(v) for v in values], n=1, cutoff=0.85)
    if matches:
        idx = [norm_text(v) for v in values].index(matches[0])
        return values[idx]
    return None

# ---------- Intent-aware category/subcategory extraction ----------
_DISPLAY_HINTS = {"monitor","monitors","display","displays","screen","screens","hdmi","displayport","dp","dvi","vga","dock","docking","multi","adapter"}
_NETWORK_HINTS = {"network","ethernet","rj45","lan","poe","gigabit","10gbe","nic"}
_STORAGE_HINTS = {"enclosure","bay","bays","drive","drives","hdd","ssd","sata","nvme"}
_SECURITY_HINTS = {"lock","locks","kensington","security","anti-theft","antitheft"}

GENERIC = {"laptop","pc","computer","something","need","looking","connect","connection","product","device","thing"}

def _contains_any_token(text_norm: str, words: set[str]) -> bool:
    toks = set(re.findall(r"[a-z0-9]+", text_norm))
    return any(w in toks for w in words)

def pick_categories_from_prompt(prompt_text: str):
    s_raw = prompt_text.lower()
    
    # Handle compound terms BEFORE general replacements
    # Replace "USB-C" / "USB C" with a single token
    s_raw = re.sub(r'\busb[\s\-]?c\b', 'usbc', s_raw, flags=re.I)
    s_raw = re.sub(r'\btype[\s\-]?c\b', 'typec', s_raw, flags=re.I)
    
    # Now do general replacements
    s_raw = s_raw.replace("wi-fi","wifi").replace("wi fi","wifi")
    
    print(f"DEBUG s_raw after replacements: {s_raw}")  # ← ADD THIS
    
    tokens = [t for t in re.findall(r"[a-z0-9]+", s_raw) if t not in _STOPWORDS and len(t) >= 3 and t not in GENERIC]
    print(f"DEBUG tokens extracted: {tokens}")  # ← ADD THIS
    
    tokens_lem = {_lemma(t) for t in tokens}
    print(f"DEBUG tokens_lem: {tokens_lem}")  # ← ADD THIS

    cats_src = categorical_values.get("category", [])
    subs_src = categorical_values.get("subcategory", [])

    # NEW CODE - returns lowercase to match Pinecone metadata
    def word_overlap_hits(pool):
        hits = []
        for v in pool:
            vn = norm_text(v)  # Normalize for matching
            vwords = _tokset(vn)
            if vwords & tokens_lem:
                hits.append(v.lower())  # ← Return lowercase to match Pinecone!
        return sorted(list(dict.fromkeys(hits)))

    cat_hits = word_overlap_hits(cats_src)
    sub_hits = word_overlap_hits(subs_src)
    
    # ========== NEW: SMART FILTERING ==========
    meaningful_tokens = tokens_lem - {"show", "cabl", "cable", "need", "want", "find", "get"}

    if len(sub_hits) > 5 and meaningful_tokens:
        # Find subcategories that contain ALL meaningful tokens
        exact_matches = []
        for s in sub_hits:
            s_tokens = _tokset(s)
            if meaningful_tokens.issubset(s_tokens):
                exact_matches.append(s)
        
        if exact_matches:
            sub_hits = exact_matches
        else:
            # Fallback: find subcategories with MOST token overlap
            scored = []
            for s in sub_hits:
                s_tokens = _tokset(s)
                overlap_count = len(meaningful_tokens & s_tokens)
                if overlap_count > 0:
                    scored.append((s, overlap_count))
            
            if scored:
                scored.sort(key=lambda x: -x[1])
                max_overlap = scored[0][1]
                sub_hits = [s for s, count in scored if count == max_overlap]
    # ========== END SMART FILTERING ==========

    print(f"DEBUG cat_hits: {cat_hits}")  # ← ADD THIS
    print(f"DEBUG sub_hits: {sub_hits}")  # ← ADD THIS

    display_intent    = _contains_any_token(s_raw, _DISPLAY_HINTS)
    networking_intent = _contains_any_token(s_raw, _NETWORK_HINTS)
    security_intent   = _contains_any_token(s_raw, _SECURITY_HINTS)

    if display_intent and not security_intent:
        sub_hits = [s for s in sub_hits if not re.search(r'\block(s)?\b|security', s)]
    if display_intent and not networking_intent:
        cat_hits = [c for c in cat_hits if "network" not in c]
        sub_hits = [s for s in sub_hits if "network" not in s]

    # strip organizer/accessory style subs
    generic_bad = ("organizer", "organisers", "fastener", "accessor")
    sub_hits = [s for s in sub_hits if not any(b in s for b in generic_bad)]

    # ========== NEW: Don't over-filter with single subcategory ==========
    # If we only have 1 subcategory match but it's very specific,
    # drop it and just use the category for a broader search
    if len(sub_hits) == 1 and len(cat_hits) >= 1:
        # Check if the subcategory is overly specific
        specific_indicators = ["drive", "usb-a", "usb-c", "thunderbolt", "type-c", "displayport", "hdmi"]
        subcat_lower = sub_hits[0].lower()
        
        # If it contains specific tech terms, drop the subcategory filter
        if any(indicator in subcat_lower for indicator in specific_indicators):
            print(f"DEBUG: Dropping overly specific subcategory '{sub_hits[0]}' to broaden search")
            sub_hits = []
    # ====================================================================

    # NEW CODE:
    if len(cat_hits) + len(sub_hits) == 0:
        return None
        
    # If we have too many hits, prioritize subcategory over category
    if len(sub_hits) > 20:
        # Keep only subcategories that exactly match the main token
        main_tokens = {"usbc", "hdmi", "displayport", "ethernet", "thunderbolt", "dp", "dvi", "vga"}
        exact_sub_hits = [s for s in sub_hits if any(tok in s for tok in (tokens_lem & main_tokens))]
        if exact_sub_hits:
            sub_hits = exact_sub_hits[:10]  # Limit to top 10
        else:
            sub_hits = sub_hits[:10]  # Fallback: just take first 10

    if len(cat_hits) > 10:
        cat_hits = cat_hits[:10]  # Limit to top 10

    out = {}
    if cat_hits:
        out["category"] = {"$in": cat_hits}

    # ========== NEW: Only add subcategory if it's genuinely helpful ==========
    # Don't add subcategory filter for broad exploratory queries
    # Only add it when:
    # 1. User is being specific (3+ meaningful tokens), OR
    # 2. Multiple subcategories match (shows user intent is clear)
    if sub_hits:
        # Count meaningful tokens (excluding stopwords and generic terms)
        query_tokens = tokens_lem - {"show", "cabl", "cable", "need", "want", "find", "get", "have", "any"}
        
        # Only apply subcategory filter if:
        # - User query has 3+ specific tokens (e.g., "usb-c thunderbolt docking station"), OR
        # - Multiple subcategories matched (shows clear intent), OR
        # - Only 1 category but multiple subcategories (refinement needed)
        if len(query_tokens) >= 3 or len(sub_hits) >= 3 or (len(cat_hits) == 1 and len(sub_hits) >= 2):
            out["subcategory"] = {"$in": sub_hits}
            print(f"DEBUG: Applying subcategory filter with {len(sub_hits)} options")
        else:
            print(f"DEBUG: Skipping subcategory filter for broad query (only {len(query_tokens)} specific tokens)")
    # ====================================================================

    return out or None

# -------------------- Anchor map with dynamic category generation --------------------
_ANCHORS = [
    (["kvm"],               "kvm",              {"sub_token": "kvm",        "forbid": ["dock", "enclosure"]}),
    (["dock","docking"],    "dock",             {"sub_token": "dock",       "forbid": ["enclosure","drive bay","drive bays","drive docking","cable","adapter"]}),
    (["enclosure","bay","bays","drive bay","drive bays"], "enclosure", {"sub_token": "enclosure",  "forbid": ["dock"]}),
    (["rack","cabinet"],    "rack",             {"sub_token": "rack",       "forbid": ["dock"]}),
]

def detect_anchor_rules(prompt_norm: str):
    for tokens, key, extras in _ANCHORS:
        if any(t in prompt_norm for t in tokens):
            return {"key": key, **extras}
    return None

def _category_values_containing(token: str):
    token = token.strip().lower()
    vals = [v for v in (categorical_values.get("category") or []) if token in (v or "").lower()]
    return sorted(list(dict.fromkeys([norm_text(v) for v in vals])))

def build_subcategory_allowlist(sub_token: str, forbid: list[str]):
    subs = [norm_text(v) for v in categorical_values.get("subcategory", [])]
    allow = []
    for s in subs:
        if sub_token and sub_token in s:
            if not any(f in s for f in (forbid or [])):
                allow.append(s)
    allow = [s for s in allow if not any(b in s for b in ("organizer", "organisers", "fastener", "accessor"))]
    return sorted(list(dict.fromkeys(allow)))

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
        unit = "ft"
    unit = unit.lower()
    if unit in ("ft", "feet", "foot"):
        return value * 304.8
    if unit in ("in", "inch", "inches"):
        return value * 25.4
    if unit in ("cm", "centimeter", "centimeters", "centimetre", "centimetres"):
        return value * 10.0
    if unit in ("m", "meter", "meters", "metre", "metres"):
        return value * 1000.0
    return value

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
    f = round(feet, 1); m = round(meters, 1)
    f_s = str(int(f)) if f.is_integer() else str(f)
    m_s = str(int(m)) if m.is_integer() else str(m)
    return f"{f_s} ft [{m_s} m]"

def _len_tol(mm: float) -> float:
    try:
        mm = float(mm)
    except Exception:
        return 0.0
    return max(25.0, mm * 0.02)

def _satisfies_numeric(meta: dict, flt: dict) -> bool:
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
        if isinstance(cond, dict):
            if "$eq" in cond:
                target = _to_num(cond["$eq"])
                if target is None or v_num is None or v_num != target:
                    return False
                continue
            if "$lt" in cond:
                t = _to_num(cond["$lt"])
                if t is None or v_num is None or not (v_num < t):
                    return False
            if "$lte" in cond:
                t = _to_num(cond["$lte"])
                if t is None or v_num is None or not (v_num <= t):
                    return False
            if "$gt" in cond:
                t = _to_num(cond["$gt"])
                if t is None or v_num is None or not (v_num > t):
                    return False
            if "$gte" in cond:
                t = _to_num(cond["$gte"])
                if t is None or v_num is None or not (v_num >= t):
                    return False
            if not any(k in cond for k in ("$lt", "$lte", "$gt", "$gte", "$eq")):
                return False
        else:
            if isinstance(cond, str):
                if str(v).strip().lower() != cond.strip().lower():
                    return False
                continue
            target = _to_num(cond)
            if target is None or v_num is None or v_num != target:
                return False
    return True

def parse_global_range(prompt_text: str):
    s = prompt_text.lower()
    s = s.replace("≧", "≥").replace("≦", "≤")
    m = re.search(r'(?:<=|=<|≤|less\s+than\s+or\s+equal\s+to|at\s*most|atmost|no\s+more\s+(?:than|then)|'
                  r'not\s+more\s+(?:than|then)|up\s*to|upto|no\s+greater\s+(?:than|then)|'
                  r'max(?:imum)?(?:\s+of)?|max)\s*(\d+(?:\.\d+)?)', s)
    if m: return {"$lte": float(m.group(1))}
    m = re.search(r'(?:<|less\s+(?:than|then)|under|below)\s*(\d+(?:\.\d+)?)', s)
    if m: return {"$lt": float(m.group(1))}
    m = re.search(r'(?:>=|=>|≥|greater\s+than\s+or\s+equal\s+to|at\s*least|atleast|no\s+less\s+(?:than|then)|'
                  r'not\s+less\s+(?:than|then)|min(?:imum)?(?:\s+of)?|min)\s*(\d+(?:\.\d+)?)', s)
    if m: return {"$gte": float(m.group(1))}
    m = re.search(r'(?:>|greater\s+(?:than|then)|more\s+(?:than|then)|over|above)\s*(\d+(?:\.\d+)?)', s)
    if m: return {"$gt": float(m.group(1))}
    m = re.search(r'(?:between|from)\s*(\d+(?:\.\d+)?)\s*(?:and|to|through|thru|-)\s*(\d+(?:\.\d+)?)', s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return {"$gte": lo, "$lte": hi}
    return None

# === Length filter (single definition; accumulates BOTH bounds) ===
def _parse_length_filter(prompt: str):
    s = prompt.lower()
    lo = None
    hi = None

    # 1) explicit ranges with units on either side
    m = re.search(
        rf'(?:between|from)\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})?\s*(?:and|to|through|thru|[-–—])\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})?',
        s
    )
    if m:
        a = float(m.group(1)); unit_a = m.group(2)
        b = float(m.group(3)); unit_b = m.group(4)
        unit = unit_a or unit_b
        if unit:
            lo = _to_mm(min(a, b), unit)
            hi = _to_mm(max(a, b), unit)

    # 2) "A to B" style
    m = re.search(
        rf'(\d+(?:\.\d+)?)\s*({_LEN_UNIT})?\s*(?:[-–—]|to|and)\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})?',
        s
    )
    if m and (lo is None and hi is None):
        a = float(m.group(1)); unit_a = m.group(2)
        b = float(m.group(3)); unit_b = m.group(4)
        unit = unit_a or unit_b
        if unit:
            lo = _to_mm(min(a, b), unit)
            hi = _to_mm(max(a, b), unit)

    # 3) accumulate single-sided bounds
    for pat, kind in [
        (rf'(?:<=|less than or equal to|at\s*most|no more than|up to)\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})', 'hi'),
        (rf'(?:<|less than|under|below)\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})', 'hi'),
        (rf'(?:>=|greater than or equal to|at\s*least|no less than|minimum of)\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})', 'lo'),
        (rf'(?:(?<!no\s)(?<!not\s)more than|(?<!no\s)(?<!not\s)greater than|over|above)\s*(\d+(?:\.\d+)?)\s*({_LEN_UNIT})', 'lo'),
    ]:
        for m in re.finditer(pat, s):
            v = _to_mm(float(m.group(1)), m.group(2))
            if kind == 'lo':
                lo = max(lo, v) if lo is not None else v
            else:
                hi = min(hi, v) if hi is not None else v

    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None and lo > hi:
        return None
    if lo is not None and hi is not None:
        return {"$gte": lo, "$lte": hi}
    if lo is not None:
        return {"$gte": lo}
    return {"$lte": hi}

def find_number_near_keywords(prompt_norm: str, keywords: list[str]):
    """
    Extract numeric filters near keywords.
    Uses the global range parser for consistency.
    """
    for kw in keywords:
        # Check if keyword is mentioned
        if kw not in prompt_norm:
            continue
        
        # First, try to find a range/comparison in the prompt
        global_range = parse_global_range(prompt_norm)
        if global_range:
            return global_range
        
        # If no range found, look for exact numbers near the keyword
        kw_esc = re.escape(kw)
        
        # Pattern: "3 ports" or "ports 3"
        m = re.search(rf'(\d+(?:\.\d+)?)\s+{kw_esc}\b', prompt_norm)
        if m:
            return {"$eq": float(m.group(1))}
        
        m = re.search(rf'\b{kw_esc}\s+(\d+(?:\.\d+)?)', prompt_norm)
        if m:
            return {"$eq": float(m.group(1))}
    
    return None

# --- phrases that imply pivoting away from the current SKU
_PIVOT_PHRASES = {
    "another", "a different", "different one", "something else", "instead",
    "similar to this but", "like this but", "do you have", "looking for",
    "need a", "not this", "without", "alternative", "alternatives",
    "other option", "other options", "different option", "different options",
    "another option", "another one"
}

# ====== NEW: clarification phrases to prevent false pivots ======
_CLARIFY_PHRASES = {
    "are you sure", "does it have", "do it have", "how many",
    "only one", "just one", "is it two", "is it 2", "not two", "not 2",
    "confirm", "double check", "double-check"
}
_PRONOUN_HINTS = {"it", "this", "that"}

def _looks_like_new_product_query(p_norm: str) -> bool:
    # clarification questions like "are you sure", "how many" shouldn't force a pivot
    if any(ph in p_norm for ph in _CLARIFY_PHRASES):
        return False

    # don't treat "any other details" (or similar) as a pivot
    if re.search(r'\bother\s+detail(s)?\b', p_norm):
        return False

    if any(phrase in p_norm for phrase in _PIVOT_PHRASES):
        return True

    has_domain = any(tok in p_norm for tok in _DOMAIN_SEEK_TOKENS)
    has_numbery = (
        bool(re.search(r'\d', p_norm)) or
        bool(re.search(_NUM_UNIT, p_norm)) or
        bool(re.search(_NUM_UNIT_NOB, p_norm)) or
        bool(_COMPARATOR_RE.search(p_norm))
    )
    color_hit = any(norm_text(c) in p_norm for c in (categorical_values.get("color", []) or []))
    material_hit = any(norm_text(m) in p_norm for m in (categorical_values.get("material", []) or []))
    wireless_hit = "wireless" in p_norm or "wifi" in p_norm or "wi fi" in p_norm
    return bool(has_domain and (has_numbery or color_hit or material_hit or wireless_hit))

def is_vague_follow_up(prompt):
    last_pn = st.session_state.get("last_product_number") or ""
    p_norm = norm_text(prompt)
    if last_pn:
        has_new_sku = bool(extract_product_number(prompt))
        anchor = detect_anchor_rules(p_norm)
        new_search_intent = _looks_like_new_product_query(p_norm)
        if has_new_sku or anchor or new_search_intent:
            return False
        return True
    if extract_product_number(prompt):
        return False
    p = prompt.lower()
    if (re.search(r'\d', p) or re.search(_NUM_UNIT, p) or re.search(_NUM_UNIT_NOB, p) or _COMPARATOR_RE.search(p)):
        return False
    pn = norm_text(prompt)
    if _NUM_WORDS_RE.search(pn):
        return False
    if re.search(r'\b(\d+|' + "|".join(_NUM_WORDS.keys()) + r')\s*[- ]?pack\b', pn):
        return False
    if any(tok in pn for tok in _DOMAIN_SEEK_TOKENS):
        return False
    anchor = detect_anchor_rules(pn)
    last_cat = (st.session_state.get("last_metadata") or {}).get("category")
    if anchor and last_cat and anchor.get("key") and anchor["key"] != last_cat:
        return False
    clean = pn
    return len(clean.split()) <= 6 or any(k in clean for k in fallback_keywords)

# ====== NEW: explicit clarification follow-up detector ======
def is_clarifying_followup(prompt: str) -> bool:
    """
    If we're already on a SKU and the user is questioning counts/specs
    with pronouns ("it/this/that") or clarify phrases, treat as follow-up,
    even if numbers are present.
    """
    if not st.session_state.get("last_product_number"):
        return False
    p = norm_text(prompt)
    if any(ph in p for ph in _CLARIFY_PHRASES) or any(w in p.split() for w in _PRONOUN_HINTS):
        if not extract_product_number(prompt):          # no new SKU
            if not _looks_like_new_product_query(p):    # not a true pivot
                return True
    return False

def _hydrate_product_from_prompt(prompt: str) -> str | None:
    pnums = extract_product_numbers(prompt)
    if not pnums:
        return None
    pn = pnums[0]
    docs = vector_store.similarity_search(query="product spec", k=5, filter={"product_number": pn})
    if docs:
        st.session_state.last_product_number = pn
        st.session_state.last_context = docs[0].page_content
        st.session_state.last_score = 1.0
        st.session_state.last_metadata = docs[0].metadata or {}
        return pn
    return None

# ====== NEW: explicit per-connector count extraction ======
_EXPL_PORT_PATTERNS = {
    "usb_c_ports": r"(?:usb[\s\-]?c|type[\s\-]?c)",
    "usb_a_ports": r"(?:usb[\s\-]?a|type[\s\-]?a|usb(?!\s*[\-]?c))",
    "hdmi_ports":  r"(?:hdmi)",
    "dp_ports":    r"(?:display\s*port|displayport|\bdp\b)",
    "vga_ports":   r"(?:vga)"
}

def _extract_explicit_port_counts(prompt: str) -> dict:
    """
    Returns dict like {"usb_c_ports": {"$eq": 4}, "hdmi_ports": {"$eq": 2}}
    when the prompt contains explicit counts near a connector term.
    Handles "4x usb-c", "usb-c x4", "2 hdmi", "hdmi 2", etc.
    """
    s = prompt.lower()
    out = {}

    for key, term in _EXPL_PORT_PATTERNS.items():
        # 4x usb-c  |  usb-c x4  |  2 usb-c  |  usb-c 2
        pats = [
            rf"\b(\d+)\s*[x×]\s*(?:{term})\b",
            rf"\b(?:{term})\s*[x×]\s*(\d+)\b",
            rf"\b(\d+)\s*(?:{term})\b",
            rf"\b(?:{term})\s*(\d+)\b",
        ]
        best = None
        for p in pats:
            m = re.search(p, s, flags=re.I)
            if m:
                try:
                    best = max(int(m.group(1)), best or 0)
                except:
                    pass
        if best:
            out[key] = {"$eq": float(best)}
    return out

def extract_filter_from_prompt(prompt):
    prompt_expanded = expand_synonyms(prompt)
    prompt_norm = norm_text(prompt_expanded)
    filters = {}

    # --- cable length ---
    has_len_kw = any(kw in prompt_norm for kw in metadata_field_keywords["cablelength"])
    mentions_length = has_len_kw or re.search(_NUM_UNIT, prompt.lower()) is not None or re.search(_NUM_UNIT_NOB, prompt.lower()) is not None
    len_cond = None
    if mentions_length:
        len_cond = _parse_length_filter(prompt)
        if len_cond:
            filters["cablelength"] = len_cond
        else:
            m_one = re.search(rf'(\d+(?:\.\d+)?)(?:\s*)({_LEN_UNIT_NOB})', prompt.lower())
            if m_one:
                val = float(m_one.group(1)); unit = m_one.group(2)
                mm = _to_mm(val, unit)
                tol = _len_tol(mm)
                filters["cablelength"] = {"$gte": mm - tol, "$lte": mm + tol}
            elif has_len_kw:
                m2 = re.search(r'\b(\d+(?:\.\d+)?)\b', prompt_norm)
                if m2:
                    mm = _to_mm(float(m2.group(1)), "ft")
                    tol = _len_tol(mm)
                    filters["cablelength"] = {"$gte": mm - tol, "$lte": mm + tol}

    # ========== CONNECTOR MATCHING FIRST ==========
    # ← NOTE: This is at the SAME level as the length section, not inside it!
    p_lower = prompt.lower()
    connector_matched = False

    conn_pattern = r'\b(usb[\s\-]?c|usb[\s\-]?a|usb|hdmi|displayport|display\s*port|dp|dvi|vga)\b'
    conn_matches = list(re.finditer(conn_pattern, p_lower, re.I))
    has_side_language = bool(re.search(r'\b(side|end)\b.*(other|side|end)\b', p_lower))

    # Pattern 0: Single connector cables (e.g., "HDMI cable", "USB-C cable")
    # BUT: Skip if there's an "X to Y" pattern (that should be handled by Pattern 2)
    has_to_pattern = re.search(r'\bto\b', p_lower) and len(conn_matches) >= 2
    
    if not has_to_pattern:
        single_conn_patterns = {
            'hdmi': r'\bhdmi\s+cable\b',
            'displayport': r'\b(?:displayport|display\s*port|dp)\s+cable\b',
            'vga': r'\bvga\s+cable\b',
            'dvi': r'\bdvi\s+cable\b',
            'usb-c': r'\busb[\s\-]?c\s+cable\b',
            'usb-a': r'\busb[\s\-]?a\s+cable\b',
        }
        
        for conn_type, pattern in single_conn_patterns.items():
            if re.search(pattern, p_lower):
                filters["interfacea_type"] = conn_type
                filters["interfaceb_type"] = conn_type
                connector_matched = True
                print(f"DEBUG: Single connector cable matched: {conn_type}")
                break

    # Pattern 1: "X on one side and Y on the other side/end"
    if not connector_matched and len(conn_matches) >= 2 and has_side_language:
        conn_a = conn_matches[0].group(1).lower().replace(' ', '').replace('-', '')
        conn_b = conn_matches[1].group(1).lower().replace(' ', '').replace('-', '')
            
        if conn_a != conn_b:
            connector_matched = True
            
            # Map connector A
            if 'usbc' in conn_a or 'typec' in conn_a:
                filters["interfacea_type"] = "usb-c"
            elif 'usba' in conn_a:
                filters["interfacea_type"] = "usb-a"
            elif conn_a == 'usb':
                filters["interfacea_type"] = "usb"
            elif conn_a == 'hdmi':
                filters["interfacea_type"] = "hdmi"
            elif 'displayport' in conn_a or conn_a == 'dp':
                filters["interfacea_type"] = "displayport"
            elif conn_a == 'dvi':
                filters["interfacea_type"] = "dvi"
            elif conn_a == 'vga':
                filters["interfacea_type"] = "vga"
            
            # Map connector B
            if 'usbc' in conn_b or 'typec' in conn_b:
                filters["interfaceb_type"] = "usb-c"
            elif 'usba' in conn_b:
                filters["interfaceb_type"] = "usb-a"
            elif conn_b == 'usb':
                filters["interfaceb_type"] = "usb"
            elif conn_b == 'hdmi':
                filters["interfaceb_type"] = "hdmi"
            elif 'displayport' in conn_b or conn_b == 'dp':
                filters["interfaceb_type"] = "displayport"
            elif conn_b == 'dvi':
                filters["interfaceb_type"] = "dvi"
            elif conn_b == 'vga':
                filters["interfaceb_type"] = "vga"
            
            print(f"DEBUG: Connector matched! interfacea_type={filters.get('interfacea_type')}, interfaceb_type={filters.get('interfaceb_type')}")  # DEBUG

    # Pattern 2: "X to Y" style patterns (only if Pattern 1 didn't match)
    if not connector_matched:
        # NEW: Handle USB-C to HDMI specifically (must come BEFORE generic USB to HDMI)
        if re.search(r'\busb[\s\-]?c\s+to\s+hdmi\b', p_lower):
            filters["interfacea_type"] = "usb-c"
            filters["interfaceb_type"] = "hdmi"
            connector_matched = True
            print(f"DEBUG: USB-C to HDMI matched!")
        elif re.search(r'\bhdmi\s+to\s+usb[\s\-]?c\b', p_lower):
            filters["interfacea_type"] = "hdmi"
            filters["interfaceb_type"] = "usb-c"
            connector_matched = True
            print(f"DEBUG: HDMI to USB-C matched!")
        # VGA patterns
        elif re.search(r'\bvga\s+to\s+hdmi\b', p_lower):
            filters["interfacea_type"] = "vga"
            filters["interfaceb_type"] = "hdmi"
            connector_matched = True
            print(f"DEBUG: VGA to HDMI matched!")
        elif re.search(r'\bhdmi\s+to\s+vga\b', p_lower):
            filters["interfacea_type"] = "hdmi"
            filters["interfaceb_type"] = "vga"
            connector_matched = True
            print(f"DEBUG: HDMI to VGA matched!")
        # Generic USB to HDMI (must come AFTER USB-C specific patterns)
        elif re.search(r'\busb\s+to\s+hdmi\b', p_lower):
            # This will only match if it's NOT USB-C (since USB-C was already handled)
            filters["interfacea_type"] = "usb"
            filters["interfaceb_type"] = "hdmi"
            connector_matched = True
            print(f"DEBUG: USB to HDMI matched!")
        # DisplayPort patterns
        elif re.search(r'\b(?:displayport|display\s*port|dp)\s+to\s+hdmi\b', p_lower):
            filters["interfacea_type"] = "displayport"
            filters["interfaceb_type"] = "hdmi"
            connector_matched = True
            print(f"DEBUG: DisplayPort to HDMI matched!")
        elif re.search(r'\bhdmi\s+to\s+(?:displayport|display\s*port|dp)\b', p_lower):
            filters["interfacea_type"] = "hdmi"
            filters["interfaceb_type"] = "displayport"
            connector_matched = True
            print(f"DEBUG: HDMI to DisplayPort matched!")
        # DVI patterns
        elif re.search(r'\bdvi\s+to\s+hdmi\b', p_lower):
            filters["interfacea_type"] = "dvi"
            filters["interfaceb_type"] = "hdmi"
            connector_matched = True
            print(f"DEBUG: DVI to HDMI matched!")
        elif re.search(r'\bhdmi\s+to\s+dvi\b', p_lower):
            filters["interfacea_type"] = "hdmi"
            filters["interfaceb_type"] = "dvi"
            connector_matched = True
            print(f"DEBUG: HDMI to DVI matched!")

    print(f"DEBUG: connector_matched = {connector_matched}")  # DEBUG
    print(f"DEBUG: Current filters = {filters}")  # DEBUG

    # ========== END CONNECTOR MATCHING ==========

    # NOW the conditional category logic:

    # ---- Anchor category (skip if connector matched) ----
    if not connector_matched:
        anchor = detect_anchor_rules(prompt_norm)
        if anchor:
            cat_values = _category_values_containing(anchor["key"])
            if cat_values:
                filters["category"] = {"$in": cat_values}
            
            # ========== NEW: Only add anchor subcategories if query is specific ==========
            # Only add subcategories if query mentions specific technologies
            query_lower = prompt_norm.lower()

            specific_tech_terms = {
                "usb", "usbc", "usb-c", "usb-a", "usba", "thunderbolt", "tb3", "tb4",
                "hdmi", "displayport", "dp", "vga", "dvi", "ethernet", "4k", "dual", "triple"
            }

            query_words_set = set(re.findall(r"[a-z0-9]+", query_lower))
            has_specific_tech = bool(query_words_set & specific_tech_terms)

            if has_specific_tech:
                allow_subs = build_subcategory_allowlist(anchor.get("sub_token"), anchor.get("forbid"))
                if allow_subs:
                    filters["subcategory"] = {"$in": allow_subs}
                    print(f"DEBUG: Anchor added {len(allow_subs)} subcategories for specific query")
            else:
                print(f"DEBUG: Anchor skipped subcategory filter for broad query")
            # ================================================================================

    # Replace the section in extract_filter_from_prompt after connector matching
    # Starting from "# ---- Token-based multi-category..." 

    # Track cable intent for smart filtering
    cable_intent = "cable" in prompt_norm or "cables" in prompt_norm

    # ---- Token-based multi-category (skip if connector matched) ----
    if not connector_matched:
        # Check if anchor already set subcategories
        anchor_set_subcategory = "subcategory" in filters
        
        multi_cat = pick_categories_from_prompt(prompt)
        multi_cat_keys = set(multi_cat.keys()) if multi_cat else set()

        if multi_cat:
            if "category" in multi_cat:
                filters["category"] = multi_cat["category"]
            # Only apply subcategory from token-based if anchor didn't set it
            if "subcategory" in multi_cat and not anchor_set_subcategory:
                filters["subcategory"] = multi_cat["subcategory"]
                print(f"DEBUG: Token-based added subcategories")
            elif anchor_set_subcategory:
                print(f"DEBUG: Keeping anchor subcategories, skipping token-based")
    else:
        # If connector matched, use broad category allowlist
        print(f"DEBUG: Connector matched - using broad categories")  # DEBUG
        all_cats = categorical_values.get("category", [])
        connector_cats = [
            norm_text(c) for c in all_cats 
            if any(keyword in norm_text(c) for keyword in 
                ["cable", "display", "video", "connectivity", "adapter", "converter", "av"])
        ]
        print(f"DEBUG: connector_cats = {connector_cats}")  # DEBUG
        if connector_cats:
            filters["category"] = {"$in": connector_cats}
        multi_cat_keys = set()  # Empty set to skip later categorical matching
        
        # NEW: If user said "cable", prevent subcategory filtering to allow adapters/converters
        if cable_intent:
            print(f"DEBUG: Cable intent detected - will skip subcategory filtering")  # DEBUG
            multi_cat_keys.add("subcategory")  # Mark as handled to prevent addition

    # === Explicit connector counts ===
    explicit_counts = _extract_explicit_port_counts(prompt)
    if explicit_counts:
        for k, v in explicit_counts.items():
            filters[k] = v

    # --- numeric fields (global / near-keyword) ---
    rng_any = parse_global_range(prompt)
    for field, keywords in metadata_field_keywords.items():
        if field == "cablelength":
            continue
        if field in filters:
            continue

        if any(kw in prompt_norm for kw in keywords):
            val = find_number_near_keywords(prompt_norm, keywords)
            if val is not None:
                filters[field] = val
            elif rng_any:
                filters[field] = rng_any

    # --- SPECIAL: pack quantity ---
    m_pack = re.search(r'\b(\d+|' + "|".join(_NUM_WORDS.keys()) + r')\s*[- ]?pack\b', prompt_norm)
    if m_pack:
        token = m_pack.group(1)
        qty = int(token) if token.isdigit() else word_to_int(token) or None
        if qty is not None:
            filters["packqty"] = {"$eq": float(qty)}

    # --- categorical (fallback fuzzy if not already set) ---
    for meta_key in ("category", "subcategory", "material", "fiberduplex", "fibertype", "color", "wireless"):
        if meta_key in multi_cat_keys:
            continue
        if meta_key in filters:
            continue
        
        # NEW: Skip subcategory when connector matched + cable intent
        if connector_matched and cable_intent and meta_key == "subcategory":
            print(f"DEBUG: Skipping subcategory filter due to cable intent + connector matching")  # DEBUG
            continue
        
        val = try_match_categorical(meta_key, prompt_norm)
        if val:
            filters[meta_key] = val

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

    # SAFETY NET: remap 'kvmports' -> 'ports'
    if "kvmports" in filters:
        cond = filters.pop("kvmports")
        if "ports" in filters and isinstance(filters["ports"], dict) and isinstance(cond, dict):
            filters["ports"] = {**filters["ports"], **cond}
        else:
            filters["ports"] = cond

    # Cable category bias
    if ("cablelength" in filters) and ("cable" in prompt_norm) and not connector_matched:
        cable_cats = [norm_text(v) for v in (categorical_values.get("category") or []) if v and "cable" in v.lower()]
        cable_cats = [c for c in cable_cats if not any(bad in c for bad in ("organizer","organiser","fastener","accessor"))]
        if cable_cats:
            filters["category"] = {"$in": cable_cats}
            if "subcategory" in filters and isinstance(filters["subcategory"], dict) and "$in" in filters["subcategory"]:
                filters["subcategory"]["$in"] = [
                    s for s in filters["subcategory"]["$in"]
                    if not any(b in s for b in ("organizer","organiser","fastener","accessor"))
                ]
                if not filters["subcategory"]["$in"]:
                    filters.pop("subcategory", None)

    # Store filters for logging
    st.session_state.last_filters_applied = filters
    
    return filters or None

def show_response(reply):
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append(AIMessage(reply))
    
    # Get the last user message
    last_user_msg = ""
    for msg in reversed(st.session_state.messages[:-1]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    
    if last_user_msg:
        # Check if this is a multi-product response
        multi_products = st.session_state.get('multi_products_shown')
        
        if multi_products:
            # Log with product list
            log_to_csv(last_user_msg, reply, products_shown=multi_products)
            # Clear the multi-product list
            st.session_state.multi_products_shown = None
        else:
            # Log as single product
            log_to_csv(last_user_msg, reply)

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
            "'That detail isn't in the spec I have.'"
        )]
        + [SystemMessage(
            "STYLE: Be conversational and concise like a knowledgeable product specialist. "
            "No section headings or tables. Prefer 3–6 sentences. "
            "Start with: 'For <PRODUCT NUMBER>:' once, then explain. "
            "Answer the user's question directly using the spec. "
            "If a quick list helps, you may include up to 3–5 short bullets, but only when the user asked for 'specs', 'what's included', or similar. "
            "If 'Included in Package' exists AND the user asks what's in the box, summarize it inline as 'In the box: ...'. "
            "When users ask about ports/connectors, read from **Connector Type**, **External Ports**, or **Host Connectors** verbatim from the spec. "
            "Avoid emojis. End with a short offer to check another detail if needed."
        )]
        + [SystemMessage(f"SPECIFICATION:\n{st.session_state.last_context}")]
        + [HumanMessage(prompt_text)]
    )
    return llm.invoke(messages).content

def sanitize_reply(reply:str)->str:
    return reply

def handle_greeting(prompt):
    greeting_keywords = ["hello","hi","hey","good morning","good afternoon","good evening"]
    wc = len(prompt.strip().split())
    if not st.session_state.greeted_user and wc <= 4 and any(re.search(rf"\b{g}\b", prompt.lower()) for g in greeting_keywords):
        show_response("Hi there! I'm here to help you find the right StarTech.com product. Tell me what you're trying to do or ask about a specific product number.")
        st.session_state.greeted_user = True
        return True
    return False

def handle_farewell(prompt):
    if is_farewell(prompt):
        show_response("Thanks for chatting! If you need anything else about StarTech.com products, just ask.")
        return True
    return False

def is_farewell(prompt):
    p = norm_text(prompt)
    return any(k in p for k in farewell_keywords)

def is_install_request(prompt):
    p = norm_text(prompt)
    return any(k in p for k in install_keywords)

def handle_install_block(prompt):
    if is_install_request(prompt):
        _hydrate_product_from_prompt(prompt)
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

# --------- Modal fallback helper (category reroute on 0 hits) ----------
def _modal_category_fallback(original_filters: dict, prompt: str, top_k: int = 30):
    results = vector_store.similarity_search_with_relevance_scores(prompt, k=top_k)
    if not results:
        return None
    freq = {}
    for d, s in results:
        cat = (d.metadata or {}).get("category")
        if not cat:
            continue
        freq[cat] = freq.get(cat, 0) + 1
    if not freq:
        return None
    winner = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    new_filters = dict(original_filters or {})
    new_filters["category"] = winner
    if "subcategory" in new_filters:
        new_filters.pop("subcategory", None)
    return new_filters

def _score_product_simplicity(metadata: dict) -> float:
    """
    Score products based on simplicity for cable/adapter queries.
    Higher score = simpler/more direct product.
    """
    subcat = (metadata.get("subcategory") or "").lower()
    
    # Penalize complex systems
    if any(word in subcat for word in ["table", "panel", "box", "system", "module", "rack"]):
        return 0.3
    
    # Boost direct cables/adapters
    if any(word in subcat for word in ["cable", "adapter", "converter"]):
        return 1.0
    
    return 0.7  # neutral

def _should_show_multiple_products(prompt: str, num_matches: int) -> bool:
    """
    Determine if we should show multiple products instead of just one.
    """
    if num_matches < 3:
        return False
    
    p_lower = prompt.lower()
    
    # Exploratory phrases that suggest wanting options
    exploratory_phrases = [
        "show me", "what are", "what options", "do you have", "can you show",
        "list", "compare", "what's available", "what models", "which ones",
        "give me options", "any", "some", "a few"
    ]
    
    if any(phrase in p_lower for phrase in exploratory_phrases):
        return True
    
    # If query is very short and vague (3 words or less)
    words = [w for w in prompt.split() if w.lower() not in _STOPWORDS]
    if len(words) <= 3:
        return True
    
    return False

def _build_multi_product_response(products: list, prompt: str) -> str:
    """
    Build a response showing multiple products (3-5 max).
    Returns formatted markdown with product summaries.
    Also stores product info in session state for logging.
    """
    # Limit to top 5
    products = products[:5]
    
    # NEW: Store products for logging
    products_for_logging = []
    
    response = f"I found {len(products)} products that match your search:\n\n"
    
    for i, doc in enumerate(products, 1):
        meta = doc.metadata or {}
        pn = meta.get("product_number", "Unknown")
        cat = meta.get("category", "").title()
        subcat = meta.get("subcategory", "").title()
        
        # NEW: Add to logging list
        products_for_logging.append({
            'product_number': pn,
            'category': meta.get('category', ''),
            'subcategory': meta.get('subcategory', ''),
            'score': 1.0  # Multi-product responses don't have individual scores
        })
        
        # Extract key specs from page_content
        content_lines = doc.page_content.split('\n')
        key_specs = []
        
        priority_keywords = ["Cable Length:", "Connector A:", "Connector B:", "Ports:", "Displays:"]
        for keyword in priority_keywords:
            for line in content_lines:
                if keyword in line:
                    spec = line.strip()
                    if spec and len(spec) < 100:
                        key_specs.append(spec)
                    break  # Found this keyword, move to next
            if len(key_specs) >= 3:
                break
        
        response += f"**{i}. {pn}**\n"
        if subcat:
            response += f"*{subcat}*\n"
        if key_specs:
            for spec in key_specs:
                response += f"- {spec}\n"
        response += "\n"
    
    response += "Would you like detailed specs on any of these? Just let me know the product number!"
    
    # NEW: Store in session state for logging
    st.session_state.multi_products_shown = products_for_logging
    
    return response

def handle_descriptive_query(prompt, f_override=None):
    f = f_override if f_override is not None else extract_filter_from_prompt(prompt)
    st.session_state.last_product_number = ""
    st.session_state.last_context = ""
    st.session_state.last_score = None
    st.session_state.last_metadata = {}
    try:
        if f:
            docs = vector_store.similarity_search("product spec", k=50, filter=f)
            
            # Store results count for logging
            st.session_state.results_count = len(docs) if docs else 0
            
            if docs:
                # If we have connector filters, re-rank by simplicity
                if "interfacea_type" in f or "interfaceb_type" in f:
                    scored = [(doc, _score_product_simplicity(doc.metadata)) for doc in docs]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    docs = [doc for doc, score in scored]
                
                # Check if we should show multiple products
                if _should_show_multiple_products(prompt, len(docs)):
                    multi_response = _build_multi_product_response(docs, prompt)
                    show_response(multi_response)
                    return True
                
                return use_top_result([(docs[0], 1.0)])
    except Exception as e:
        print(f"Metadata-filtered search failed: {e}")

    results = vector_store.similarity_search_with_relevance_scores(prompt, k=12)
    results = [(doc, score) for doc, score in results if score >= 0.35]
    if not results:
        return False

    if f:
        for d, s in results:
            if _satisfies_numeric(d.metadata or {}, f):
                return use_top_result([(d, s)])
        show_response("I couldn't find a product that meets that requirement. If you can adjust it, I'll try again.")
        return "no-match"

    return use_top_result(results)

def use_top_result(results):
    top_doc, score = results[0]
    st.session_state.last_product_number = top_doc.metadata.get("product_number", "")
    st.session_state.last_context = top_doc.page_content
    st.session_state.last_score = score
    st.session_state.last_metadata = top_doc.metadata or {}
    return True

# -------------------- app state --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(
        "You are a StarTech.com assistant. Always be friendly and professional. "
        "You only answer questions about StarTech.com products using the provided context. "
        "Do not mention or recommend products from any other company, supplier, or brand — only StarTech.com. "
        "Do not make up information. If you're unsure, ask for clarification. "
        "Stay helpful, polite, and focused on StarTech.com solutions."
    )]
if "last_product_number" not in st.session_state: st.session_state.last_product_number = ""
if "last_context" not in st.session_state: st.session_state.last_context = ""
if "last_score" not in st.session_state: st.session_state.last_score = None
if "greeted_user" not in st.session_state: st.session_state.greeted_user = False
if "last_metadata" not in st.session_state: st.session_state.last_metadata = {}
if "last_rerouted_filters" not in st.session_state: st.session_state.last_rerouted_filters = None

for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage): continue
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# ========== ADMIN SECTION: Password Protected Conversation Viewer ==========
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔐 Admin Access")
    admin_password = st.text_input("Enter Admin Password", type="password", key="admin_pw")
    
    # Check password (uses environment variable or default)
    correct_password = os.getenv("ADMIN_PASSWORD", "startech2025")
    
    if admin_password and admin_password == correct_password:
        st.success("✅ Admin access granted")
        
        # Show conversation logs button
        if st.button("📊 View Conversation Logs"):
            if os.path.exists('conversations.csv'):
                try:
                    df = pd.read_csv('conversations.csv')
                    
                    st.subheader("📋 Recent Conversations")
                    st.caption(f"Total conversations logged: {len(df)}")
                    
                    # Show filters
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        filter_type = st.selectbox(
                            "Filter by Response Type",
                            ["All", "single-product", "multi-product", "no-product"]
                        )
                    
                    with col2:
                        filter_status = st.selectbox(
                            "Filter by Match Status",
                            ["All", "success", "no-match", "other"]
                        )
                    
                    # Apply filters
                    filtered_df = df.copy()
                    if filter_type != "All":
                        filtered_df = filtered_df[filtered_df['Response Type'] == filter_type]
                    if filter_status != "All":
                        filtered_df = filtered_df[filtered_df['Match Status'] == filter_status]
                    
                    # Show number of rows to display
                    num_rows = st.slider("Number of rows to display", 10, 100, 50)
                    
                    # Display the data
                    st.dataframe(
                        filtered_df.tail(num_rows),
                        use_container_width=True,
                        height=400
                    )
                    
                    st.caption(f"Showing {min(num_rows, len(filtered_df))} of {len(filtered_df)} filtered rows")
                    
                    # Download button
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Full CSV",
                        data=csv_data,
                        file_name=f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download all conversation logs for Power BI analysis"
                    )
                    
                    # Quick stats
                    st.markdown("---")
                    st.markdown("### 📊 Quick Stats")
                    
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        total_sessions = df['Session ID'].nunique()
                        st.metric("Total Sessions", total_sessions)
                    
                    with stats_col2:
                        avg_queries = df.groupby('Session ID')['Query Number'].max().mean()
                        st.metric("Avg Queries/Session", f"{avg_queries:.1f}")
                    
                    with stats_col3:
                        success_rate = (df['Match Status'] == 'success').sum() / len(df) * 100
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            else:
                st.warning("⚠️ No conversation logs found yet. Start chatting to generate data!")
    
    elif admin_password:
        st.error("❌ Incorrect password")

prompt = st.chat_input("Ask about a StarTech.com product")
if prompt:
    st.session_state.messages.append(HumanMessage(prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    if handle_install_block(prompt): st.stop()
    if handle_greeting(prompt) or handle_farewell(prompt): st.stop()

    handled = handle_explicit_product(prompt)
    if not handled:
        # ====== NEW: treat clarification as follow-up on current SKU ======
        if is_clarifying_followup(prompt):
            if not st.session_state.last_context:
                if handle_vague_follow_up(prompt): st.stop()
            reply = render_conversational_answer(prompt)
            reply = sanitize_reply(reply)
            show_response(reply)
            st.stop()

        if is_vague_follow_up(prompt):
            if handle_vague_follow_up(prompt): st.stop()
        else:
            active_filters = extract_filter_from_prompt(prompt)
            status = handle_descriptive_query(prompt, f_override=active_filters)

            print("\n--- Debug Info ---")
            print(f"Active Filters: {active_filters}")
            if st.session_state.get("last_rerouted_filters"):
                print(f"Rerouted Filters (modal fallback): {st.session_state.last_rerouted_filters}")
                st.session_state.last_rerouted_filters = None
            print(f"User Prompt: {prompt}")
            print(f"Product Number: {st.session_state.get('last_product_number') or ''}")
            print(f"Similarity Score: {st.session_state.get('last_score') if st.session_state.get('last_score') is not None else 'N/A'}")

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
                "interface", "mounting_options",
                # per-connector derived fields:
                "usb_c_ports", "usb_a_ports", "hdmi_ports", "dp_ports", "vga_ports"
            )
            print("Resolved Metadata (top doc):")
            for k in interesting_keys:
                if k in md:
                    print(f"- {k}: {md[k]}")
            print("Context Retrieved:")
            print(st.session_state.get("last_context") or "(none)")
            print("-------------------\n")

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

    # Only show fallback if we haven't already given a response
    if not st.session_state.last_context and len(st.session_state.messages) > 0:
        # Check if last message was from assistant (we already responded)
        if isinstance(st.session_state.messages[-1], AIMessage):
            st.stop()  # Already responded, don't show fallback
        
        show_response(
            "I couldn't find a specific match yet. If you share a product number or more detail (e.g., interface, length, color), I'll pull the exact specs."
        )
        st.stop()

    reply = render_conversational_answer(prompt)
    reply = sanitize_reply(reply)
    show_response(reply)


    