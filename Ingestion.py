# Imports & Environment loading
import os
import time
import json
import re
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm

load_dotenv()

# ---- env helpers (consistent, safe) ----
def env_required(name: str, hint: str = "") -> str:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Missing environment variable '{name}'. {hint}")
    return v

def env_optional(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or str(v).strip() == "" else v

def env_optional_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def env_optional_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

# Required
PINECONE_API_KEY = env_required("PINECONE_API_KEY", "Set it in your .env or deployment secrets.")
OPENAI_API_KEY   = env_required("OPENAI_API_KEY", "Set it in your .env or deployment secrets.")
INDEX_NAME       = env_required("PINECONE_INDEX_NAME", "This is the Pinecone index name to use/create.")

# Optional tunables
EMBED_MODEL = env_optional("EMBED_MODEL", "text-embedding-3-large")
# Derive embedding dim from model (override via EMBED_DIM if needed)
MODEL_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}
EMBED_DIM = int(env_optional("EMBED_DIM", str(MODEL_DIMS.get(EMBED_MODEL, 3072))))

PINECONE_CLOUD  = env_optional("PINECONE_CLOUD", "aws")
PINECONE_REGION = env_optional("PINECONE_REGION", "us-east-1")
DATA_XLSX_PATH  = env_optional("DATA_XLSX_PATH", "documents/Main Data AI Bot.xlsx")

BATCH_SIZE = env_optional_int("BATCH_SIZE", 100)
SLEEP_BETWEEN_BATCHES = env_optional_float("SLEEP_BETWEEN_BATCHES", 0.0)  # seconds

# ---- Pinecone setup ----
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index '{INDEX_NAME}' (dim={EMBED_DIM}, metric=cosine, {PINECONE_CLOUD}/{PINECONE_REGION})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
else:
    desc = pc.describe_index(INDEX_NAME)
    serverless = desc.get("spec", {}).get("serverless", {})
    dim = desc.get("dimension")
    metric = desc.get("metric")
    if dim is not None and int(dim) != EMBED_DIM:
        raise RuntimeError(
            f"Pinecone index '{INDEX_NAME}' dimension={dim} != expected EMBED_DIM={EMBED_DIM}. "
            f"Either set EMBED_DIM to {dim}, switch EMBED_MODEL accordingly, or recreate the index."
        )
    if metric and metric.lower() != "cosine":
        raise RuntimeError(
            f"Pinecone index '{INDEX_NAME}' metric={metric} but this script expects 'cosine'. "
            "Recreate the index with metric='cosine' or adjust your code."
        )
    if serverless:
        print(f"Using existing Pinecone index '{INDEX_NAME}' ({serverless.get('cloud')}/{serverless.get('region')}), dim={dim}, metric={metric}")

index = pc.Index(INDEX_NAME)

embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ---- EXCEL PROCESSING ----
if not os.path.exists(DATA_XLSX_PATH):
    raise FileNotFoundError(
        f"Could not find data file at '{DATA_XLSX_PATH}'. "
        "Set DATA_XLSX_PATH env var or place the Excel file at the default path."
    )

print(f"Loading data from: {DATA_XLSX_PATH}")
df = pd.read_excel(DATA_XLSX_PATH)

# Normalize headers so trailing spaces / weird casing don't break lookups
df.rename(columns=lambda c: str(c).strip(), inplace=True)

# Ensure valid IDs (recommended)
if "Product Number" not in df.columns:
    raise KeyError("The Excel file is missing the required 'Product Number' column.")

df["Product Number"] = df["Product Number"].astype(str).str.strip()

# Drop blank IDs
df = df[df["Product Number"].ne("")].copy()

# (Optional) drop duplicate IDs, keep first
before = len(df)
df = df.drop_duplicates(subset=["Product Number"], keep="first").copy()
dupes = before - len(df)
if dupes:
    print(f"Dropped {dupes} duplicate Product Numbers")

print(f"Total rows to index: {len(df)}")

# ---- Export SKU vocabulary for chatbot matching ----
sku_vocab = sorted(df["Product Number"].astype(str).str.strip().str.upper().unique().tolist())
os.makedirs("documents", exist_ok=True)
with open("documents/sku_vocab.json", "w", encoding="utf-8") as f:
    json.dump({"skus": sku_vocab}, f, ensure_ascii=False, indent=2)
print(f"Exported SKU vocab: {len(sku_vocab)} SKUs -> documents/sku_vocab.json")

# --- Data fixes / transforms ---

def num_or_none(x):
    if pd.isna(x):
        return None
    m = re.search(r'[-+]?\d+(?:\.\d+)?', str(x))
    return float(m.group(0)) if m else None

# Ports: max of TOTALPORTS / NUMBERPORTS / KVMPORTS
totalports  = (df.get("TOTALPORTS")  if "TOTALPORTS"  in df.columns else pd.Series([None]*len(df))).apply(num_or_none)
numberports = (df.get("NUMBERPORTS") if "NUMBERPORTS" in df.columns else pd.Series([None]*len(df))).apply(num_or_none)
kvmports    = (df.get("KVMPORTS")    if "KVMPORTS"    in df.columns else pd.Series([None]*len(df))).apply(num_or_none)
df["Ports"] = pd.concat([totalports, numberports, kvmports], axis=1).max(axis=1)

# Displays: max of DOCKNUMDISPLAYS / NUMOFDISPLAY
docknumdisplays = (df.get("DOCKNUMDISPLAYS") if "DOCKNUMDISPLAYS" in df.columns else pd.Series([None]*len(df))).apply(num_or_none)
numofdisplay    = (df.get("NUMOFDISPLAY")    if "NUMOFDISPLAY"    in df.columns else pd.Series([None]*len(df))).apply(num_or_none)
df["Displays"]  = pd.concat([docknumdisplays, numofdisplay], axis=1).max(axis=1)

# Drop the originals now that we've merged
df.drop(columns=["TOTALPORTS", "NUMBERPORTS", "KVMPORTS"], errors="ignore", inplace=True)
df.drop(columns=["DOCKNUMDISPLAYS", "NUMOFDISPLAY"], errors="ignore", inplace=True)

# ---- Build Material with priority: ENCLOSURETYPE -> CONSTMATERIAL -> AMZ_MAT ----
enc = df["ENCLOSURETYPE"] if "ENCLOSURETYPE" in df.columns else pd.Series([None] * len(df))
con = df["CONSTMATERIAL"] if "CONSTMATERIAL" in df.columns else pd.Series([None] * len(df))
amz = df["AMZ_MAT"] if "AMZ_MAT" in df.columns else pd.Series([None] * len(df))

enc_clean = enc.mask(enc.astype(str).str.strip() == "")
con_clean = con.mask(con.astype(str).str.strip() == "")
amz_clean = amz.mask(amz.astype(str).str.strip() == "")

df["Material"] = enc_clean.fillna(con_clean).fillna(amz_clean)
df.drop(columns=["AMZ_MAT", "CONSTMATERIAL", "ENCLOSURETYPE"], errors="ignore", inplace=True)

# ---- Interface (IOINTERFACE -> KVMINTERFACE) ----
io_int = df["IOINTERFACE"] if "IOINTERFACE" in df.columns else pd.Series([None] * len(df))
kvm_int = df["KVMINTERFACE"] if "KVMINTERFACE" in df.columns else pd.Series([None] * len(df))

io_clean = io_int.mask(io_int.astype(str).str.strip() == "")
kvm_clean = kvm_int.mask(kvm_int.astype(str).str.strip() == "")

df["Interface"] = io_clean.fillna(kvm_clean)
df.drop(columns=["IOINTERFACE", "KVMINTERFACE"], errors="ignore", inplace=True)

# ---- Mounting Options (MOUNTOPTIONS -> KVMRACKMOUNT -> RACKSPECFEAT) ----
mo = df["MOUNTOPTIONS"] if "MOUNTOPTIONS" in df.columns else pd.Series([None] * len(df))
kvm_rm = df["KVMRACKMOUNT"] if "KVMRACKMOUNT" in df.columns else pd.Series([None] * len(df))
mhole = df["RACKSPECFEAT"] if "RACKSPECFEAT" in df.columns else pd.Series([None] * len(df))

mo_clean = mo.mask(mo.astype(str).str.strip() == "")
kvm_rm_clean = kvm_rm.mask(kvm_rm.astype(str).str.strip() == "")
mhole_clean = mhole.mask(mhole.astype(str).str.strip() == "")

df["Mounting Options"] = mo_clean.fillna(kvm_rm_clean).fillna(mhole_clean)
df.drop(columns=["MOUNTOPTIONS", "KVMRACKMOUNT", "RACKSPECFEAT"], errors="ignore", inplace=True)

# ---- Max Distance (MAXTRANLENGTH -> MAXDISTANCE) ----
tran_len = df["MAXTRANLENGTH"] if "MAXTRANLENGTH" in df.columns else pd.Series([None] * len(df))
max_dist = df["MAXDISTANCE"] if "MAXDISTANCE" in df.columns else pd.Series([None] * len(df))

tran_len_clean = tran_len.mask(tran_len.astype(str).str.strip() == "")
max_dist_clean = max_dist.mask(max_dist.astype(str).str.strip() == "")

df["Max Distance"] = tran_len_clean.fillna(max_dist_clean)
df.drop(columns=["MAXTRANLENGTH", "MAXDISTANCE"], errors="ignore", inplace=True)

# ---- Ethernet Speed (NETWORKSPEED -> DUPESPEED) ----
net_spd = df["NETWORKSPEED"] if "NETWORKSPEED" in df.columns else pd.Series([None] * len(df))
dupe_spd = df["DUPESPEED"] if "DUPESPEED" in df.columns else pd.Series([None] * len(df))

net_spd_clean = net_spd.mask(net_spd.astype(str).str.strip() == "")
dupe_spd_clean = dupe_spd.mask(dupe_spd.astype(str).str.strip() == "")

df["Ethernet Speed"] = net_spd_clean.fillna(dupe_spd_clean)
df.drop(columns=["NETWORKSPEED", "DUPESPEED"], errors="ignore", inplace=True)

# ---- Tokenize Material into tags for partial matching ----
def material_tokens(s):
    if pd.isna(s) or not str(s).strip():
        return []
    s = str(s).lower().replace("-", " ")
    parts = re.split(r'\b(?:and|or)\b|[\/,&+]', s)
    parts = [' '.join(p.strip().split()) for p in parts if p and p.strip()]
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

df["_material_tags"] = df["Material"].apply(material_tokens)

# ---- Normalization helpers ----
def norm_text(x):
    if pd.isna(x): return None
    return str(x).strip().lower()

# Map metadata keys -> source column names in your Excel
CATEGORICAL_FIELDS = {
    "category": "Category",
    "subcategory": "Sub Category",
    "material": "Material",
    "fiberduplex": "FIBERDUPLEX",
    "fibertype": "FIBERTYPE",
    "color": "COLOR",
    "wireless": "WIRELESS",
    # extras for context:
    "interface": "Interface",
    "mounting_options": "Mounting Options",
}

# Create normalized columns and collect distinct values for each
categorical_values = {}
for meta_key, col_name in CATEGORICAL_FIELDS.items():
    norm_col = f"_{meta_key}_norm"
    col = df[col_name] if col_name in df.columns else pd.Series([None] * len(df))
    df[norm_col] = col.apply(norm_text)
    categorical_values[meta_key] = sorted([v for v in df[norm_col].dropna().unique()])

# Add material tag vocabulary for the chatbot
tag_vocab = sorted({t for tags in df["_material_tags"] if isinstance(tags, list) for t in tags})
categorical_values["material_tags"] = tag_vocab

# Persist distinct categorical values for the chatbot
os.makedirs("documents", exist_ok=True)
with open("documents/categorical_values.json", "w", encoding="utf-8") as f:
    json.dump(categorical_values, f, ensure_ascii=False, indent=2)
print("Wrote documents/categorical_values.json")

# ---- Fields to render + index ----
fields = [
    "Product Number", "Material", "FIBERDUPLEX", "FIBERTYPE", "MAXDATARATE", "Max Distance", "MTBF", "Ethernet Speed",
    "PACKQTY", "POWERCONSUMPTION", "WARRANTY", "WAVELENGTH", "ZCONTENTITEM", "INTERFACEA", "INTERFACEB", "WIRELESS",
    "OUTPUTVOLTS", "COLOR", "HUMIDITY", "INPUTAMPS", "INPUTVOLTS", "OPERATINGTEMP", "OUTPUTAMP",
    "PLUGTYPE", "STANDARDS", "STORAGETEMP", "BUSTYPE", "CHIPID", "DOCK4KSUPPORT", "DOCKFASTCHARGE",
    "EXTERNALPORTS", "HOSTCONNECTOR", "Interface", "K_LOCK_SLOT", "LED", "MAXDVIRESOLUTION",
    "OSCOMPATIBILITY", "POWERADAPTER", "POWERDELIVERY", "Ports", "UASP_YN", "USBTYPE", "WAKEONLAN",
    "CABLELENGTH", "FULLDUPLEX", "AVINPUT", "AVOUTPUT", "MEMORY", "SUPRESOLUTIONS", "USBPASSTHRU",
    "WIDESCREEN", "KVMAUDIO", "CONNPLATING", "FIRERATING", "JACKETTYPE", "NWCABLETYPE", "SHIELDTYPE", "WIREGUAGE",
    "AUTOMDIX", "POWERADAPTERPOL", "CARDPROFILE", "CONNTYPE", "INTERNALPORTS", "PORTSTYLE", "ANTITHEFT",
    "CURVEDTV", "FLATPACK", "MAXDISPLAYSIZE", "MINDISPLAYSIZE", "Mounting Options",
    "VESAPATTERN", "VIDEOWALL", "MAXRESOLUTION", "ASPECTRATIO", "UHEIGHT", "WALLMOUNT_YN",
    "DRIVECONNECTOR", "MEDIATYPE", "HARDDRIVECOM", "INSERTIONRATE", "NUMHARDDRIVE", "CONNSTYLE", "NUMBERCONDUCTORS",
    "DRIVESIZE", "FRAMETYPE", "AVCABLING", "KVMCASCADABLE", "RATING", "LOCALCONNECTORS", "LADDERTYPE",
    "RACKTYPE", "CONDUCTORTYPE", "HOT_KEYS", "KVMCABLESINCLUDE", "MOUNTHOLETYPE",
    "KVMCONCONSOLE", "KVMIPCONTROL", "KVMPCVIDEO",
    "OSDSUPPORT", "WIRED", "WHQL",
    "DRIVECAPACITY", "POE_YN", "WDM_YN", "DUPEMODES", "MAXUSERS", "ERASE_MODES",
    "Package Height", "Package Length", "Package Width", "Product Height", "Product Length", "Product Width",
    "Shipping (Package) Weight", "Weight of Product", "Category", "Sub Category", "Displays"
]

# ---- Column labels for row rendering ----
column_map = {
    "Material": "Material",
    "FIBERDUPLEX": "Fiber Duplex",
    "FIBERTYPE": "Fiber Type",
    "MAXDATARATE": "Max Data Transfer Rate",
    "Max Distance": "Max Distance",
    "MTBF": "MTBF (Mean Time Between Failures)",
    "Ethernet Speed": "Ethernet Speed",
    "PACKQTY": "Package Quantity",
    "POWERCONSUMPTION": "Power Consumption (Watts)",
    "WARRANTY": "Warranty Period",
    "WAVELENGTH": "Wavelength",
    "ZCONTENTITEM": "Included in Package",
    "INTERFACEA": "Connector A",
    "INTERFACEB": "Connector B",
    "WIRELESS": "Wireless Capability",
    "OUTPUTVOLTS": "Output Voltage",
    "COLOR": "Color",
    "HUMIDITY": "Humidity",
    "INPUTAMPS": "Input Current",
    "INPUTVOLTS": "Input Voltage",
    "OPERATINGTEMP": "Operating Temperature",
    "OUTPUTAMP": "Output Current",
    "PLUGTYPE": "Plug Type",
    "STANDARDS": "Industry Standards",
    "STORAGETEMP": "Storage Temperature",
    "BUSTYPE": "Bus Type",
    "CHIPID": "Chipset ID",
    "DOCK4KSUPPORT": "4K Display Support",
    "DOCKFASTCHARGE": "Fast Charge Ports",
    "Displays": "Number of Displays",
    "EXTERNALPORTS": "External Ports",
    "HOSTCONNECTOR": "Host Connectors",
    "Interface": "Interface",
    "K_LOCK_SLOT": "Compatible Lock Slot",
    "LED": "LED Indicators",
    "MAXDVIRESOLUTION": "Maximum Digital Resolution",
    "OSCOMPATIBILITY": "OS Compatibility",
    "POWERADAPTER": "Power Source",
    "POWERDELIVERY": "Power Delivery",
    "Ports": "Ports",
    "UASP_YN": "UASP Support",
    "USBTYPE": "Type and Rate",
    "WAKEONLAN": "Wake On Lan",
    "CABLELENGTH": "Cable Length",
    "FULLDUPLEX": "Full Duplex",
    "AVINPUT": "AV Input",
    "AVOUTPUT": "AV Output",
    "MEMORY": "Memory",
    "SUPRESOLUTIONS": "Supported Resolutions",
    "USBPASSTHRU": "USB Passthrough",
    "WIDESCREEN": "Wide Screen Supported",
    "KVMAUDIO": "Audio",
    "CONNPLATING": "Connector Plating",
    "FIRERATING": "Fire Rating",
    "JACKETTYPE": "Cable Jacket Material",
    "NWCABLETYPE": "Cable Type",
    "SHIELDTYPE": "Cable Shield Material",
    "WIREGUAGE": "Wire Gauge",
    "AUTOMDIX": "Auto MDIX",
    "POWERADAPTERPOL": "Center Tip Polarity",
    "CARDPROFILE": "Card Type",
    "CONNTYPE": "Connector Type",
    "INTERNALPORTS": "Internal Ports",
    "PORTSTYLE": "Port Style",
    "ANTITHEFT": "Security Slot Support",
    "CURVEDTV": "Curved TV Compatible",
    "FLATPACK": "Flat Pack (Assembly Required)",
    "MAXDISPLAYSIZE": "Maximum Display Size",
    "MINDISPLAYSIZE": "Minimum Display Size",
    "Mounting Options": "Mounting Options",
    "MOUNTHOLETYPE": "Mounting Hole Type",
    "VESAPATTERN": "VESA Hole Patterns",
    "VIDEOWALL": "Video Wall",
    "MAXRESOLUTION": "Maximum Analog Resolution",
    "ASPECTRATIO": "Aspect Ratio",
    "UHEIGHT": "U Height",
    "WALLMOUNT_YN": "Wall Mountable",
    "DRIVECONNECTOR": "Drive Connectors",
    "MEDIATYPE": "Memory Media Type",
    "HARDDRIVECOM": "Compatible Drive Types",
    "INSERTIONRATE": "Insertion Rating",
    "NUMHARDDRIVE": "Number of Hard Drives",
    "CONNSTYLE": "Connector Style",
    "NUMBERCONDUCTORS": "Number of Conductors",
    "DRIVESIZE": "Drive Size",
    "FRAMETYPE": "Frame Type",
    "AVCABLING": "Cabling",
    "KVMCASCADABLE": "Daisy-Chain",
    "RATING": "Cable Rating",
    "LOCALCONNECTORS": "Local Unit Connectors",
    "LADDERTYPE": "Mounting Rail Profile",
    "RACKTYPE": "Rack Type",
    "CONDUCTORTYPE": "Conductor Type",
    "HOT_KEYS": "Hot-Key Selection",
    "KVMCABLESINCLUDE": "KVM Cables Included",
    "KVMCONCONSOLE": "Console Interface",
    "KVMIPCONTROL": "IP Control",
    "KVMPCVIDEO": "PC Video Type",
    "OSDSUPPORT": "On-Screen Display",
    "WIRED": "Wiring Standard",
    "WHQL": "Microsoft WHQL Certified",
    "DRIVECAPACITY": "Max Drive Capacity",
    "POE_YN": "PoE",
    "WDM_YN": "WDM",
    "DUPEMODES": "Duplication Modes",
    "MAXUSERS": "Max Users",
    "ERASE_MODES": "Erase Modes",
    "Package Height": "Package Height",
    "Package Length": "Package Length",
    "Package Width": "Package Width",
    "Product Height": "Product Height",
    "Product Length": "Product Length",
    "Product Width": "Product Width",
    "Shipping (Package) Weight": "Shipping Weight",
    "Weight of Product": "Product Weight",
    "Category": "Product Category",
    "Sub Category": "Product Subcategory"
}

# === Keep these fields as raw text (don’t numeric-parse) ===
TEXT_ONLY_FIELDS = {
    "CONNTYPE", "EXTERNALPORTS", "HOSTCONNECTOR",
    "INTERFACEA", "INTERFACEB", "ZCONTENTITEM"
}

def clean_value(val, field=None):
    if pd.isnull(val):
        return None

    # Keep rich text for connector/package fields
    if field and field.upper() in TEXT_ONLY_FIELDS:
        return str(val).strip()

    if field == "PACKQTY" and isinstance(val, str) and val.strip() == "1, 1":
        return 1
    try:
        f = float(val)
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        pass
    s = str(val).strip()
    m = re.search(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?', s)
    if m:
        try:
            num = float(m.group(0).replace(',', ''))
            if field == "CABLELENGTH":
                if re.search(r'\b(m|meter|meters|metre|metres)\b', s, flags=re.I):
                    num = num * 1000.0
                elif re.search(r'\b(in|inch|inches)\b', s, flags=re.I):
                    num = num * 25.4
                elif re.search(r'\b(ft|foot|feet)\b', s, flags=re.I):
                    num = num * 304.8
                elif re.search(r'\b(cm|centimeter|centimetre|centimeters|centimetres)\b', s, flags=re.I):
                    num = num * 10.0
            return int(num) if float(num).is_integer() else float(num)
        except Exception:
            pass
    return s

def normalize_product_number(pn):
    return pn.strip().upper() if isinstance(pn, str) else pn

def _fmt_num(n, decimals=1):
    s = f"{n:.{decimals}f}"
    return s[:-2] if s.endswith(".0") else s

def _format_cable_length(mm_val):
    try:
        mm = float(mm_val)
    except Exception:
        return str(mm_val)
    if mm <= 300:
        inches = mm / 25.4
        cm = mm / 10.0
        return f"{_fmt_num(inches, 1)}in [{int(round(cm))}cm]"
    else:
        feet = mm / 304.8
        meters = mm / 1000.0
        feet_str = _fmt_num(round(feet, 1), 1)
        return f"{feet_str}ft [{_fmt_num(round(meters, 1), 1)}m]"

def _format_weight_grams(g_val):
    try:
        g = float(g_val)
    except Exception:
        return str(g_val)
    if g <= 454:
        oz = g / 28.349523125
        return f"{int(round(oz))} oz [{int(round(g))} g]"
    else:
        lbs = g / 453.59237
        kg = g / 1000.0
        return f"{_fmt_num(lbs, 1)} lbs [{_fmt_num(kg, 1)} kg]"

_PRETTY_MM_FIELDS = {
    "CABLELENGTH",
    "Package Height", "Package Length", "Package Width",
    "Product Height", "Product Length", "Product Width",
}

_PRETTY_WEIGHT_FIELDS = {
    "Shipping (Package) Weight", "Weight of Product"
}

def row_to_text(row):
    lines = [f"Product Number: {row.get('Product Number')}"]
    for field in fields:
        if field == "Product Number": continue
        value = clean_value(row.get(field), field)
        if value is not None:
            label = column_map.get(field, field)
            if field in _PRETTY_MM_FIELDS and isinstance(value, (int, float)):
                lines.append(f"{label}: {_format_cable_length(value)}")
            elif field in _PRETTY_WEIGHT_FIELDS and isinstance(value, (int, float)):
                lines.append(f"{label}: {_format_weight_grams(value)}")
            else:
                lines.append(f"{label}: {value}")
    return "\n".join(lines).strip()

# === Generalized port counting (USB-C, USB-A, HDMI, DP, VGA) ===
PORT_PATTERNS = {
    "usb_c_ports": re.compile(r'\b(usb[\s\-]?c|type[\s\-]?c)\b', re.I),
    "usb_a_ports": re.compile(r'\b(usb[\s\-]?a|type[\s\-]?a)\b', re.I),
    "hdmi_ports":  re.compile(r'\bhdmi\b', re.I),
    "dp_ports":    re.compile(r'\b(display\s*port|displayport|\bdp\b)\b', re.I),
    "vga_ports":   re.compile(r'\bvga\b', re.I),
}

MULT_PATTERNS = [
    re.compile(r'\(\s*x\s*(\d+)\s*\)', re.I),   # (x2)
    re.compile(r'×\s*(\d+)', re.I),             # ×2 (unicode)
    re.compile(r'\b(\d+)\s*x\b', re.I),         # 2x
]

def count_ports(text: str, port_regex: re.Pattern) -> int | None:
    if not isinstance(text, str) or not text.strip():
        return None
    total = 0
    for seg in re.split(r'[;,\n]|\/', text):
        if not port_regex.search(seg):
            continue
        # explicit multiplier
        mult = 0
        for pat in MULT_PATTERNS:
            m = pat.search(seg)
            if m:
                mult = max(mult, int(m.group(1)))
        if mult:
            total += mult
            continue
        # fallback: "2 HDMI" or "HDMI 2" or "USB-C ports 3"
        m = re.search(r'(\d+)\s*(?:ports?|x)?\b', seg, re.I)
        total += int(m.group(1)) if m else 1
    return total or None

def build_metadata(row):
    meta = {"product_number": normalize_product_number(row["Product Number"])}
    # numeric fields
    for field in fields:
        if field == "Product Number": continue
        value = clean_value(row.get(field), field)
        if isinstance(value, (int, float)):
            meta[field.lower()] = value
    # categorical fields
    for meta_key in CATEGORICAL_FIELDS.keys():
        norm_col = f"_{meta_key}_norm"
        val = row.get(norm_col)
        if pd.notna(val):
            meta[meta_key] = val
    # material tags
    tags = row.get("_material_tags")
    if isinstance(tags, list) and tags:
        meta["material_tags"] = tags
        for t in tags:
            meta[f"mtag_{t}"] = True
    # derived: per-connector port counts from multiple text columns
    text_sources = [row.get("CONNTYPE"), row.get("EXTERNALPORTS"), row.get("HOSTCONNECTOR")]
    for key, rx in PORT_PATTERNS.items():
        best = 0
        for src in text_sources:
            c = count_ports(src, rx) if isinstance(src, str) else None
            if c:
                best = max(best, c)
        if best:
            meta[key] = float(best)
    return meta

documents = [Document(page_content=row_to_text(row), metadata=build_metadata(row)) for _, row in df.iterrows()]
uuids = [normalize_product_number(p) for p in df["Product Number"].tolist()]

print(f"Beginning upload: {len(documents)} docs, batch size {BATCH_SIZE}")
for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Uploading to Pinecone"):
    batch_docs = documents[i:i + BATCH_SIZE]
    batch_ids  = uuids[i:i + BATCH_SIZE]
    vector_store.add_documents(documents=batch_docs, ids=batch_ids)
    if SLEEP_BETWEEN_BATCHES > 0:
        time.sleep(SLEEP_BETWEEN_BATCHES)

print("Upload complete.")
# The vector size from OpenAI’s text-embedding-3-large is 3072.
