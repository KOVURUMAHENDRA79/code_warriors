# project/src/preprocess.py
import os
import re
import json
import pickle
import logging
from typing import Dict, List, Optional
import pandas as pd

# Try to import unidecode for safe unicode -> ascii normalization; it's optional
try:
    from unidecode import unidecode
except Exception:
    unidecode = lambda x: x

# Import helper to resolve dataset split path (keeps consistency with Step A)
from src.utils import resolve_train_dir, find_dataset_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Cache path (project-level)
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, "preprocess_cache.pkl")

# Small curated abbreviation map (extend as needed)
ABBREV_MAP = {
    "pvt": "private",
    "pvt.": "private",
    "ltd": "limited",
    "ltd.": "limited",
    "co": "company",
    "co.": "company",
    "corp": "corporation",
    "corp.": "corporation",
    "&": "and",
    "rd": "road",
    "rd.": "road",
    "st": "street",
    "st.": "street",
    "blk": "block",
    "bldg": "building",
    "atm": "atm",
    "fl": "floor",
    "ph": "phone",
    "blr": "bangalore",
    "mfg": "manufacturing",
    "pvtltd": "private limited",
}

# Noise words to drop from names (common)
NOISE_WORDS = {"the", "company", "services", "service", "and", "of", "india", "inc", "llp"}


# ---------- helper utilities ----------
def load_cache(path: str = CACHE_PATH) -> Dict:
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            logging.warning("Failed to load cache, continuing with empty cache.")
    return {"name_norm": {}, "addr_norm": {}}


def save_cache(cache: Dict, path: str = CACHE_PATH):
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def lower_strip(s: Optional[str]) -> str:
    if pd.isna(s) or s is None:
        return ""
    return str(s).strip().lower()


def replace_multiple_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ---------- normalization functions ----------
def expand_abbreviations(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        t_clean = t.strip().strip(".").lower()
        if t_clean in ABBREV_MAP:
            mapped = ABBREV_MAP[t_clean]
            out += mapped.split()
        else:
            out.append(t)
    return out


def remove_punctuation_keep_alnum_space(s: str) -> str:
    # Keep hyphen and slash only if needed; otherwise remove punctuation
    return re.sub(r"[^0-9a-zA-Z\s]", " ", s)


def tokenize_and_clean(s: str) -> List[str]:
    s = lower_strip(s)
    s = unidecode(s)
    s = remove_punctuation_keep_alnum_space(s)
    s = replace_multiple_spaces(s)
    tokens = [t for t in s.split(" ") if t and t not in NOISE_WORDS]
    tokens = expand_abbreviations(tokens)
    tokens = [t for t in tokens if t and t not in NOISE_WORDS]
    return tokens


PIN_REGEX = re.compile(r"\b\d{5,6}\b")  # handles 5 or 6 digit PINs common in India/US zip variants


def extract_pin(s: str) -> Optional[str]:
    if not s:
        return None
    s = str(s)
    m = PIN_REGEX.search(s)
    if m:
        return m.group(0)
    return None


def normalize_name(s: str, cache: Dict) -> str:
    BUSINESS_SUFFIXES = {
    "private", "limited", "llc", "llp", "inc", "corporation",
    "corp", "company", "co", "pvt"
}
BUSINESS_SUFFIXES = {
    "private", "limited", "llc", "llp", "inc", "corporation",
    "corp", "company", "co", "pvt"
}

def normalize_name(s: str, cache: Dict) -> str:
    key = "" if s is None else str(s)

    if key in cache["name_norm"]:
        return cache["name_norm"][key]

    tokens = tokenize_and_clean(key)

    # ❌ remove useless business suffixes
    tokens = [t for t in tokens if t not in BUSINESS_SUFFIXES]

    # ❗ VERY IMPORTANT: sort tokens (handles order mismatch)
    tokens = sorted(tokens)

    # ❗ remove duplicates
    tokens = list(dict.fromkeys(tokens))

    signature = " ".join(tokens)

    cache["name_norm"][key] = signature
    return signature


def normalize_address(s: str, cache: Dict) -> str:
    key = "" if s is None else str(s)

    if key in cache["addr_norm"]:
        return cache["addr_norm"][key]

    tokens = tokenize_and_clean(key)

    # ❗ remove weak tokens
    tokens = [t for t in tokens if len(t) > 2]

    # ❗ remove duplicates
    tokens = list(dict.fromkeys(tokens))

    # ❗ sort for consistency
    tokens = sorted(tokens)

    signature = " ".join(tokens)

    cache["addr_norm"][key] = signature
    return signature


# ---------- top-level preprocessing on DataFrame ----------
def preprocess_df(
    df: pd.DataFrame,
    id_col: str = "entity_id",
    name_col: str = "business_name",
    addr_col: str = "business_address",
    cache: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Adds the following columns in-place:
      - name_norm
      - address_norm
      - name_tokens (list)
      - address_tokens (list)
      - pin (extracted if any)
      - name_signature (sorted space-joined tokens)
    Returns the processed DataFrame (same object).
    """
    if cache is None:
        cache = load_cache()

    # ensure columns exist
    for c in (id_col, name_col, addr_col):
        if c not in df.columns:
            raise KeyError(f"Missing expected column {c} in dataframe")

    # operate on unique values for speed & cache reuse
    # Precompute unique names & addresses
    unique_names = pd.Series(df[name_col].fillna("").astype(str).unique())
    unique_addrs = pd.Series(df[addr_col].fillna("").astype(str).unique())

    logging.info("Unique names: %d, unique addresses: %d", len(unique_names), len(unique_addrs))

    # Normalize names (vectorized via python map)
    name_norm_map = {}
    for name in unique_names:
        name_norm_map[name] = normalize_name(name, cache)

    addr_norm_map = {}
    for addr in unique_addrs:
        addr_norm_map[addr] = normalize_address(addr, cache)

    # map back
    df["name_norm"] = df[name_col].fillna("").astype(str).map(name_norm_map)
    df["address_norm"] = df[addr_col].fillna("").astype(str).map(addr_norm_map)

    # tokens lists (use simple split on normalized signature)
    df["name_tokens"] = df["name_norm"].apply(lambda x: x.split() if x else [])
    df["address_tokens"] = df["address_norm"].apply(lambda x: x.split() if x else [])

    # pin extraction (prefer explicit numeric tokens in original address)
    df["pin"] = df[addr_col].astype(str).apply(extract_pin)

    # create small signatures for blocking: sorted first-3 tokens
    def signature_from_tokens(tokens: List[str], k: int = 3):
        if not tokens:
            return ""
        return " ".join(sorted(tokens)[:k])

    df["name_sig"] = df["name_tokens"].apply(lambda t: signature_from_tokens(t, k=3))
    df["addr_sig"] = df["address_tokens"].apply(lambda t: signature_from_tokens(t, k=3))

    # save cache
    save_cache(cache)

    return df


# ---------- CLI convenience: preprocess all train files for a split ----------
def preprocess_split(dataset_root: str = "dataset", split: str = "full", out_dir: Optional[str] = None):
    """
    Preprocess train_source1/2/3 for the given split and write processed TSVs to the same folder
    as <train_dir>/preprocessed_<originalname>.tsv (so original files are preserved).
    """
    train_dir = resolve_train_dir(dataset_root, split)
    logging.info("Preprocessing split=%s at %s", split, train_dir)

    # find files (use existing helper)
    files = find_dataset_files(dataset_root, data_split=split)
    train_source_files = [files.get("train_source1"), files.get("train_source2"), files.get("train_source3")]

    for p in train_source_files:
        if p is None:
            logging.warning("Missing file, skipping: %s", p)
            continue
        logging.info("Loading %s", p)
        df = pd.read_csv(p, sep="\t", dtype=str)
        # detect entity column name (common is "entity_id")
        id_col = "entity_id" if "entity_id" in df.columns else df.columns[0]
        # detect name/address columns heuristically
        name_col = "business_name" if "business_name" in df.columns else next((c for c in df.columns if "name" in c.lower()), df.columns[1])
        addr_col = "business_address" if "business_address" in df.columns else next((c for c in df.columns if "address" in c.lower()), df.columns[2])

        logging.info("Using columns: id=%s name=%s addr=%s", id_col, name_col, addr_col)
        df_proc = preprocess_df(df, id_col=id_col, name_col=name_col, addr_col=addr_col)

        # write out
        out_path = os.path.join(os.path.dirname(p), f"preprocessed_{os.path.basename(p)}")
        df_proc.to_csv(out_path, sep="\t", index=False)
        logging.info("Wrote preprocessed file: %s", out_path)

    logging.info("Preprocessing complete for split=%s", split)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess train files (rule-based, cached)")
    parser.add_argument("--dataset", type=str, default="dataset", help="dataset root")
    parser.add_argument("--split", type=str, default="small", choices=["full", "small", "medium"], help="which split to preprocess")
    args = parser.parse_args()
    preprocess_split(args.dataset, args.split)