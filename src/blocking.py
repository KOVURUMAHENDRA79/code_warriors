# project/src/blocking.py
import os
import argparse
import logging
import pickle
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.utils import resolve_train_dir, find_dataset_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------
# small utilities
# ---------------------
def soundex(s: str) -> str:
    """
    Simple Soundex implementation to get a cheap phonetic key.
    Not perfect but useful for blocking transliteration-ish cases.
    """
    if not s:
        return ""
    s = s.upper()
    # keep first letter
    first = s[0]
    # map letters to digits
    mapping = {
        **dict.fromkeys(list("BFPV"), "1"),
        **dict.fromkeys(list("CGJKQSXZ"), "2"),
        **dict.fromkeys(list("DT"), "3"),
        **dict.fromkeys(list("L"), "4"),
        **dict.fromkeys(list("MN"), "5"),
        **dict.fromkeys(list("R"), "6"),
    }
    digits = []
    for ch in s[1:]:
        d = mapping.get(ch, "0")
        if d != "0":
            digits.append(d)
    # remove consecutive duplicates
    out = []
    prev = None
    for ch in digits:
        if ch != prev:
            out.append(ch)
        prev = ch
    code = first + "".join(out)
    code = (code + "000")[:4]
    return code


def topk_indices_to_pairs(s1_ids: List[str], cand_idx_matrix: np.ndarray, s2_ids: List[str]) -> Dict[str, Set[str]]:
    """
    Convert kneighbors indices (per row) to mapping s1_id -> set(s2_id)
    cand_idx_matrix: shape (n_s1, k) matrix with indices into s2_ids
    """
    mapping = defaultdict(set)
    for i, s1 in enumerate(s1_ids):
        idxs = cand_idx_matrix[i]
        for idx in idxs:
            if idx < 0:
                continue
            mapping[s1].add(s2_ids[int(idx)])
    return mapping


# ---------------------
# blocking passes
# ---------------------
def pin_block(df_s1: pd.DataFrame, df_s2: pd.DataFrame, id_col: str, pin_col: str = "pin") -> Dict[str, Set[str]]:
    """
    Exact PIN block: if a record has pin X, match all records with same pin.
    Returns s1_id -> set(s2_id)
    """
    out = defaultdict(set)
    # build map pin -> list of ids for s2
    s2_map = defaultdict(list)
    for _, r in df_s2[[id_col, pin_col]].iterrows():
        pin = r.get(pin_col)
        if pd.isna(pin) or not pin:
            continue
        s2_map[str(pin)].append(r[id_col])
    # for each s1 with pin, return all s2 ids
    for _, r in df_s1[[id_col, pin_col]].iterrows():
        pin = r.get(pin_col)
        if pd.isna(pin) or not pin:
            continue
        for sid in s2_map.get(str(pin), []):
            out[r[id_col]].add(sid)
    logging.info("PIN block produced candidates for %d source1 rows", len(out))
    return out


def prefix_sig_block(df_s1: pd.DataFrame, df_s2: pd.DataFrame, id_col: str,
                     sig_col: str = "name_sig") -> Dict[str, Set[str]]:
    """
    Use precomputed name_sig (sorted first tokens) and name prefix (first token) to block.
    """
    out = defaultdict(set)
    s2_by_sig = defaultdict(list)
    s2_by_prefix = defaultdict(list)
    for _, r in df_s2[[id_col, sig_col]].iterrows():
        sig = r.get(sig_col, "")
        if sig:
            s2_by_sig[sig].append(r[id_col])
        prefix = str(sig).split()[0] if sig else ""
        if prefix:
            s2_by_prefix[prefix].append(r[id_col])

    for _, r in df_s1[[id_col, sig_col]].iterrows():
        sid = r[id_col]
        sig = r.get(sig_col, "")
        if sig and sig in s2_by_sig:
            for c in s2_by_sig[sig]:
                out[sid].add(c)
        prefix = str(sig).split()[0] if sig else ""
        if prefix and prefix in s2_by_prefix:
            for c in s2_by_prefix[prefix]:
                out[sid].add(c)
    logging.info("Prefix/sig block produced candidates for %d source1 rows", len(out))
    return out


def phonetic_block(df_s1: pd.DataFrame, df_s2: pd.DataFrame, id_col: str, name_col: str = "name_norm") -> Dict[str, Set[str]]:
    """
    Block on simple soundex of normalized name.
    """
    out = defaultdict(set)
    s2_map = defaultdict(list)
    for _, r in df_s2[[id_col, name_col]].iterrows():
        k = soundex(str(r.get(name_col, "")).replace(" ", ""))
        if k:
            s2_map[k].append(r[id_col])
    for _, r in df_s1[[id_col, name_col]].iterrows():
        k = soundex(str(r.get(name_col, "")).replace(" ", ""))
        if k and k in s2_map:
            for c in s2_map[k]:
                out[r[id_col]].add(c)
    logging.info("Phonetic block produced candidates for %d source1 rows", len(out))
    return out


def tfidf_ann_block(df_s1: pd.DataFrame, df_s2: pd.DataFrame, id_col: str,
                    name_col: str = "name_norm", k: int = 100, cache_key: str = "tfidf_cache") -> Dict[str, Set[str]]:
    """
    TF-IDF on char n-grams (3-5) over name_norm, then NearestNeighbors with cosine to get top-k candidates.
    Caches vectorizer + nn model to speed repeated runs.
    """
    s1_texts = df_s1[name_col].fillna("").astype(str).tolist()
    s2_texts = df_s2[name_col].fillna("").astype(str).tolist()
    s1_ids = df_s1[id_col].astype(str).tolist()
    s2_ids = df_s2[id_col].astype(str).tolist()

    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    vectorizer = None
    nn = None
    X2 = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                saved = pickle.load(f)
                vectorizer = saved["vectorizer"]
                X2 = saved["X2"]
                logging.info("Loaded TF-IDF cache from %s", cache_file)
        except Exception:
            logging.warning("Failed to load TF-IDF cache; rebuilding.")
            vectorizer = None

    if vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode")
        X2 = vectorizer.fit_transform(s2_texts)
        # cache vectorizer + X2
        with open(cache_file, "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "X2": X2}, f)
        logging.info("Built and cached TF-IDF for %d s2 strings", X2.shape[0])

    # transform s1
    X1 = vectorizer.transform(s1_texts)

    # build NN on X2
    nn = NearestNeighbors(n_neighbors=min(k, X2.shape[0]), metric="cosine", n_jobs=-1)
    nn.fit(X2)

    # kneighbors (returns distances, indices)
    dists, idxs = nn.kneighbors(X1, return_distance=True)
    # convert indices to id mapping
    mapping = topk_indices_to_pairs(s1_ids, idxs, s2_ids)
    logging.info("TF-IDF ANN block produced candidates for %d source1 rows", len(mapping))
    return mapping

def address_sig_block(df_s1, df_s2, id_col, addr_col="address_norm"):
    from collections import defaultdict

    def get_sig(addr):
        if not addr:
            return ""
        tokens = addr.split()
        return " ".join(tokens[:3])  # first 3 tokens

    out = defaultdict(set)

    s2_map = defaultdict(list)
    for _, r in df_s2[[id_col, addr_col]].iterrows():
        sig = get_sig(str(r[addr_col]))
        if sig:
            s2_map[sig].append(r[id_col])

    for _, r in df_s1[[id_col, addr_col]].iterrows():
        sig = get_sig(str(r[addr_col]))
        if sig in s2_map:
            for x in s2_map[sig]:
                out[r[id_col]].add(x)

    return out

def geo_block(df_s1, df_s2, id_col, max_per_key=200):
    from collections import defaultdict
    import random

    def get_key(row):
        pin = str(row.get("pin", "")).strip()
        city = str(row.get("city", "")).strip()
        if not pin and not city:
            return None
        return f"{pin}_{city}"

    out = defaultdict(set)

    # Build limited map
    s2_map = defaultdict(list)
    for _, r in df_s2.iterrows():
        key = get_key(r)
        if key:
            s2_map[key].append(r[id_col])

    # 🔥 LIMIT candidates per key (CRITICAL)
    for key in s2_map:
        if len(s2_map[key]) > max_per_key:
            s2_map[key] = random.sample(s2_map[key], max_per_key)

    # Match
    for _, r in df_s1.iterrows():
        key = get_key(r)
        if key and key in s2_map:
            out[r[id_col]].update(s2_map[key])

    return out


def ngram_overlap_block(df_s1: pd.DataFrame, df_s2: pd.DataFrame, id_col: str,
                        name_col: str = "name_norm", min_shared_ngrams: int = 2) -> Dict[str, Set[str]]:
    """
    Token 3-gram overlap on name_norm (tokenized into words then trigrams).
    Keep candidate if shared trigrams >= min_shared_ngrams.
    Implemented in naive but vectorized manner for speed on moderate sizes.
    """
    def trigrams_of_tokens(tokens: List[str]) -> Set[str]:
        s = " ".join(tokens)
        toks = s.split()
        grams = set()
        for t in toks:
            if len(t) < 3:
                continue
            for i in range(len(t) - 2):
                grams.add(t[i:i+3])
        return grams

    s1_ids = df_s1[id_col].astype(str).tolist()
    s2_ids = df_s2[id_col].astype(str).tolist()

    s2_grams = []
    for txt in df_s2[name_col].fillna("").astype(str).tolist():
        s2_grams.append(trigrams_of_tokens(txt.split()))
    s1_grams = []
    for txt in df_s1[name_col].fillna("").astype(str).tolist():
        s1_grams.append(trigrams_of_tokens(txt.split()))

    mapping = defaultdict(set)
    # naive nested loop but limited by earlier blocking candidates in practice; keep safe guard
    for i, g1 in enumerate(s1_grams):
        if not g1:
            continue
        for j, g2 in enumerate(s2_grams):
            if not g2:
                continue
            if len(g1.intersection(g2)) >= min_shared_ngrams:
                mapping[s1_ids[i]].add(s2_ids[j])
    logging.info("N-gram overlap block produced candidates for %d source1 rows", len(mapping))
    return mapping


# ---------------------
# helpers to union & cap
# ---------------------
def union_candidate_maps(maps: List[Dict[str, Set[str]]]) -> Dict[str, Set[str]]:
    union_map = defaultdict(set)
    for m in maps:
        for k, v in m.items():
            union_map[k].update(v)
    return union_map


def cap_candidates(union_map: Dict[str, Set[str]], cap: int = 500) -> Dict[str, List[str]]:
    out = {}
    for k, v in union_map.items():
        if len(v) <= cap:
            out[k] = list(v)
        else:
            # simple heuristic: keep deterministic sorted top cap
            out[k] = sorted(list(v))[:cap]
    return out


# ---------------------
# validation: blocking recall
# ---------------------
def compute_blocking_recall(candidate_map: Dict[str, List[str]], gt_df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    candidate_map: s1_id -> list(s2_id)
    gt_df: ground truth DataFrame (cols: first col source1 id, second col matched ids comma-separated)
    returns recall fraction and a small diagnostics dict
    """
    gt_s1_col = gt_df.columns[0]
    gt_match_col = gt_df.columns[1]
    total = 0
    found = 0
    missing_examples = {}
    for _, r in gt_df.iterrows():
        s1 = str(r[gt_s1_col])
        matches_raw = r[gt_match_col]
        if pd.isna(matches_raw) or str(matches_raw).strip() == "":
            # singleton: no matches expected -> counts as found (not relevant to blocking recall)
            continue
        total += 1
        true_ids = [m.strip() for m in str(matches_raw).split(",") if m.strip()]
        cand_set = set(candidate_map.get(s1, []))
        if any(t in cand_set for t in true_ids):
            found += 1
        else:
            missing_examples[s1] = true_ids[:3]
    recall = found / total if total > 0 else 1.0
    diag = {"total_pairs": total, "found_pairs": found, "missing_examples_sample": dict(list(missing_examples.items())[:10])}
    return recall, diag


# ---------------------
# main entry
# ---------------------
def run_blocking(dataset_root: str = "dataset", split: str = "small", k: int = 100, cap: int = 500, validate: bool = True,
                 out_path: str = None):
    train_dir = resolve_train_dir(dataset_root, split)
    logging.info("Blocking split=%s at %s", split, train_dir)

    files = find_dataset_files(dataset_root, data_split=split)
    p1 = files.get("train_source1")
    p2 = files.get("train_source2")
    p3 = files.get("train_source3")
    p_gt = files.get("train_ground_truth")

    if not p1 or not p2 or not p3:
        raise FileNotFoundError("Expected train_source1/2/3 present in split train dir")

    # prefer preprocessed files if present
    def preproc_path(p):
        folder = os.path.dirname(p)
        name = "preprocessed_" + os.path.basename(p)
        candidate = os.path.join(folder, name)
        return candidate if os.path.exists(candidate) else p

    p1p = preproc_path(p1)
    p2p = preproc_path(p2)
    p3p = preproc_path(p3)

    logging.info("Loading S1 from %s", p1p)
    df1 = pd.read_csv(p1p, sep="\t", dtype=str)
    logging.info("Loading S2 from %s", p2p)
    df2 = pd.read_csv(p2p, sep="\t", dtype=str)
    logging.info("Loading S3 from %s", p3p)
    df3 = pd.read_csv(p3p, sep="\t", dtype=str)

    # unify column names: pick id col
    id_col = "entity_id" if "entity_id" in df1.columns else df1.columns[0]
    name_col = "name_norm" if "name_norm" in df1.columns else next((c for c in df1.columns if "name" in c.lower()), df1.columns[1])

    # We'll block S1 vs S2 and S1 vs S3 separately and union results
    logging.info("Blocking S1->S2 (k=%d cap=%d)", k, cap)
    maps = []

    # PIN block S1->S2
    maps.append(pin_block(df1, df2, id_col=id_col, pin_col="pin"))

    # prefix/sig block S1->S2
    maps.append(prefix_sig_block(df1, df2, id_col=id_col, sig_col="name_sig"))

    # phonetic block S1->S2
    maps.append(phonetic_block(df1, df2, id_col=id_col, name_col=name_col))

    # tfidf ann block S1->S2
    maps.append(tfidf_ann_block(df1, df2, id_col=id_col, name_col=name_col, k=k, cache_key=f"tfidf_s2_{split}"))

    maps.append(address_sig_block(df1, df2, id_col))
    maps.append(geo_block(df1, df2, id_col))

    union_s12 = union_candidate_maps(maps)
    capped_s12 = cap_candidates(union_s12, cap=cap)

    # Repeat for S1 -> S3
    logging.info("Blocking S1->S3 (k=%d cap=%d)", k, cap)
    maps = []
    maps.append(pin_block(df1, df3, id_col=id_col, pin_col="pin"))
    maps.append(prefix_sig_block(df1, df3, id_col=id_col, sig_col="name_sig"))
    maps.append(phonetic_block(df1, df3, id_col=id_col, name_col=name_col))
    maps.append(tfidf_ann_block(df1, df3, id_col=id_col, name_col=name_col, k=k, cache_key=f"tfidf_s3_{split}"))
    maps.append(address_sig_block(df1, df3, id_col))
    maps.append(geo_block(df1, df3, id_col))

    union_s13 = union_candidate_maps(maps)
    capped_s13 = cap_candidates(union_s13, cap=cap)

    # merge both S2 and S3 candidates into final candidate map per S1
    final_map = defaultdict(list)
    for s1, lst in capped_s12.items():
        final_map[s1].extend(lst)
    for s1, lst in capped_s13.items():
        final_map[s1].extend(lst)

    # dedupe each list
    for k_ in list(final_map.keys()):
        final_map[k_] = sorted(list(set(final_map[k_])))

    # write out candidate_pairs.tsv
    out_file = out_path if out_path else os.path.join(train_dir, "candidate_pairs.tsv")
    logging.info("Writing candidate pairs to %s", out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("source1_id\tcandidate_id\n")
        for s1, cands in final_map.items():
            for c in cands:
                f.write(f"{s1}\t{c}\n")

    logging.info("Wrote %d candidate pairs (rows) for %d source1 entities", sum(len(v) for v in final_map.values()), len(final_map))

    # validation
    if validate and p_gt:
        gt_df = pd.read_csv(p_gt, sep="\t", dtype=str)
        recall, diag = compute_blocking_recall(final_map, gt_df)
        logging.info("Blocking recall: %.4f (%d/%d). Diagnostics: %s", recall, diag["found_pairs"], diag["total_pairs"], diag["missing_examples_sample"])
        # Also write small diagnostics file
        with open(os.path.join(train_dir, "blocking_diag.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"recall\t{recall}\n")
            fh.write(f"found\t{diag['found_pairs']}\n")
            fh.write(f"total\t{diag['total_pairs']}\n")
            fh.write("missing_examples_sample:\n")
            for k, v in diag["missing_examples_sample"].items():
                fh.write(f"{k}\t{v}\n")

    logging.info("Blocking complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-pass blocking (S1->S2/S3)")
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--split", type=str, default="small", choices=["small", "medium", "full"])
    parser.add_argument("--k", type=int, default=100, help="top-K neighbors for TF-IDF ANN")
    parser.add_argument("--cap", type=int, default=500, help="max candidates per S1")
    parser.add_argument("--out", type=str, default=None, help="output candidate_pairs.tsv path")
    parser.add_argument("--no-validate", dest="validate", action="store_false", help="skip validation against ground truth")
    args = parser.parse_args()
    run_blocking(args.dataset, args.split, k=args.k, cap=args.cap, validate=args.validate, out_path=args.out)