# src/features.py

import numpy as np
import pandas as pd
from collections import Counter


# ---------- UTIL ----------

def to_set(x):
    if isinstance(x, str):
        return set(x.split())
    return set(x or [])


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap(a, b):
    return len(a & b)


# ---------- PHONETIC (SIMPLE SOUNDEX) ----------

def soundex(name):
    name = name.lower()
    if not name:
        return ""

    mapping = {
        "bfpv": "1",
        "cgjkqsxz": "2",
        "dt": "3",
        "l": "4",
        "mn": "5",
        "r": "6"
    }

    def code(c):
        for k in mapping:
            if c in k:
                return mapping[k]
        return "0"

    first = name[0].upper()
    encoded = [code(c) for c in name[1:]]

    # remove duplicates
    filtered = []
    prev = ""
    for e in encoded:
        if e != prev and e != "0":
            filtered.append(e)
        prev = e

    return (first + "".join(filtered) + "000")[:4]


# ---------- LIGHT EMBEDDING (CHAR VECTOR) ----------

def char_vector(s):
    vec = Counter(s)
    keys = sorted(vec.keys())
    return np.array([vec[k] for k in keys], dtype=float), keys


def cosine_from_counter(s1, s2):
    v1 = Counter(s1)
    v2 = Counter(s2)

    all_keys = set(v1.keys()) | set(v2.keys())

    a = np.array([v1.get(k, 0) for k in all_keys])
    b = np.array([v2.get(k, 0) for k in all_keys])

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- MAIN FEATURE BUILDER ----------

def build_features(candidate_map, df_s1, df_candidates, gt_df=None):

    s1_lookup = df_s1.set_index("entity_id")
    c_lookup = df_candidates.set_index("entity_id")

    # preprocess tokens
    s1_lookup["name_tokens"] = s1_lookup["name_tokens"].apply(to_set)
    s1_lookup["address_tokens"] = s1_lookup["address_tokens"].apply(to_set)

    c_lookup["name_tokens"] = c_lookup["name_tokens"].apply(to_set)
    c_lookup["address_tokens"] = c_lookup["address_tokens"].apply(to_set)

    # phonetic precompute
    s1_lookup["soundex"] = s1_lookup["name_norm"].apply(soundex)
    c_lookup["soundex"] = c_lookup["name_norm"].apply(soundex)

    # ground truth
    gt_map = {}
    if gt_df is not None:
        for i in range(len(gt_df)):
            s1 = gt_df.iloc[i, 0]
            gt_map[s1] = set(str(gt_df.iloc[i, 1]).split(","))

    X = []
    y = []

    for s1_id, cands in candidate_map.items():

        if s1_id not in s1_lookup.index:
            continue

        s1 = s1_lookup.loc[s1_id]

        for c in cands:

            if c not in c_lookup.index:
                continue

            cand = c_lookup.loc[c]

            # ---------- BASE FEATURES ----------
            name_jacc = jaccard(s1["name_tokens"], cand["name_tokens"])
            name_overlap = overlap(s1["name_tokens"], cand["name_tokens"])
            name_len_diff = abs(len(s1["name_tokens"]) - len(cand["name_tokens"]))

            addr_jacc = jaccard(s1["address_tokens"], cand["address_tokens"])
            addr_overlap = overlap(s1["address_tokens"], cand["address_tokens"])

            pin_match = 1 if s1.get("pin", "") and s1["pin"] == cand.get("pin", "") else 0
            same_country = 1 if s1.get("country", "") == cand.get("country", "") else 0

            # ---------- PHONETIC ----------
            phonetic_match = 1 if s1["soundex"] == cand["soundex"] else 0

            # ---------- LIGHT EMBEDDING ----------
            char_sim = cosine_from_counter(
                s1.get("name_norm", ""),
                cand.get("name_norm", "")
            )

            features = [
                name_jacc,
                name_overlap,
                name_len_diff,
                addr_jacc,
                addr_overlap,
                pin_match,
                same_country,
                phonetic_match,
                char_sim
            ]

            X.append(features)

            if gt_df is not None:
                label = 1 if (s1_id in gt_map and c in gt_map[s1_id]) else 0
                y.append(label)

    X = np.array(X, dtype=float)

    if gt_df is not None:
        return X, np.array(y, dtype=int)

    return X, None