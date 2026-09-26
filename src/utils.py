# project/src/utils.py
import os
import glob
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------
# Configurable dataset split
# -----------------------------
# Allowed values: "full" / "small" / "medium"
# Resolves to:
#   full   -> <dataset_root>/train
#   small  -> <dataset_root>/small/train
#   medium -> <dataset_root>/medium/train
DATA_SPLIT = "full"

SPLIT_TO_TRAIN_SUBDIR = {
    "full": os.path.join("train"),
    "small": os.path.join("small", "train"),
    "medium": os.path.join("medium", "train"),
}


def resolve_train_dir(dataset_root: str = "dataset", data_split: str = DATA_SPLIT) -> str:
    """
    Resolve the dynamic train base path for a given split.
    Keeps path logic in one place so all loaders stay in sync.
    """
    if data_split not in SPLIT_TO_TRAIN_SUBDIR:
        raise ValueError(
            f"Invalid DATA_SPLIT={data_split!r}. Must be one of {list(SPLIT_TO_TRAIN_SUBDIR)}"
        )
    return os.path.join(dataset_root, SPLIT_TO_TRAIN_SUBDIR[data_split])


def find_dataset_files(dataset_root: str = "dataset", data_split: str = DATA_SPLIT) -> Dict[str,str]:
    """
    Find expected train/test TSV files. Returns dictionary of paths.
    Looks for files like train_source1.tsv, test_source2.tsv, train_ground_truth.tsv

    Train files are loaded from the dynamic base path controlled by data_split:
      full   -> <dataset_root>/train
      small  -> <dataset_root>/small/train
      medium -> <dataset_root>/medium/train
    Test files always come from <dataset_root>/test.
    """
    dataset_root = os.path.abspath(dataset_root)
    train_dir = os.path.abspath(resolve_train_dir(dataset_root, data_split))
    logging.info("Using DATA_SPLIT=%s -> train_dir=%s", data_split, train_dir)
    files = {}
    # patterns to look for (train uses dynamic base path, test is fixed)
    patterns = {
        "train_source1": os.path.join(train_dir, "*train_source1*.tsv"),
        "train_source2": os.path.join(train_dir, "*train_source2*.tsv"),
        "train_source3": os.path.join(train_dir, "*train_source3*.tsv"),
        "train_ground_truth": os.path.join(train_dir, "*train_ground_truth*.tsv"),
        "test_source1": os.path.join(dataset_root, "test", "*test_source1*.tsv"),
        "test_source2": os.path.join(dataset_root, "test", "*test_source2*.tsv"),
        "test_source3": os.path.join(dataset_root, "test", "*test_source3*.tsv"),
    }
    for key, pattern in patterns.items():
        found = glob.glob(pattern)
        if found:
            files[key] = found[0]
        else:
            files[key] = None
            logging.warning("Could not find file for pattern: %s", pattern)
    return files

def load_tsv(path: str, use_chunks: bool = False, chunksize: int = 200_000, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load a TSV as pandas DataFrame. If use_chunks is True, returns concatenated frame computed chunkwise.
    Use nrows for quick tests.
    """
    if path is None:
        raise ValueError("path is None; file not found")
    logging.info("Loading TSV: %s (chunks=%s, nrows=%s)", path, use_chunks, nrows)
    if use_chunks:
        it = pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize, nrows=nrows)
        df = pd.concat(it, ignore_index=True)
    else:
        df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows)
    # ensure columns exist and unify types
    df.columns = [c.strip() for c in df.columns]
    return df

def safe_len(x):
    return 0 if x is None else len(x)

def basic_sanity_checks(paths: Dict[str,str], dataset_root: str = "dataset", use_chunks: bool = False, data_split: str = DATA_SPLIT) -> Dict[str,object]:
    """
    Load minimal data and run sanity checks. Returns a dictionary with stats and saves a short text report.
    NOTE: data_split is config-only (for logging/reproducibility); check logic is unchanged.
    """
    report = {}
    # load training sources
    logging.info("Finding dataset files under %s (split=%s -> %s)", dataset_root, data_split, resolve_train_dir(dataset_root, data_split))
    stats = {}
    # load small head for quick preview, then full load only for ground truth analysis if needed
    df_train_src = {}
    df_test_src = {}
    for src in ("train_source1", "train_source2", "train_source3"):
        p = paths.get(src)
        if p:
            try:
                # For memory-safety, use chunk mode only for full load if use_chunks True
                df = load_tsv(p, use_chunks=use_chunks, nrows=5000 if not use_chunks else None)
                df_train_src[src] = df
                stats[src] = {"rows_preview": len(df), "cols": list(df.columns)}
            except Exception as e:
                logging.exception("Failed to load %s: %s", p, e)
                df_train_src[src] = None
                stats[src] = {"error": str(e)}
        else:
            df_train_src[src] = None
            stats[src] = {"found": False}
    # load ground truth fully (small file)
    gt_path = paths.get("train_ground_truth")
    df_gt = None
    if gt_path:
        try:
            df_gt = load_tsv(gt_path, use_chunks=False)
            stats["train_ground_truth"] = {"rows": len(df_gt), "cols": list(df_gt.columns)}
        except Exception as e:
            logging.exception("Failed to load ground truth: %s", e)
            stats["train_ground_truth"] = {"error": str(e)}
    else:
        stats["train_ground_truth"] = {"found": False}

    # Basic checks: uniqueness & id prefixes
    for key, df in df_train_src.items():
        if df is None:
            continue
        eid_col = "entity_id" if "entity_id" in df.columns else df.columns[0]
        total = len(df)
        unique_ids = df[eid_col].nunique()
        stats[key].update({"total_rows": total, "unique_entity_ids": unique_ids})
        if unique_ids != total:
            stats[key]["unique_warning"] = f"{total - unique_ids} duplicate entity_id rows"

        # check prefix
        prefixes = df[eid_col].str.split("-", n=1, expand=True)[0].value_counts().to_dict()
        stats[key]["prefix_counts"] = prefixes

        # missingness
        missing = df.isna().sum().to_dict()
        stats[key]["missing_counts_preview"] = {k: missing.get(k,0) for k in list(df.columns)[:6]}

    # Ground truth checks: singletons, avg matches
    if df_gt is not None:
        # matched_entity_ids column may be empty string for singletons
        mcol = "matched_entity_ids" if "matched_entity_ids" in df_gt.columns else df_gt.columns[1]
        def parse_list(s):
            if pd.isna(s) or str(s).strip()=="":
                return []
            return [x.strip() for x in str(s).split(",") if x.strip()]
        df_gt["_matches_list"] = df_gt[mcol].apply(parse_list)
        df_gt["_n_matches"] = df_gt["_matches_list"].apply(len)
        singleton_count = int((df_gt["_n_matches"]==0).sum())
        avg_matches = float(df_gt["_n_matches"].mean())
        stats["ground_truth_matches"] = {
            "total_source1": len(df_gt),
            "singletons": singleton_count,
            "avg_matches_per_source1": avg_matches,
            "max_matches": int(df_gt["_n_matches"].max())
        }

    # Cross-check: all source1 ids appear in ground truth
    if df_train_src.get("train_source1") is not None and df_gt is not None:
        eid_col = "entity_id" if "entity_id" in df_train_src["train_source1"].columns else df_train_src["train_source1"].columns[0]
        s1_ids = set(df_train_src["train_source1"][eid_col].astype(str).tolist())
        gt_s1_ids = set(df_gt[df_gt.columns[0]].astype(str).tolist())
        missing_in_gt = s1_ids - gt_s1_ids
        stats["s1_vs_gt"] = {
            "s1_count": len(s1_ids),
            "gt_count": len(gt_s1_ids),
            "s1_missing_in_gt_count": len(missing_in_gt),
            "sample_missing_in_gt": list(list(missing_in_gt)[:5])
        }

    # Save report
    report["stats"] = stats
    return report

def write_report(report: Dict, out_path: str = "outputs/sanity_report.txt"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        f.write("SANITY REPORT\n")
        f.write("================\n")
        json.dump(report, f, indent=2)
    logging.info("Wrote sanity report to %s", out_path)

if __name__ == "__main__":
    # quick local test convenience — switch DATA_SPLIT to "small"/"medium" as needed
    files = find_dataset_files(data_split=DATA_SPLIT)
    rpt = basic_sanity_checks(files, use_chunks=False, data_split=DATA_SPLIT)
    write_report(rpt)
    logging.info("Done.")