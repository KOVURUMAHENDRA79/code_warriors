# project/src/main.py
import argparse
import logging
from src.utils import (
    DATA_SPLIT,
    SPLIT_TO_TRAIN_SUBDIR,
    find_dataset_files,
    basic_sanity_checks,
    resolve_train_dir,
    write_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Configurable dataset split: "full" / "small" / "medium"
#   full   -> dataset/train
#   small  -> dataset/small/train
#   medium -> dataset/medium/train
DEFAULT_DATA_SPLIT = DATA_SPLIT

def main(dataset_root: str, data_split: str = DEFAULT_DATA_SPLIT, use_chunks: bool = False, out: str = "outputs/sanity_report.txt"):
    train_dir = resolve_train_dir(dataset_root, data_split)
    logging.info("DATA_SPLIT=%s, train_dir=%s", data_split, train_dir)
    files = find_dataset_files(dataset_root, data_split=data_split)
    logging.info("Files discovered: %s", files)
    report = basic_sanity_checks(files, dataset_root, use_chunks=use_chunks, data_split=data_split)
    # record which split was used (reproducibility, no logic change)
    report["data_split"] = data_split
    report["train_dir"] = train_dir
    write_report(report, out_path=out)
    logging.info("Sanity check complete. Report at %s", out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset sanity checks for ER challenge")
    parser.add_argument("--dataset", type=str, default="dataset", help="root dataset folder (default: dataset)")
    parser.add_argument("--split", type=str, default=DEFAULT_DATA_SPLIT, choices=["full", "small", "medium"], help="dataset split to use: full (dataset/train), small (dataset/small/train), medium (dataset/medium/train)")
    parser.add_argument("--chunks", action="store_true", help="use chunked reads (safer for low RAM)")
    parser.add_argument("--out", type=str, default="outputs/sanity_report.txt", help="report output path")
    args = parser.parse_args()
    main(args.dataset, args.split, args.chunks, args.out)