import pandas as pd
import os

# Paths
DATASET_PATH = "dataset/train"
OUTPUT_PATH = "dataset"

# Load data
s1 = pd.read_csv(f"{DATASET_PATH}/train_source1.tsv", sep="\t")
s2 = pd.read_csv(f"{DATASET_PATH}/train_source2.tsv", sep="\t")
s3 = pd.read_csv(f"{DATASET_PATH}/train_source3.tsv", sep="\t")
gt = pd.read_csv(f"{DATASET_PATH}/train_ground_truth.tsv", sep="\t")

# -----------------------------
# Function to create split
# -----------------------------
def create_split(sample_ratio, split_name):
    print(f"\nCreating {split_name} dataset ({sample_ratio*100}%)")

    # Step 1: Sample Source1 entities
    s1_sample = s1.sample(frac=sample_ratio, random_state=42)

    # Get selected entity IDs
    selected_s1_ids = set(s1_sample["entity_id"])

    # Step 2: Filter ground truth
    gt_sample = gt[gt["source1_entity_id"].isin(selected_s1_ids)].copy()

    # Step 3: Collect all matched IDs from GT
    def extract_ids(x):
        if pd.isna(x) or x == "":
            return []
        return [i.strip() for i in str(x).split(",")]

    matched_ids = set()
    for ids in gt_sample["matched_entity_ids"]:
        matched_ids.update(extract_ids(ids))

    # Step 4: Filter Source2 and Source3
    s2_sample = s2[s2["entity_id"].isin(matched_ids)]
    s3_sample = s3[s3["entity_id"].isin(matched_ids)]

    # -----------------------------
    # Save files
    # -----------------------------
    save_path = f"{OUTPUT_PATH}/{split_name}/train"
    os.makedirs(save_path, exist_ok=True)

    s1_sample.to_csv(f"{save_path}/train_source1.tsv", sep="\t", index=False)
    s2_sample.to_csv(f"{save_path}/train_source2.tsv", sep="\t", index=False)
    s3_sample.to_csv(f"{save_path}/train_source3.tsv", sep="\t", index=False)
    gt_sample.to_csv(f"{save_path}/train_ground_truth.tsv", sep="\t", index=False)

    print(f"{split_name} dataset created at {save_path}")
    print(f"S1: {len(s1_sample)}, S2: {len(s2_sample)}, S3: {len(s3_sample)}")


# -----------------------------
# Create datasets
# -----------------------------
if __name__ == "__main__":
    create_split(0.05, "small")   # 5%
    create_split(0.30, "medium")  # 30%