import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem import DataStructs

from molcollisions.datasets import Dockstring
from molcollisions.fingerprints import ExactFP, CompressedFP

TARGET = "ESR2"
FP_SIZES = [512, 1024, 2048, 4096]


def compute_pairwise_collisions(fp1_exact, fp2_exact, fp_size):
    """Count how many different Morgan IDs collide between two molecules."""
    morgan_ids_1 = set(fp1_exact.GetNonzeroElements().keys())
    morgan_ids_2 = set(fp2_exact.GetNonzeroElements().keys())
    
    num_collisions = 0
    for mid1 in morgan_ids_1:
        for mid2 in morgan_ids_2:
            if mid1 != mid2 and (mid1 % fp_size) == (mid2 % fp_size):
                num_collisions += 1

    return num_collisions


def compute_collisions(smiles_list, fp_size, radius=2):

    exact_fp = ExactFP(radius=radius)

    all_morgan_ids = set()

    for i, smiles in enumerate(smiles_list):
        fp = exact_fp(smiles)
        elements = fp.GetNonzeroElements()

        for morgan_id in elements.keys():
            all_morgan_ids.add(morgan_id)

    num_unique_substructures = len(all_morgan_ids)

    results = {}

    # Map each Morgan ID to its hash bucket
    hash_to_morgan_ids = defaultdict(set)

    for morgan_id in all_morgan_ids:
        hash_value = morgan_id % fp_size
        hash_to_morgan_ids[hash_value].add(morgan_id)

    num_occupied_buckets = len(hash_to_morgan_ids)
    num_buckets_with_collisions = sum(1 for ids in hash_to_morgan_ids.values() if len(ids) > 1)

    # Total number of collisions = sum(# IDs in bucket - 1)
    total_collisions = sum(len(ids) - 1 for ids in hash_to_morgan_ids.values())

    # Average number of Morgan IDs per bucket
    avg_morgan_ids_per_bucket = np.mean([len(ids) for ids in hash_to_morgan_ids.values()])
    max_morgan_ids_per_bucket = max([len(ids) for ids in hash_to_morgan_ids.values()])

    # Collision rate: what fraction of unique substructures are involved in collisions
    num_structures_in_collisions = sum(
        len(ids) for ids in hash_to_morgan_ids.values() if len(ids) > 1
    )
    collision_rate = num_structures_in_collisions / num_unique_substructures

    # Load factor: how "full" is the hash table? Number of elements in hash table / number of buckets
    load_factor = num_unique_substructures / fp_size

    results = {
        "fp_size": fp_size,
        "unique_substructures": num_unique_substructures,
        "occupied_buckets": num_occupied_buckets,
        "buckets_with_collisions": num_buckets_with_collisions,
        "total_collisions": total_collisions,
        "collision_rate": collision_rate,
        "avg_morgan_ids_per_bucket": avg_morgan_ids_per_bucket,
        "max_morgan_ids_per_bucket": max_morgan_ids_per_bucket,
        "load_factor": load_factor,
    }

    return results


def analyze_dataset():

    train_results = []
    test_results = []
    full_results = []

    # Analyze training set for different seeds
    for seed in range(10):
        print(f"Loading dataset with seed {seed}...")

        dataset = Dockstring(target=TARGET, n_train=10000, seed=seed)
        smiles_train, smiles_test, _, _ = dataset.load()

        for fp_size in FP_SIZES:
            results = compute_collisions(smiles_train, fp_size)
            results["seed"] = seed
            train_results.append(results)

    # Analyze test set
    for fp_size in FP_SIZES:
        results = compute_collisions(smiles_test, fp_size)
        test_results.append(results)

    # Analyze full dataset
    dataset = Dockstring(target=TARGET, n_train=1000000)
    smiles_train, smiles_test, _, _ = dataset.load()
    all_smiles = np.concatenate([smiles_train, smiles_test])

    for fp_size in FP_SIZES:
        results = compute_collisions(all_smiles, fp_size)
        full_results.append(results)

    train_df_raw = pd.DataFrame(train_results)
    test_df = pd.DataFrame(test_results)
    full_df = pd.DataFrame(full_results)

    train_summary = []
    for fp_size in FP_SIZES:
        fp_data = train_df_raw[train_df_raw["fp_size"] == fp_size]

        summary = {
            "fp_size": fp_size,
            "unique_substructures_mean": fp_data["unique_substructures"].mean(),
            "unique_substructures_std": fp_data["unique_substructures"].std(),
            "occupied_buckets_mean": fp_data["occupied_buckets"].mean(),
            "occupied_buckets_std": fp_data["occupied_buckets"].std(),
            "buckets_with_collisions_mean": fp_data["buckets_with_collisions"].mean(),
            "buckets_with_collisions_std": fp_data["buckets_with_collisions"].std(),
            "total_collisions_mean": fp_data["total_collisions"].mean(),
            "total_collisions_std": fp_data["total_collisions"].std(),
            "collision_rate_mean": fp_data["collision_rate"].mean(),
            "collision_rate_std": fp_data["collision_rate"].std(),
            "avg_morgan_ids_per_bucket_mean": fp_data["avg_morgan_ids_per_bucket"].mean(),
            "avg_morgan_ids_per_bucket_std": fp_data["avg_morgan_ids_per_bucket"].std(),
            "max_morgan_ids_per_bucket_mean": fp_data["max_morgan_ids_per_bucket"].mean(),
            "max_morgan_ids_per_bucket_std": fp_data["max_morgan_ids_per_bucket"].std(),
            "load_factor_mean": fp_data["load_factor"].mean(),
            "load_factor_std": fp_data["load_factor"].std(),
        }
        train_summary.append(summary)

    train_df = pd.DataFrame(train_summary)

    output_path = Path("results/collisions/")
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train.csv", index=False, float_format="%.4f")
    test_df.to_csv(output_path / "test.csv", index=False, float_format="%.4f")
    full_df.to_csv(output_path / "full.csv", index=False, float_format="%.4f")


def analyze_pairs(num_pairs=10000):
    
    dataset = Dockstring(target=TARGET, n_train=1000)
    smiles_list, _, _, _ = dataset.load()

    n_mols = len(smiles_list)
    pairs = []
    for _ in range(num_pairs):
        idx1, idx2 = np.random.choice(n_mols, size=2, replace=False)
        pairs.append((smiles_list[idx1], smiles_list[idx2]))

    exact_fp = ExactFP(radius=2)
    compressed_fps = {size: CompressedFP(radius=2, fp_size=size) for size in FP_SIZES}

    # Analyze each pair
    results_list = []
    for i, (smiles_1, smiles_2) in enumerate(pairs):
        fp1_exact = exact_fp(smiles_1)
        fp2_exact = exact_fp(smiles_2)
        tanimoto_exact = DataStructs.TanimotoSimilarity(fp1_exact, fp2_exact)

        for fp_size in FP_SIZES:
            fp1_compressed = compressed_fps[fp_size](smiles_1)
            fp2_compressed = compressed_fps[fp_size](smiles_2)
            tanimoto_compressed = DataStructs.TanimotoSimilarity(fp1_compressed, fp2_compressed)
            num_collisions = compute_pairwise_collisions(fp1_exact, fp2_exact, fp_size)
            results = {
                'pair_idx': i,
                'fp_size': fp_size,
                'num_collisions': num_collisions,
                'tanimoto_exact': tanimoto_exact,
                'tanimoto_compressed': tanimoto_compressed,
                'tanimoto_compressed_difference': tanimoto_compressed - tanimoto_exact,
            }
            results_list.append(results)
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(Path("results/collisions/pairs.csv"), index=False, float_format="%.4f")

    return results_df


def create_pairwise_summary(results_df):
    summary = {}
    for fp_size in FP_SIZES:
        fp_data = results_df[results_df["fp_size"] == fp_size]
        num_collisions_mean = fp_data["num_collisions"].mean()
        num_collisions_median = fp_data["num_collisions"].median()
        num_collisions_std = fp_data["num_collisions"].std()

        tanimoto_exact_mean = fp_data["tanimoto_exact"].mean()
        tanimoto_compressed_mean = fp_data["tanimoto_compressed"].mean()

        tanimoto_difference_mean = fp_data["tanimoto_compressed_difference"].mean()
        tanimoto_difference_median = fp_data["tanimoto_compressed_difference"].median()
        tanimoto_difference_std = fp_data["tanimoto_compressed_difference"].std()

        summary[fp_size] = {
            'num_collisions_mean': num_collisions_mean,
            'num_collisions_median': num_collisions_median,
            'num_collisions_std': num_collisions_std,
            'tanimoto_exact_mean': tanimoto_exact_mean,
            'tanimoto_compressed_mean': tanimoto_compressed_mean,
            'tanimoto_difference_mean': tanimoto_difference_mean,
            'tanimoto_difference_median': tanimoto_difference_median,
            'tanimoto_difference_std': tanimoto_difference_std,
        }
    
    df = pd.DataFrame(summary)
    df.to_csv(Path("results/collisions/pairs_summary.csv"), index=False, float_format="%.4f")


def main(args):

    if args.dataset:
        analyze_dataset()
    elif args.pairs:
        results_df = analyze_pairs()
        create_pairwise_summary(results_df)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--pairs", action="store_true")
    args = parser.parse_args()

    main(args)
