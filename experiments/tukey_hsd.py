import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import r2_score

from molcollisions.datasets import Dockstring


def load_r2_scores(target: str, fp_config: str, optimize_hp: bool = False):
    """Load R2 scores for all trials of a specific configuration."""
    
    suffix = "-opt" if optimize_hp else ""
    results_path = Path("results/regression") / target / (fp_config + suffix)
    
    if not os.path.exists(results_path):
        return None
    
    trial_files = list(results_path.glob("trial_*.pkl"))
    if not trial_files:
        return None
    
    dataset = Dockstring(target)
    _, _, _, y_test = dataset.load()
    
    r2_scores = []
    for trial_file in sorted(trial_files):
        with open(trial_file, "rb") as f:
            preds = pickle.load(f)
            mean_preds = preds["mean_preds"]
            r2 = r2_score(y_test, mean_preds)
            r2_scores.append(r2)
    
    return np.array(r2_scores)


def compare_exact_to_others(target: str, optimize_hp: bool = False):
    """
    Compare exact-r2 fingerprint to all other fingerprint types.
    
    Returns a DataFrame with only comparisons involving exact-r2.
    """
    
    fp_configs = [
        "exact-r2",
        "compressed512-r2", 
        "compressed1024-r2",
        "compressed2048-r2",
        "compressed4096-r2",
        "sortslice512-r2",
        "sortslice1024-r2",
        "sortslice2048-r2",
        "sortslice4096-r2"
    ]
    
    # Load R2 scores and build dataframe
    all_data = []
    for fp_config in fp_configs:
        scores = load_r2_scores(target, fp_config, optimize_hp)
        if scores is not None:
            for score in scores:
                all_data.append({
                    'r2_score': score,
                    'fingerprint': fp_config
                })
    
    if len(all_data) == 0:
        print(f"No data for {target}")
        return None
    
    df = pd.DataFrame(all_data)
    
    # Check if we have exact-r2 data
    if 'exact-r2' not in df['fingerprint'].values:
        print(f"No exact-r2 data for {target}")
        return None
    
    # Run ANOVA first
    groups = [group['r2_score'].values for name, group in df.groupby('fingerprint')]
    f_stat, p_value_anova = f_oneway(*groups)
    
    print(f"\n{'='*70}")
    print(f"Target: {target} (optimize_hp={optimize_hp})")
    print(f"{'='*70}")
    print(f"ANOVA: F={f_stat:.4f}, p={p_value_anova:.4e}")
    
    if p_value_anova > 0.05:
        print("  → No significant differences detected by ANOVA (α=0.05)")
    else:
        print("  → Significant differences detected by ANOVA")
    
    # Run Tukey's HSD
    tukey_result = pairwise_tukeyhsd(
        endog=df['r2_score'],
        groups=df['fingerprint'],
        alpha=0.05
    )
    
    # Convert to DataFrame
    results_df = pd.DataFrame(data=tukey_result.summary().data[1:], 
                              columns=tukey_result.summary().data[0])
    
    # Filter to only comparisons involving exact-r2
    exact_comparisons = results_df[
        (results_df['group1'] == 'exact-r2') | 
        (results_df['group2'] == 'exact-r2')
    ].copy()
    
    # Reorder so exact-r2 is always in group1 for consistency
    def swap_if_needed(row):
        if row['group2'] == 'exact-r2':
            return pd.Series({
                'group1': row['group2'],
                'group2': row['group1'],
                'meandiff': -float(row['meandiff']),
                'p-adj': row['p-adj'],
                'lower': -float(row['upper']),
                'upper': -float(row['lower']),
                'reject': row['reject']
            })
        return row
    
    exact_comparisons = exact_comparisons.apply(swap_if_needed, axis=1)
    exact_comparisons = exact_comparisons.sort_values('meandiff', ascending=False)
    
    # Print results
    print(f"\nComparisons with exact-r2 (positive = exact is better):")
    print("-" * 70)
    for _, row in exact_comparisons.iterrows():
        sig_marker = "***" if row['reject'] else "   "
        print(f"{sig_marker} {row['group1']:20s} vs {row['group2']:20s} | "
              f"Δ={float(row['meandiff']):7.4f} | p={float(row['p-adj']):.4f}")
    
    return exact_comparisons, f_stat, p_value_anova


def analyze_all_targets(optimize_hp: bool = False):
    """Run comparisons for all targets."""
    
    targets = ["ESR2", "F2", "KIT", "PARP1", "PGR"]
    
    all_results = {}
    for target in targets:
        result = compare_exact_to_others(target, optimize_hp=optimize_hp)
        if result is not None:
            df, f_stat, p_anova = result
            all_results[target] = {
                'df': df,
                'f_stat': f_stat,
                'p_anova': p_anova
            }
    
    return all_results


def create_summary_table(all_results):
    """Create a summary table showing which comparisons are significant."""
    
    summary = []
    for target, data in all_results.items():
        df = data['df']
        for _, row in df.iterrows():
            summary.append({
                'Target': target,
                'Comparison': f"exact vs {row['group2'].replace('-r2', '')}",
                'Mean Diff': float(row['meandiff']),
                'p-value': float(row['p-adj']),
                'Significant': 'Yes' if row['reject'] else 'No'
            })
    
    summary_df = pd.DataFrame(summary)
    return summary_df


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TUKEY'S HSD: EXACT-R2 vs OTHER FINGERPRINTS")
    print("="*70)
    
    print("\n" + "="*70)
    print("NON-OPTIMIZED HYPERPARAMETERS")
    print("="*70)
    results_nonopt = analyze_all_targets(optimize_hp=False)
    
    print("\n" + "="*70)
    print("OPTIMIZED HYPERPARAMETERS")
    print("="*70)
    results_opt = analyze_all_targets(optimize_hp=True)
    
    # Create summary tables
    
    summary_nonopt = create_summary_table(results_nonopt)
    summary_nonopt.to_csv("results/tukey/summary_nonopt.csv", index=False)

    summary_opt = create_summary_table(results_opt)
    summary_opt.to_csv("results/tukey/summary_opt.csv", index=False)