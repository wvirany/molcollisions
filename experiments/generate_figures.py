import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from analyze_bo_results import compute_bo_trial_statistics, load_bo_predictions

# Set up Seaborn plotting style
sns.set_style(
    "darkgrid",
    {
        "axes.facecolor": ".95",
        "axes.edgecolor": "#000000",
        "grid.color": "#EBEBE7",
        "font.family": "serif",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "grid.alpha": 0.4,
    },
)

# Get the colorblind palette colors
colorblind_palette = sns.color_palette("colorblind")

blue = colorblind_palette[0]
reddish_orange = colorblind_palette[3]
green = colorblind_palette[2]

COLORS = {"exact": blue, "compressed": reddish_orange, "sortslice": green}

FILL_COLORS = {"exact": blue, "compressed": reddish_orange, "sortslice": green}


def parse_metric_tuple(metric_str):
    """Parse '(0.533, 0.004)' string into (mean, std) floats."""
    # Remove parentheses and split by comma
    clean_str = metric_str.strip("()")
    mean_str, std_str = clean_str.split(", ")
    return float(mean_str), float(std_str)


def get_filtered_df(df: pd.DataFrame, target: str = "ESR2", optimize_hp: bool = False):
    df_filtered = df[df["target"] == target][df["optimized params"] == optimize_hp]
    return df_filtered


def plot_performance_vs_fpdim(
    df,
    target: str = "ESR2",
    metric: str = "R2",
    optimize_hp: bool = True,
    ax=None,
    standalone: bool = False,
):
    """Plot performance vs fingerprint size for different FP types."""

    # Get fingerprint types
    fp_types = ["exact", "sortslice", "compressed"]
    fp_sizes = [512, 1024, 2048, 4096]

    for fp_type in fp_types:
        fp_data = df[df["fp type"] == fp_type]

        # Parse metrics
        metrics = [parse_metric_tuple(m) for m in fp_data[f"{metric} (mean, std)"]]
        means = np.array([m[0] for m in metrics])
        stds = np.array([m[1] for m in metrics])

        if fp_type == "exact":
            # ExactFP - plot as horizontal line across all sizes
            exact_mean = means[0]  # Should only be one value
            exact_std = stds[0]

            ax.axhline(y=exact_mean, color=COLORS[fp_type], label="Exact FP", linewidth=2)
            ax.fill_between(
                fp_sizes,
                exact_mean - exact_std,
                exact_mean + exact_std,
                color=FILL_COLORS[fp_type],
                alpha=0.1,
            )
        else:
            ax.plot(
                fp_sizes,
                means,
                color=COLORS[fp_type],
                marker="o",
                markersize=4,
                label=f"{fp_type.title()} FP",
                linewidth=2,
            )
            ax.fill_between(
                fp_sizes, means - stds, means + stds, color=FILL_COLORS[fp_type], alpha=0.1
            )

    # Formatting
    ax.set_xlim(fp_sizes[0] - 50, fp_sizes[-1] + 50)
    ax.set_xticks(fp_sizes)
    ax.set_title(f"Target: {target}")

    # Formatting options for standalone figure
    if standalone:
        ax.set_xlabel("Fingerprint size")
        if metric == "R2":
            ax.set_ylabel("$R^2$ Score")
        else:
            ax.set_ylabel(metric)
        ax.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=3)
        plt.tight_layout()

    return ax


def plot_two_targets(
    df, targets, metric: str = "R2", optimize_hp: bool = True, save_path: str = None
):
    """Plot two targets side by side."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, target in enumerate(targets):
        # Filter data
        filtered_df = df[(df["target"] == target) & (df["optimized params"] == optimize_hp)]

        plot_performance_vs_fpdim(
            df=filtered_df, target=target, metric=metric, optimize_hp=optimize_hp, ax=axes[i]
        )

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.03), loc="upper center", ncol=3)

    fig.supxlabel("Fingerprint size")
    if metric == "R2":
        fig.supylabel("$R^2$ score")
    else:
        fig.supylabel(metric)

    plt.tight_layout()

    plt.savefig(f"{save_path}.svg", format="svg", bbox_inches="tight")
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    print(f"Figure saved to {save_path}")


def make_all_regression_figures(results_df):
    """Generate all individual regression figures."""

    targets = results_df["target"].unique()
    metrics = ["R2", "MSE", "MAE"]
    optimize_hp_options = [True, False]

    for target in targets:
        for metric in metrics:
            for optimize_hp in optimize_hp_options:
                # Create target-specific directory
                save_dir = Path("figures/regression") / target
                save_dir.mkdir(parents=True, exist_ok=True)

                # Create standalone figure
                fig, ax = plt.subplots(figsize=(10, 6))
                filtered_df = results_df[
                    (results_df["target"] == target)
                    & (results_df["optimized params"] == optimize_hp)
                ]

                plot_performance_vs_fpdim(
                    filtered_df, target, metric, optimize_hp=optimize_hp, ax=ax, standalone=True
                )

                # Save figure with suffix for optimization
                suffix = "-opt" if optimize_hp else ""
                save_path = save_dir / f"{metric}_vs_fpdim{suffix}.png"
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Figure saved to {save_path}")


def make_regression_figures(results_df, make_all: bool = False, paper: bool = False):
    """Generate regression figures."""

    if make_all:
        make_all_regression_figures(results_df)

    if paper:
        targets = ["ESR2", "KIT"]
        metrics = ["R2"]

        figures_dir = Path("figures/paper")
        figures_dir.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            save_path = figures_dir / f"{targets[0]}-{targets[1]}_{metric}_vs_fpdim"
            plot_two_targets(
                results_df, targets=targets, metric=metric, optimize_hp=True, save_path=save_path
            )


def plot_bo_performance(
    target: str = "ESR2",
    acq: str = "ei",
    ax=None,
    standalone: bool = False,
    show_percentiles: bool = True,
):
    """Plot BO performance for a single target."""

    # Load config to get fingerprint configurations
    config_file = Path("configs/bo_experiments.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Aggregate statistics for each fingerprint config
    results = {}
    for fp_config in config["fingerprints"]:
        predictions = load_bo_predictions(target, acq, fp_config)
        stats = compute_bo_trial_statistics(predictions, target, acq, all_preds=True)
        results[fp_config] = stats

    # Compute dataset reference lines
    from molcollisions.datasets import Dockstring

    dataset = Dockstring(target, n_train=1000000)
    _, _, y_train, y_test = dataset.load()
    y = -np.concatenate([y_train, y_test])

    percentile999 = -np.quantile(y, 0.999)
    best_score = -np.max(y)

    # Plot results
    xs = np.arange(len(results["exact-r2"]["best_median"]))

    for fp_config in results:
        # Determine fingerprint type and size
        if "exact" in fp_config:
            fp_type = "exact"
            fp_size = None
        elif "compressed" in fp_config:
            fp_type = "compressed"
            fp_size = int(fp_config[10:14])
        elif "sortslice" in fp_config:
            fp_type = "sortslice"
            fp_size = int(fp_config[9:13])

        # Configure alpha and offsets based on dimension
        if fp_size == 1024:
            alpha = 0.35
        elif fp_size == 2048:
            alpha = 1.0
        else:
            alpha = 1.0

        # Sample data
        ys = results[fp_config]["best_median"]
        y_25 = results[fp_config]["best_25"]
        y_75 = results[fp_config]["best_75"]

        # Small offsets to prevent overlapping
        off = 5
        offset = {"exact": -off, "compressed": 0, "sortslice": off}
        xs_offset = xs + offset[fp_type]

        off_y = 0.02
        offset_y = {"exact": -off_y, "compressed": 0, "sortslice": off_y}
        ys_offset = ys + offset_y[fp_type]
        y_25_offset = y_25 + offset_y[fp_type]
        y_75_offset = y_75 + offset_y[fp_type]

        # Create label
        label = f"{fp_type.title()} FP"
        if fp_size:
            label += f" ({fp_size})"

        # Plot line
        ax.plot(xs_offset, ys_offset, color=COLORS[fp_type], alpha=alpha, label=label, linewidth=2)

        # Add confidence bands
        if show_percentiles:
            ax.fill_between(
                xs_offset, y_25_offset, y_75_offset, color=FILL_COLORS[fp_type], alpha=0.1
            )

    # Add reference lines
    ax.axhline(
        percentile999,
        color="red",
        linestyle="dashed",
        linewidth=0.75,
        label="$99.9^\\text{th}$ percentile",
    )
    ax.axhline(
        best_score, color="purple", linestyle="dashed", linewidth=0.75, label="Best Molecule"
    )

    # Formatting
    ax.set_xlim(-30, 1030)
    ax.set_title(f"Target: {target}")

    if standalone:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Docking score")
        ax.legend()
        plt.tight_layout()

    return ax


def plot_four_targets_bo(targets, acq: str = "ei", save_path: str = None):
    """Plot BO results for four targets in 2x2 grid."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, target in enumerate(targets):
        plot_bo_performance(target=target, acq=acq, ax=axes[i], show_percentiles=True)

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.02), loc="upper center", ncol=7)

    # Shared labels
    fig.supxlabel("Iteration")
    fig.supylabel("Docking score")

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}.svg", format="svg", bbox_inches="tight")
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")


def make_all_bo_figures(targets, acq: str = "ei"):
    """Generate individual BO figures for each target."""

    for target in targets:
        # Create target-specific directory
        save_dir = Path("figures/bo")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create standalone figure
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_bo_performance(target=target, acq=acq, ax=ax, standalone=True)

        # Save figure
        save_path = save_dir / f"{target}-best.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Figure saved to {save_path}")


def make_bo_figures(results_df, make_all: bool = False, paper: bool = False):

    if make_all:
        targets = results_df["target"].unique()
        acq = "ei"
        make_all_bo_figures(targets, acq=acq)

    if paper:
        targets = ["ESR2", "KIT", "PARP1", "PGR"]
        acq = "ei"

        figures_dir = Path("figures/paper")
        figures_dir.mkdir(parents=True, exist_ok=True)

        save_path = figures_dir / "best_mol-four_targets"

        plot_four_targets_bo(targets=targets, acq=acq, save_path=save_path)


def main(regression: bool = False, make_all: bool = False, paper: bool = False, bo: bool = False):

    if regression:
        # Load regression data
        regression_file = Path("results/regression_summary.csv")

        if not regression_file.exists():
            print(f"Error: {regression_file} not found. Run analyze_regression_results.py first.")
            return

        # Generate regression figures
        print("Generating regression figures...")
        regression_results_df = pd.read_csv(regression_file)
        make_regression_figures(regression_results_df, make_all=make_all, paper=paper)

    if bo:
        # Load BO data
        bo_file = Path("results/bo_summary.csv")

        if not bo_file.exists():
            print(f"Error: {bo_file} not found. Run analyze_bo_results.py first.")
            return

        # Generate BO figures
        print("Generating BO figures...")
        bo_results_df = pd.read_csv(bo_file)
        make_bo_figures(bo_results_df, make_all=make_all, paper=paper)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--make_all", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--bo", action="store_true")

    args = parser.parse_args()

    if args.regression and not (args.make_all or args.paper):
        raise ValueError("Must specify either --make_all or --paper when using --regression")

    main(regression=args.regression, make_all=args.make_all, paper=args.paper, bo=args.bo)
