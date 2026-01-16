import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# General settings
# -----------------------------
BASE_DIR = ""          # Leave empty if the project is in the current folder
LOG_DIR = "tmp/logs"   # Same directory used in setup.json
PLOTS_DIR = "plots"    # Output directory for generated plots


def find_runs(log_root):
    """
    Find all runs that contain an episodes.csv file.
    Each run folder looks like:
    tmp/logs/wandb/<agent>-20251123-123456/
    and contains episodes.csv and summary.json.
    """
    runs = []
    for root, dirs, files in os.walk(log_root):
        if "episodes.csv" in files:
            episodes_path = os.path.join(root, "episodes.csv")
            summary_path = os.path.join(root, "summary.json")
            run_name = os.path.basename(root)

            # Extract algorithm name from the beginning of the run folder name (e.g., dqn-agent-2025...)
            algo = run_name.split("-")[0]

            env_name = "unknown"
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                    params = summary.get("params", {})
                    env_name = params.get("env", "unknown")
                except Exception:
                    pass

            runs.append(
                {
                    "run_dir": root,
                    "episodes_path": episodes_path,
                    "summary_path": summary_path if os.path.exists(summary_path) else None,
                    "algo": algo,
                    "env": env_name,
                }
            )
    return runs


def load_all_episodes(log_root):
    """
    Load all episodes.csv files and merge them into a single DataFrame.
    Important columns include:
    idx, type, completions_avg, deadlocks_avg, avg_delay_avg, scores_avg, ...
    plus env and algo.
    """
    runs = find_runs(log_root)
    all_dfs = []

    if not runs:
        print("No episodes.csv found. Are you sure you executed run_experiments.py?")
        return None

    for r in runs:
        ep_path = r["episodes_path"]
        algo = r["algo"]
        env = r["env"]

        try:
            df = pd.read_csv(ep_path)
        except Exception as e:
            print(f"Error reading {ep_path}: {e}")
            continue

        # Keep only episode rows (type == 'epsd')
        if "type" in df.columns:
            df = df[df["type"] == "epsd"].copy()

        df["algo"] = algo
        df["env"] = env

        all_dfs.append(df)

    if not all_dfs:
        print("No valid data loaded from episodes.csv files.")
        return None

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


def ensure_plots_dir():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_metric_by_env(df, metric_base_name, env_name, ylabel=None):
    """
    For a specific environment, plot the metric against episodes for each algorithm.
    metric_base_name examples: 'completions', 'deadlocks', 'avg_delay', 'scores', 'loss'
    The column searched in episodes.csv is <metric>_avg.
    """
    col_avg = metric_base_name + "_avg"
    if col_avg not in df.columns:
        print(f"Column {col_avg} not found in data, skipping this metric.")
        return

    subset = df[df["env"] == env_name]
    if subset.empty:
        print(f"No data found for env = {env_name}.")
        return

    plt.figure(figsize=(8, 5))
    for algo, grp in subset.groupby("algo"):
        grp_sorted = grp.sort_values("idx")
        plt.plot(grp_sorted["idx"], grp_sorted[col_avg], label=algo)
        
        
    plt.xlabel("Episode")
    plt.ylabel(ylabel if ylabel else metric_base_name)
    plt.title(f"{metric_base_name} vs Episodes  ({env_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"{metric_base_name}_{env_name}.png".replace(" ", "_")
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f" Plot for {metric_base_name} ({env_name}) saved at {save_path}.")


def summary_table(df, metric_base_name, last_n=50):
    """
    Compute the mean of a metric over the last N episodes for each (env, algo).
    Returns a pivoted DataFrame for reporting.
    """
    col_avg = metric_base_name + "_avg"
    if col_avg not in df.columns:
        print(f"Column {col_avg} not found, cannot build summary.")
        return None

    summary_rows = []
    for (env, algo), grp in df.groupby(["env", "algo"]):
        grp_sorted = grp.sort_values("idx")
        tail = grp_sorted.tail(last_n)
        if tail.empty:
            continue
        mean_val = tail[col_avg].mean()
        summary_rows.append({"env": env, "algo": algo, metric_base_name: mean_val})

    if not summary_rows:
        print(f"No summary rows computed for {metric_base_name}.")
        return None

    summary_df = pd.DataFrame(summary_rows)
    pivot = summary_df.pivot(index="env", columns="algo", values=metric_base_name)
    return pivot


def plot_bar_summary(pivot_df, metric_base_name):
    """
    Using the pivot summary table, create a bar chart comparing algorithms per environment.
    """
    if pivot_df is None:
        return

    plt.figure(figsize=(10, 6))
    pivot_df.plot(kind="bar")
    plt.ylabel(metric_base_name)
    plt.title(f"{metric_base_name} (last episodes, mean)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    filename = f"summary_{metric_base_name}.png"
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f" Summary bar plot for {metric_base_name} saved at {save_path}.")


def main():
    log_root = os.path.join(BASE_DIR, LOG_DIR)
    ensure_plots_dir()

    print(f"Reading logs from: {log_root}")
    df = load_all_episodes(log_root)
    if df is None:
        return

    # List of available environments
    envs = sorted(df["env"].unique())
    print("Available environments:", envs)

    # Metrics to plot
    metrics = [
        ("completions", "Completion rate"),
        ("deadlocks", "Deadlock rate"),
        ("avg_delay", "Avg delay"),
        ("scores", "Score"),
        ("loss", "Loss"),
    ]

    # Line plots for each env and each metric
    for env in envs:
        for metric_base, ylabel in metrics:
            plot_metric_by_env(df, metric_base, env, ylabel=ylabel)

    # Summary bar charts for last N episodes
    last_n = 50
    for metric_base, _ in metrics:
        pivot = summary_table(df, metric_base, last_n=last_n)
        if pivot is not None:
            print(f"\nSummary {metric_base} (mean of last {last_n} episodes):")
            print(pivot)
            plot_bar_summary(pivot, metric_base)


if __name__ == "__main__":
    main()
