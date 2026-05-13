import argparse
import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------- 0. Configuration ----------------------
ROOT_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = ROOT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = ROOT_DIR / "azure_lmm_trace_enhanced.csv.gz"
SUMMARY_OUTPUT_PATH = ROOT_DIR / "stage4_2_policy_summary.csv"
FTL_OUTPUT_PATH = ROOT_DIR / "stage4_2_ftl_by_type.csv"
FAIRNESS_OUTPUT_PATH = ROOT_DIR / "stage4_2_fairness_by_type.csv"
REQUEST_OUTPUT_PATH = ROOT_DIR / "stage4_2_request_results.csv.gz"
LATENCY_FIGURE_PATH = ROOT_DIR / "stage4_2_policy_latency_comparison.png"
FTL_FIGURE_PATH = ROOT_DIR / "stage4_2_ftl_by_type.png"
WAIT_FIGURE_PATH = ROOT_DIR / "stage4_2_wait_by_type.png"
FAIRNESS_FIGURE_PATH = ROOT_DIR / "stage4_2_max_wait_by_type.png"

LIGHT_MULTIMODAL_MAX_IMAGES = 1
IMAGE_TOKEN_EQUIVALENT = 1000
GPU_THROUGHPUT = 2500
STARVATION_SLOWDOWN_THRESHOLD = 10.0
EPSILON = 1e-9

# Heavy-Wait Safeguarded Priority parameters
HEAVY_WAIT_CAP_S = 15_000.0

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)
plt.switch_backend("Agg")
sns.set_theme(style="whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4.2 heavy-wait safeguarded scheduling evaluation."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N requests after trace sorting.",
    )
    parser.add_argument(
        "--heavy-wait-cap",
        type=float,
        default=HEAVY_WAIT_CAP_S,
        help="Force one heavy request when the oldest heavy request waits this many seconds.",
    )
    return parser.parse_args()


# ---------------------- 1. Data Loading ----------------------
def load_trace(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    if "request_time" in df.columns:
        arrival_ts = pd.to_datetime(df["request_time"], utc=True, format="mixed")
    elif "TIMESTAMP" in df.columns:
        arrival_ts = pd.to_datetime(df["TIMESTAMP"], utc=True, format="mixed")
    elif "arrival_time" in df.columns:
        arrival_ts = pd.to_datetime(df["arrival_time"], utc=True, format="mixed")
    else:
        raise ValueError(
            "No supported arrival-time column found. Expected one of: "
            "request_time, TIMESTAMP, arrival_time."
        )

    df = df.copy()
    df["arrival_timestamp"] = arrival_ts
    base_time = arrival_ts.min()
    df["arrival_time"] = (arrival_ts - base_time).dt.total_seconds()
    df["service_time"] = df["latency_est"].astype(float)
    df["carbon_kg"] = df["carbon_est"].astype(float) / 1000.0
    df["ftl_service_component"] = (
        df["ContextTokens"].astype(float)
        + df["NumImages"].astype(float) * IMAGE_TOKEN_EQUIVALENT
    ) / GPU_THROUGHPUT

    sort_index = np.argsort(df["arrival_time"].to_numpy(dtype=float), kind="mergesort")
    return df.iloc[sort_index].reset_index(drop=True)


def classify_request_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conditions = [
        df["NumImages"] == 0,
        (df["NumImages"] >= 1) & (df["NumImages"] <= LIGHT_MULTIMODAL_MAX_IMAGES),
        df["NumImages"] > LIGHT_MULTIMODAL_MAX_IMAGES,
    ]
    request_types = ["Text-only", "Light multimodal", "Heavy multimodal"]
    priorities = [1, 2, 3]

    df["request_type"] = np.select(conditions, request_types, default="Heavy multimodal")
    df["priority"] = np.select(conditions, priorities, default=3).astype(int)
    df["request_id"] = np.arange(len(df))

    return df


# ---------------------- 2. Shared Helpers ----------------------
def build_policy_frame(
    df: pd.DataFrame,
    policy_name: str,
    starts: np.ndarray,
    completion: np.ndarray,
    wait: np.ndarray,
) -> pd.DataFrame:
    result = df[
        [
            "request_id",
            "arrival_timestamp",
            "arrival_time",
            "NumImages",
            "ContextTokens",
            "GeneratedTokens",
            "is_multimodal",
            "request_type",
            "priority",
            "service_time",
            "carbon_kg",
            "ftl_service_component",
        ]
    ].copy()

    result["policy"] = policy_name
    result["start_time"] = starts
    result["wait_time"] = wait
    result["completion_time"] = completion
    result["total_latency"] = result["wait_time"] + result["service_time"]
    result["ftl"] = result["wait_time"] + result["ftl_service_component"]
    result["slowdown"] = result["total_latency"] / (result["service_time"] + EPSILON)
    result["starved"] = result["slowdown"] > STARVATION_SLOWDOWN_THRESHOLD

    return result


def build_wide_request_export(results: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "request_id",
        "arrival_timestamp",
        "arrival_time",
        "NumImages",
        "ContextTokens",
        "GeneratedTokens",
        "is_multimodal",
        "request_type",
        "priority",
        "service_time",
        "carbon_kg",
        "ftl_service_component",
    ]

    static = results[base_columns].drop_duplicates(subset=["request_id"])
    sort_index = np.argsort(static["request_id"].to_numpy(dtype=int), kind="mergesort")
    static = static.iloc[sort_index].reset_index(drop=True)

    policy_metrics = (
        results[
            [
                "request_id",
                "policy",
                "start_time",
                "wait_time",
                "completion_time",
                "total_latency",
                "ftl",
                "slowdown",
                "starved",
            ]
        ]
        .pivot(index="request_id", columns="policy")
    )

    policy_metrics.columns = [
        f"{metric}_{policy.lower().replace(' ', '_').replace('-', '_')}"
        for metric, policy in policy_metrics.columns
    ]
    policy_metrics = policy_metrics.reset_index()

    return static.merge(policy_metrics, on="request_id", how="left")


# ---------------------- 3. Baselines ----------------------
def simulate_fcfs(df: pd.DataFrame) -> pd.DataFrame:
    arrivals = df["arrival_time"].to_numpy(dtype=float)
    service = df["service_time"].to_numpy(dtype=float)

    starts = np.empty(len(df), dtype=float)
    wait = np.empty(len(df), dtype=float)
    completion = np.empty(len(df), dtype=float)

    next_free_time = 0.0
    for i in range(len(df)):
        starts[i] = max(arrivals[i], next_free_time)
        wait[i] = starts[i] - arrivals[i]
        completion[i] = starts[i] + service[i]
        next_free_time = completion[i]

    return build_policy_frame(df, "FCFS", starts, completion, wait)


def simulate_modality_priority(df: pd.DataFrame) -> pd.DataFrame:
    arrivals = df["arrival_time"].to_numpy(dtype=float)
    service = df["service_time"].to_numpy(dtype=float)
    priorities = df["priority"].to_numpy(dtype=int)
    n = len(df)

    starts = np.empty(n, dtype=float)
    wait = np.empty(n, dtype=float)
    completion = np.empty(n, dtype=float)

    queues = {1: deque(), 2: deque(), 3: deque()}
    next_arrival_idx = 0
    completed = 0
    current_time = 0.0

    while completed < n:
        if not any(queues.values()):
            current_time = max(current_time, arrivals[next_arrival_idx])

        while next_arrival_idx < n and arrivals[next_arrival_idx] <= current_time:
            req_idx = next_arrival_idx
            queues[priorities[req_idx]].append(req_idx)
            next_arrival_idx += 1

        selected_idx = None
        for priority in (1, 2, 3):
            if queues[priority]:
                selected_idx = queues[priority].popleft()
                break

        if selected_idx is None:
            current_time = arrivals[next_arrival_idx]
            continue

        starts[selected_idx] = current_time
        wait[selected_idx] = starts[selected_idx] - arrivals[selected_idx]
        completion[selected_idx] = starts[selected_idx] + service[selected_idx]
        current_time = completion[selected_idx]
        completed += 1

    return build_policy_frame(df, "Static Modality Priority", starts, completion, wait)


# ---------------------- 4. Heavy-Wait Safeguarded Priority ----------------------
def simulate_hwsp(df: pd.DataFrame, heavy_wait_cap_s: float) -> pd.DataFrame:
    """
    Static modality priority with one weak fairness override:
    if the oldest heavy request has waited beyond a fixed cap, serve exactly one
    heavy request next; otherwise follow the original static priority order.
    """
    arrivals = df["arrival_time"].to_numpy(dtype=float)
    service = df["service_time"].to_numpy(dtype=float)
    priorities = df["priority"].to_numpy(dtype=int)
    n = len(df)

    starts = np.empty(n, dtype=float)
    wait = np.empty(n, dtype=float)
    completion = np.empty(n, dtype=float)

    queues = {1: deque(), 2: deque(), 3: deque()}
    next_arrival_idx = 0
    completed = 0
    current_time = 0.0

    while completed < n:
        if not any(queues.values()):
            current_time = max(current_time, arrivals[next_arrival_idx])

        while next_arrival_idx < n and arrivals[next_arrival_idx] <= current_time:
            req_idx = next_arrival_idx
            queues[priorities[req_idx]].append(req_idx)
            next_arrival_idx += 1

        selected_idx = None
        if queues[3]:
            oldest_heavy_idx = queues[3][0]
            oldest_heavy_wait = current_time - arrivals[oldest_heavy_idx]
            if oldest_heavy_wait >= heavy_wait_cap_s:
                selected_idx = queues[3].popleft()

        if selected_idx is None:
            for priority in (1, 2, 3):
                if queues[priority]:
                    selected_idx = queues[priority].popleft()
                    break

        if selected_idx is None:
            current_time = arrivals[next_arrival_idx]
            continue

        starts[selected_idx] = current_time
        wait[selected_idx] = starts[selected_idx] - arrivals[selected_idx]
        completion[selected_idx] = starts[selected_idx] + service[selected_idx]
        current_time = completion[selected_idx]
        completed += 1

    return build_policy_frame(df, "HWSP", starts, completion, wait)


# ---------------------- 5. Evaluation Tables ----------------------
def summarize_policies(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("policy")
        .agg(
            avg_total_latency_s=("total_latency", "mean"),
            p95_total_latency_s=("total_latency", lambda x: np.percentile(x, 95)),
            p99_total_latency_s=("total_latency", lambda x: np.percentile(x, 99)),
            avg_ftl_s=("ftl", "mean"),
            avg_wait_time_s=("wait_time", "mean"),
            max_wait_time_s=("wait_time", "max"),
            avg_slowdown=("slowdown", "mean"),
            p99_slowdown=("slowdown", lambda x: np.percentile(x, 99)),
            starvation_rate=("starved", "mean"),
            total_carbon_kg=("carbon_kg", "sum"),
        )
        .reset_index()
    )
    return summary.round(6)


def summarize_ftl_by_type(results: pd.DataFrame) -> pd.DataFrame:
    table = (
        results.groupby(["policy", "request_type"])
        .agg(
            avg_total_latency_s=("total_latency", "mean"),
            avg_ftl_s=("ftl", "mean"),
            avg_wait_time_s=("wait_time", "mean"),
            requests=("request_id", "count"),
        )
        .reset_index()
    )

    type_order = ["Text-only", "Light multimodal", "Heavy multimodal"]
    policy_order = ["FCFS", "Static Modality Priority", "HWSP"]
    table["request_type"] = pd.Categorical(table["request_type"], categories=type_order, ordered=True)
    table["policy"] = pd.Categorical(table["policy"], categories=policy_order, ordered=True)
    sort_index = np.lexsort(
        (
            table["request_type"].cat.codes.to_numpy(),
            table["policy"].cat.codes.to_numpy(),
        )
    )
    return table.iloc[sort_index].reset_index(drop=True).round(6)


def summarize_fairness_by_type(results: pd.DataFrame) -> pd.DataFrame:
    table = (
        results.groupby(["policy", "request_type"])
        .agg(
            avg_wait_time_s=("wait_time", "mean"),
            p99_wait_time_s=("wait_time", lambda x: np.percentile(x, 99)),
            max_wait_time_s=("wait_time", "max"),
            avg_slowdown=("slowdown", "mean"),
            p99_slowdown=("slowdown", lambda x: np.percentile(x, 99)),
            starvation_rate=("starved", "mean"),
            requests=("request_id", "count"),
        )
        .reset_index()
    )

    type_order = ["Text-only", "Light multimodal", "Heavy multimodal"]
    policy_order = ["FCFS", "Static Modality Priority", "HWSP"]
    table["request_type"] = pd.Categorical(table["request_type"], categories=type_order, ordered=True)
    table["policy"] = pd.Categorical(table["policy"], categories=policy_order, ordered=True)
    sort_index = np.lexsort(
        (
            table["request_type"].cat.codes.to_numpy(),
            table["policy"].cat.codes.to_numpy(),
        )
    )
    return table.iloc[sort_index].reset_index(drop=True).round(6)


# ---------------------- 6. Visualization ----------------------
def plot_policy_summary(summary_table: pd.DataFrame) -> None:
    plot_df = summary_table.melt(
        id_vars="policy",
        value_vars=["avg_total_latency_s", "avg_ftl_s", "p99_total_latency_s"],
        var_name="metric",
        value_name="seconds",
    )
    metric_labels = {
        "avg_total_latency_s": "Mean Total Latency",
        "avg_ftl_s": "Mean FTL",
        "p99_total_latency_s": "P99 Total Latency",
    }
    plot_df["metric"] = plot_df["metric"].map(metric_labels)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=plot_df, x="metric", y="seconds", hue="policy", palette="Set2")
    ax.set_title("Stage 4.2 Policy Comparison", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=8)
    ax.legend(title="")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(LATENCY_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ftl_by_type(ftl_table: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=ftl_table,
        x="request_type",
        y="avg_ftl_s",
        hue="policy",
        palette="Set2",
    )
    ax.set_title("Mean First-Token Latency by Request Type", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("Seconds")
    ax.legend(title="")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(FTL_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_wait_by_type(ftl_table: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=ftl_table,
        x="request_type",
        y="avg_wait_time_s",
        hue="policy",
        palette="Set2",
    )
    ax.set_title("Mean Queueing Delay by Request Type", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("Seconds")
    ax.legend(title="")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(WAIT_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def plot_max_wait_by_type(fairness_table: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=fairness_table,
        x="request_type",
        y="max_wait_time_s",
        hue="policy",
        palette="Set2",
    )
    ax.set_title("Maximum Wait Time by Request Type", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("Seconds")
    ax.legend(title="")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(FAIRNESS_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------- 7. Main Entry ----------------------
def main() -> None:
    args = parse_args()

    print("=== Stage 4.2: Heavy-Wait Safeguarded Priority Evaluation ===")
    print(f"Loading enhanced trace from: {INPUT_PATH}")

    trace_df = load_trace(INPUT_PATH)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        trace_df = trace_df.iloc[: args.limit].copy().reset_index(drop=True)

    trace_df = classify_request_types(trace_df)

    print(f"Loaded {len(trace_df):,} requests")
    if args.limit is not None:
        print(f"Running in limited mode with the first {args.limit:,} requests")
    print(f"Heavy wait cap: {args.heavy_wait_cap:.0f} s")
    print("\nRequest-type distribution:")
    print(trace_df["request_type"].value_counts().rename_axis("request_type").to_frame("requests"))

    fcfs_results = simulate_fcfs(trace_df)
    static_priority_results = simulate_modality_priority(trace_df)
    hwsp_results = simulate_hwsp(trace_df, args.heavy_wait_cap)
    all_results = pd.concat(
        [fcfs_results, static_priority_results, hwsp_results],
        ignore_index=True,
    )

    summary_table = summarize_policies(all_results)
    ftl_table = summarize_ftl_by_type(all_results)
    fairness_table = summarize_fairness_by_type(all_results)
    request_export = build_wide_request_export(all_results)

    summary_table.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    ftl_table.to_csv(FTL_OUTPUT_PATH, index=False)
    fairness_table.to_csv(FAIRNESS_OUTPUT_PATH, index=False)
    request_export.to_csv(REQUEST_OUTPUT_PATH, index=False, compression="gzip")
    plot_policy_summary(summary_table)
    plot_ftl_by_type(ftl_table)
    plot_wait_by_type(ftl_table)
    plot_max_wait_by_type(fairness_table)

    print("\n=== Policy Comparison Summary ===")
    print(summary_table.to_string(index=False))

    print("\n=== Fairness Metrics by Policy and Request Type ===")
    print(fairness_table.to_string(index=False))

    print("\n=== Output Files ===")
    print(f"Summary table:   {SUMMARY_OUTPUT_PATH.relative_to(ROOT_DIR)}")
    print(f"FTL by type:     {FTL_OUTPUT_PATH.relative_to(ROOT_DIR)}")
    print(f"Fairness table:  {FAIRNESS_OUTPUT_PATH.relative_to(ROOT_DIR)}")
    print(f"Per-request:     {REQUEST_OUTPUT_PATH.relative_to(ROOT_DIR)}")
    print(f"Policy chart:    {LATENCY_FIGURE_PATH.relative_to(ROOT_DIR)}")
    print(f"FTL chart:       {FTL_FIGURE_PATH.relative_to(ROOT_DIR)}")
    print(f"Wait chart:      {WAIT_FIGURE_PATH.relative_to(ROOT_DIR)}")
    print(f"Max-wait chart:  {FAIRNESS_FIGURE_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
