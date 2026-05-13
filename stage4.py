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
SUMMARY_OUTPUT_PATH = ROOT_DIR / "stage4_policy_summary.csv"
FTL_OUTPUT_PATH = ROOT_DIR / "stage4_ftl_by_type.csv"
FAIRNESS_OUTPUT_PATH = ROOT_DIR / "stage4_fairness_by_type.csv"
REQUEST_OUTPUT_PATH = ROOT_DIR / "stage4_request_results.csv.gz"
LATENCY_FIGURE_PATH = ROOT_DIR / "stage4_policy_latency_comparison.png"
FTL_FIGURE_PATH = ROOT_DIR / "stage4_ftl_by_type.png"
WAIT_FIGURE_PATH = ROOT_DIR / "stage4_wait_by_type.png"
FAIRNESS_FIGURE_PATH = ROOT_DIR / "stage4_max_wait_by_type.png"

LIGHT_MULTIMODAL_MAX_IMAGES = 1
IMAGE_TOKEN_EQUIVALENT = 1000
GPU_THROUGHPUT = 2500

# FAAMS-BW parameters: bounded-wait guardrails for overloaded systems.
TEXT_BURST_LIMIT = 8
LIGHT_BURST_LIMIT = 4
LIGHT_URGENT_WAIT_S = 50_000.0
HEAVY_URGENT_WAIT_S = 150_000.0
STARVATION_SLOWDOWN_THRESHOLD = 10.0
EPSILON = 1e-9

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)
plt.switch_backend("Agg")
sns.set_theme(style="whitegrid")


# ---------------------- 1. Data Loading ----------------------
def load_trace(input_path: Path) -> pd.DataFrame:
    """Load the enhanced trace and normalize the columns used in scheduling."""
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
    """Assign request type and initial scheduler class from NumImages."""
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
    """Attach per-request scheduling metrics for one policy."""
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
    """Export one row per request with all policy metrics side by side."""
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
    """Single-server FCFS trace replay."""
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
    """Single-server non-preemptive static modality-aware priority scheduling."""
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


# ---------------------- 4. FAAMS-BW ----------------------
def simulate_faams_bw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fairness-Aware Aging Modality Scheduler with Bounded Waiting.

    Requests keep the original modality ordering, but the scheduler enforces:
    - a maximum burst of high-priority text dispatches while lower classes wait
    - a maximum burst of light multimodal dispatches while heavy requests wait
    - urgent wait caps that force service for light/heavy requests after long delay

    This design is more stable than pure threshold-based promotion under
    overloaded traces, where many waiting jobs can otherwise be promoted at once
    and the policy collapses back toward FCFS.
    """
    arrivals = df["arrival_time"].to_numpy(dtype=float)
    service = df["service_time"].to_numpy(dtype=float)
    n = len(df)

    starts = np.empty(n, dtype=float)
    wait = np.empty(n, dtype=float)
    completion = np.empty(n, dtype=float)

    queues = {1: deque(), 2: deque(), 3: deque()}

    next_arrival_idx = 0
    completed = 0
    current_time = 0.0
    text_streak = 0
    light_streak = 0

    def enqueue_request(req_idx: int) -> None:
        klass = int(df.at[req_idx, "priority"])
        queues[klass].append(req_idx)

    def process_arrivals_up_to(now: float) -> None:
        nonlocal next_arrival_idx
        while next_arrival_idx < n and arrivals[next_arrival_idx] <= now:
            enqueue_request(next_arrival_idx)
            next_arrival_idx += 1

    def oldest_wait(klass: int) -> float:
        if not queues[klass]:
            return -1.0
        return current_time - arrivals[queues[klass][0]]

    def pick_next_class() -> int | None:
        nonlocal text_streak, light_streak

        if queues[3] and oldest_wait(3) >= HEAVY_URGENT_WAIT_S:
            return 3
        if queues[2] and oldest_wait(2) >= LIGHT_URGENT_WAIT_S and text_streak >= TEXT_BURST_LIMIT:
            return 2

        text_blocked = text_streak >= TEXT_BURST_LIMIT and (queues[2] or queues[3])
        light_blocked = light_streak >= LIGHT_BURST_LIMIT and queues[3]

        if queues[1] and not text_blocked:
            return 1
        if queues[2] and not light_blocked:
            return 2
        if queues[3]:
            return 3
        if queues[2]:
            return 2
        if queues[1]:
            return 1
        return None

    while completed < n:
        if not any(queues.values()) and next_arrival_idx < n:
            current_time = max(current_time, arrivals[next_arrival_idx])

        process_arrivals_up_to(current_time)

        selected_class = pick_next_class()
        if selected_class is None:
            if next_arrival_idx < n:
                current_time = arrivals[next_arrival_idx]
                continue
            break

        selected_idx = queues[selected_class].popleft()

        if selected_class == 1:
            text_streak += 1
            light_streak = 0
        elif selected_class == 2:
            text_streak = 0
            light_streak += 1
        else:
            text_streak = 0
            light_streak = 0

        starts[selected_idx] = current_time
        wait[selected_idx] = starts[selected_idx] - arrivals[selected_idx]
        completion[selected_idx] = starts[selected_idx] + service[selected_idx]
        current_time = completion[selected_idx]
        completed += 1

    return build_policy_frame(df, "FAAMS-BW", starts, completion, wait)


# ---------------------- 5. Evaluation Tables ----------------------
def summarize_policies(results: pd.DataFrame) -> pd.DataFrame:
    """Compute overall policy-level metrics."""
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
    """Compute FTL and queueing delay by policy and request type."""
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
    policy_order = ["FCFS", "Static Modality Priority", "FAAMS-BW"]
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
    """Report fairness-oriented metrics by request type."""
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
    policy_order = ["FCFS", "Static Modality Priority", "FAAMS-BW"]
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
    """Plot overall latency comparison between policies."""
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
    ax.set_title("Stage 4 Policy Comparison", pad=16)
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
    """Plot mean first-token latency by policy and request type."""
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
    """Plot mean queueing delay by policy and request type."""
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
    """Plot max wait time to visualize starvation risk by class."""
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
    print("=== Stage 4: FAAMS-BW Scheduling Evaluation ===")
    print(f"Loading enhanced trace from: {INPUT_PATH}")

    trace_df = load_trace(INPUT_PATH)
    trace_df = classify_request_types(trace_df)

    print(f"Loaded {len(trace_df):,} requests")
    print("\nRequest-type distribution:")
    print(trace_df["request_type"].value_counts().rename_axis("request_type").to_frame("requests"))
    print("\nFAAMS-BW thresholds:")
    print(f"  Max consecutive text dispatches:  {TEXT_BURST_LIMIT}")
    print(f"  Max consecutive light dispatches: {LIGHT_BURST_LIMIT}")
    print(f"  Light urgent wait cap:            {LIGHT_URGENT_WAIT_S:.0f} s")
    print(f"  Heavy urgent wait cap:            {HEAVY_URGENT_WAIT_S:.0f} s")

    fcfs_results = simulate_fcfs(trace_df)
    static_priority_results = simulate_modality_priority(trace_df)
    faams_results = simulate_faams_bw(trace_df)
    all_results = pd.concat(
        [fcfs_results, static_priority_results, faams_results],
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

    print("\n=== Mean FTL by Policy and Request Type ===")
    print(ftl_table.to_string(index=False))

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
