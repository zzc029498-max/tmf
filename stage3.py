from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------- 0. Configuration ----------------------
ROOT_DIR = Path(__file__).resolve().parent
INPUT_PATH = ROOT_DIR / "azure_lmm_trace_enhanced.csv.gz"
SUMMARY_OUTPUT_PATH = ROOT_DIR / "stage3_policy_summary.csv"
FTL_OUTPUT_PATH = ROOT_DIR / "stage3_ftl_by_type.csv"
REQUEST_OUTPUT_PATH = ROOT_DIR / "stage3_request_results.csv.gz"

# Priority-class threshold.
# The prompt text omits the exact light/heavy boundary, so the default is:
# - Text-only: NumImages == 0
# - Light multimodal: NumImages == 1
# - Heavy multimodal: NumImages >= 2
LIGHT_MULTIMODAL_MAX_IMAGES = 1

# Stage 2 / thesis constants
IMAGE_TOKEN_EQUIVALENT = 1000
GPU_THROUGHPUT = 2500  # tokens per second on NVIDIA A10

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


# ---------------------- 1. Data Loading ----------------------
def load_trace(input_path: Path) -> pd.DataFrame:
    """Load the enhanced trace and normalize the columns used in Stage 3."""
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
        df["ContextTokens"].astype(float) +
        df["NumImages"].astype(float) * IMAGE_TOKEN_EQUIVALENT
    ) / GPU_THROUGHPUT

    return df.sort_values(["arrival_time", "arrival_timestamp"]).reset_index(drop=True)


def classify_request_types(df: pd.DataFrame) -> pd.DataFrame:
    """Assign request type and scheduler priority from NumImages."""
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


# ---------------------- 2. Discrete-Event Simulation ----------------------
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
    """
    Single-server non-preemptive priority scheduling.
    Within each priority class, requests keep FCFS order to prevent starvation.
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

    return build_policy_frame(df, "Modality-Aware Priority", starts, completion, wait)


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

    return result


# ---------------------- 3. Evaluation Tables ----------------------
def summarize_policies(results: pd.DataFrame) -> pd.DataFrame:
    """Compute overall policy-level metrics."""
    summary = (
        results.groupby("policy")
        .agg(
            avg_total_latency_s=("total_latency", "mean"),
            p99_total_latency_s=("total_latency", lambda x: np.percentile(x, 99)),
            total_carbon_kg=("carbon_kg", "sum"),
            avg_ftl_s=("ftl", "mean"),
        )
        .reset_index()
    )
    return summary.round(6)


def summarize_ftl_by_type(results: pd.DataFrame) -> pd.DataFrame:
    """Compute FTL by policy and request type."""
    ftl_table = (
        results.groupby(["policy", "request_type"])
        .agg(
            avg_ftl_s=("ftl", "mean"),
            avg_wait_time_s=("wait_time", "mean"),
            requests=("request_id", "count"),
        )
        .reset_index()
    )

    type_order = ["Text-only", "Light multimodal", "Heavy multimodal"]
    ftl_table["request_type"] = pd.Categorical(
        ftl_table["request_type"], categories=type_order, ordered=True
    )

    return ftl_table.sort_values(["policy", "request_type"]).round(6)


def build_wide_request_export(results: pd.DataFrame) -> pd.DataFrame:
    """Export one row per request with both policies side by side."""
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

    static = (
        results[base_columns]
        .drop_duplicates(subset=["request_id"])
        .sort_values("request_id")
        .reset_index(drop=True)
    )

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


# ---------------------- 4. Main Entry ----------------------
def main() -> None:
    print("=== Stage 3: Trace-Driven Scheduling Evaluation ===")
    print(f"Loading enhanced trace from: {INPUT_PATH}")

    trace_df = load_trace(INPUT_PATH)
    trace_df = classify_request_types(trace_df)

    print(f"Loaded {len(trace_df):,} requests")
    print("\nRequest-type distribution:")
    print(trace_df["request_type"].value_counts().rename_axis("request_type").to_frame("requests"))

    fcfs_results = simulate_fcfs(trace_df)
    priority_results = simulate_modality_priority(trace_df)
    all_results = pd.concat([fcfs_results, priority_results], ignore_index=True)

    summary_table = summarize_policies(all_results)
    ftl_table = summarize_ftl_by_type(all_results)
    request_export = build_wide_request_export(all_results)

    summary_table.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    ftl_table.to_csv(FTL_OUTPUT_PATH, index=False)
    request_export.to_csv(REQUEST_OUTPUT_PATH, index=False, compression="gzip")

    print("\n=== Policy Comparison Summary ===")
    print(summary_table.to_string(index=False))

    print("\n=== Average FTL by Policy and Request Type ===")
    print(ftl_table.to_string(index=False))

    print("\n=== Output Files ===")
    print(f"Summary table: {SUMMARY_OUTPUT_PATH.relative_to(ROOT_DIR)}")
    print(f"FTL by type:   {FTL_OUTPUT_PATH.relative_to(ROOT_DIR)}")
    print(f"Per-request:   {REQUEST_OUTPUT_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
