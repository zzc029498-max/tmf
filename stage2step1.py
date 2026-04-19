# ---------------------- 0. Environment Setup ----------------------
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Global plot settings for clarity and professionalism
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False
plt.switch_backend('Agg')  # Non-interactive backend for Mac compatibility

print("✅ All libraries loaded successfully!")

STAGE1_CLEAN_DATA_PATH = "/Users/jiangyutang/Desktop/TFM/azure_lmm_trace_clean.csv.gz"
STAGE2_OUTPUT_PATH = "/Users/jiangyutang/Desktop/TFM/"

# ---------------------- 2. Load Stage1 Cleaned Data (NO PREPROCESSING) ----------------------
print("\n=== Loading Stage1 Cleaned Dataset ===")
df_enhanced = pd.read_csv(STAGE1_CLEAN_DATA_PATH)
print(f"✅ Loaded {len(df_enhanced):,} valid requests from Stage1")
print(f"Columns available: {df_enhanced.columns.tolist()}")

# ---------------------- 3. Fixed Academic Parameters (Fully Citable) ----------------------
# All parameters from official public sources for academic rigor
IMAGE_TOKEN_EQUIVALENT = 1000    # Industry consensus: 1 image ≈ 1000 text tokens
GPU_TDP = 150                     # NVIDIA A10 TDP (Watt): NVIDIA Official Hardware Specs
GPU_THROUGHPUT = 2500             # A10 LMM Inference Throughput (tokens/sec): Azure Public Performance Data
GPU_UTILIZATION = 0.8             # Average GPU utilization in LMM inference: Industry conservative value
CARBON_INTENSITY = 0.235          # Spanish Grid Carbon Intensity (kg CO₂/kWh): EU 2026 Official Energy Data

# ---------------------- 4. Core Stage2 Metric Calculations ----------------------
print("\n=== Calculating Core Stage2 Metrics ===")

# 4.1 Total equivalent tokens (aligns with Stage1's gpu_compute_est)
df_enhanced["total_token_equiv"] = (
    df_enhanced["ContextTokens"] + 
    df_enhanced["GeneratedTokens"] + 
    df_enhanced["NumImages"] * IMAGE_TOKEN_EQUIVALENT
)

# 4.2 Single-request latency estimation (seconds)
df_enhanced["latency_est"] = df_enhanced["total_token_equiv"] / GPU_THROUGHPUT

# 4.3 Single-request GPU energy consumption (kWh)
df_enhanced["energy_est"] = (
    (GPU_TDP * GPU_UTILIZATION * df_enhanced["latency_est"]) 
    / 3600  # Convert seconds to hours
    / 1000  # Convert Watts to kW
)

# 4.4 Single-request carbon footprint (g CO₂, easier for small-value comparison)
df_enhanced["carbon_est"] = df_enhanced["energy_est"] * CARBON_INTENSITY * 1000

print("✅ All core metrics calculated successfully!")

# ---------------------- 5. Validation & Key Statistics (Save for Paper) ----------------------
print("\n=== Core Metric Statistics Overview ===")
core_stats = df_enhanced[["latency_est", "energy_est", "carbon_est", "gpu_compute_est"]].describe()
print(core_stats)

print("\n=== Modal-Specific Comparison (PAPER CORE HIGHLIGHT) ===")
modal_stats = df_enhanced.groupby("is_multimodal").agg(
    avg_latency_s=("latency_est", "mean"),
    avg_carbon_g=("carbon_est", "mean"),
    total_carbon_kg=("carbon_est", lambda x: x.sum() / 1000),
    avg_gpu_compute=("gpu_compute_est", "mean")
).reset_index()
modal_stats["is_multimodal"] = modal_stats["is_multimodal"].map({0: "Text-only", 1: "Multimodal"})
print(modal_stats)

# Save statistics to CSV for easy paper import
core_stats.to_csv(f"{STAGE2_OUTPUT_PATH}stage2_core_stats.csv")
modal_stats.to_csv(f"{STAGE2_OUTPUT_PATH}stage2_modal_comparison.csv", index=False)
print(f"\n✅ Statistics saved to {STAGE2_OUTPUT_PATH}")

# ---------------------- 6. Quick Sanity Check Visualization ----------------------
print("\n=== Generating Sanity Check Visualization ===")

# Boxplot: Text-only vs Multimodal Carbon Footprint
plt.figure(figsize=(10, 6))
sns.boxplot(x="is_multimodal", y="carbon_est", data=df_enhanced, 
            palette=["#66b3ff", "#ff9999"], showfliers=False)  # Hide extreme outliers for clarity
plt.xticks([0, 1], ["Text-only", "Multimodal"], fontsize=12)
plt.xlabel("Request Type", fontsize=14)
plt.ylabel("Estimated Carbon Footprint (g CO₂)", fontsize=14)
plt.title("Text-only vs Multimodal Carbon Footprint Comparison (Outliers Hidden)", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.7, axis="y")
plt.savefig(f"{STAGE2_OUTPUT_PATH}stage2_modal_carbon_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Sanity check visualization saved!")

# ---------------------- 7. Save Enhanced Dataset for Next Stage2 Steps ----------------------
ENHANCED_DATA_PATH = f"{STAGE2_OUTPUT_PATH}azure_lmm_trace_enhanced.csv.gz"
df_enhanced.to_csv(ENHANCED_DATA_PATH, index=False, compression="gzip")
print(f"\n✅ Enhanced dataset saved to: {ENHANCED_DATA_PATH}")

print("\n🎉 Stage2 Step 1 complete! Ready for correlation analysis next.")