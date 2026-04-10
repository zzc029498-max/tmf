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
plt.switch_backend('Agg')

print(f"✅ Python Version: {sys.version}")
print(f"✅ Pandas Version: {pd.__version__}")
print("✅ All libraries loaded successfully!")

# ---------------------- 1. Load Raw Dataset ----------------------
RAW_DATA_PATH = "/Users/jiangyutang/Downloads/AzureLMMInferenceTrace_multimodal.csv"
df = pd.read_csv(RAW_DATA_PATH)

print(f"\n=== Raw Dataset Overview ===")
print(f"Total raw requests: {len(df):,}")
print(f"All column names: {df.columns.tolist()}")
print("\nFirst 5 rows of raw data:")
print(df.head())

# ---------------------- 2. Data Preprocessing ----------------------
print("\n=== Starting Data Preprocessing ===")

# 2.1 Filter out empty and invalid requests
df_clean = df[
    (df["ContextTokens"] > 0) &      # Input tokens must be > 0
    (df["GeneratedTokens"] > 0) &    # Output tokens must be > 0
    (df["NumImages"] >= 0)           # Number of images cannot be negative
].copy()
print(f"Requests after filtering empty/invalid: {len(df_clean):,}")

# 2.2 Filter extreme outliers (99th percentile threshold, conservative)
print("\n=== 99th Percentile of Key Features (Outlier Threshold) ===")
quantile_99 = df_clean[["NumImages", "ContextTokens", "GeneratedTokens"]].quantile(0.99)
print(quantile_99)

# Apply threshold
df_clean = df_clean[
    (df_clean["NumImages"] <= quantile_99["NumImages"]) &
    (df_clean["ContextTokens"] <= quantile_99["ContextTokens"]) &
    (df_clean["GeneratedTokens"] <= quantile_99["GeneratedTokens"])
]
print(f"\nRequests after filtering extreme outliers: {len(df_clean):,}")
print(f"Data retention rate: {len(df_clean)/len(df)*100:.2f}%")

# 2.3 Time formatting + Derived feature creation
# Convert TIMESTAMP to datetime format for future time-series analysis
df_clean["request_time"] = pd.to_datetime(df_clean["TIMESTAMP"])

# New feature: Distinguish text-only vs multimodal requests (0=Text-only, 1=Multimodal)
df_clean["is_multimodal"] = df_clean["NumImages"].apply(lambda x: 1 if x >= 1 else 0)

# New feature: Simple GPU compute estimation (for future carbon footprint analysis, not exact)
# Assumption: 1 text token ≈ 1 compute unit, 1 image ≈ 1000 compute units
df_clean["gpu_compute_est"] = (
    df_clean["ContextTokens"] + 
    df_clean["GeneratedTokens"] + 
    df_clean["NumImages"] * 1000
)

print("\n✅ Data preprocessing complete!")
print("\n=== Key Statistics of Cleaned Data ===")
print(df_clean[["NumImages", "ContextTokens", "GeneratedTokens", "gpu_compute_est"]].describe())

print("\n=== Text-only vs Multimodal Request Proportion ===")
print(df_clean["is_multimodal"].value_counts(normalize=True))

# ---------------------- 3. Basic Statistical Analysis ----------------------
print("\n=== Basic Statistical Analysis ===")

# 3.1 Modal-specific statistics (Core!)
print("\n=== Modal-specific Statistics Comparison ===")
modal_stats = df_clean.groupby("is_multimodal").agg(
    avg_num_images=("NumImages", "mean"),
    avg_context_tokens=("ContextTokens", "mean"),
    avg_generated_tokens=("GeneratedTokens", "mean"),
    avg_gpu_compute=("gpu_compute_est", "mean")
).reset_index()
modal_stats["is_multimodal"] = modal_stats["is_multimodal"].map({0: "Text-only", 1: "Multimodal"})
print(modal_stats)

# 3.2 Time-series statistics (7-day request volume)
print("\n=== 7-Day Hourly Request Volume Statistics ===")
hourly_requests = df_clean.resample("H", on="request_time").size()
print(f"Average hourly requests: {hourly_requests.mean():.0f}")
print(f"Maximum hourly requests: {hourly_requests.max():.0f}")
print(f"Minimum hourly requests: {hourly_requests.min():.0f}")
print(f"Hourly request standard deviation: {hourly_requests.std():.0f} (Validates traffic burstiness)")

# ---------------------- 4. Initial Visualization (Core Deliverables!) ----------------------
print("\n=== Generating Visualization Charts ===")

# 【MUST REPLACE】Change to your own save path
SAVE_PATH = "/Users/jiangyutang/Desktop/TFM/"

# 4.1 Core Chart 1: Text-only vs Multimodal Request Proportion Pie Chart
plt.figure(figsize=(8, 8))
labels = ["Text-only", "Multimodal"]
sizes = df_clean["is_multimodal"].value_counts(normalize=True).values
colors = ["#66b3ff", "#ff9999"]
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 14})
plt.title("Proportion of Text-only vs Multimodal Requests", fontsize=16)
plt.axis("equal")
plt.savefig(f"{SAVE_PATH}01_modal_proportion.png", dpi=300, bbox_inches="tight")
plt.close()  # 🔧 优化：关闭绘图，释放内存
print("✅ Chart 1 saved: 01_modal_proportion.png")

# 4.2 Core Chart 2: Context Tokens Log-Log Histogram (Validates Power-Law Distribution)
plt.figure(figsize=(12, 6))

print(f"\n🔍 ContextTokens数据检查：")
print(f"minimal: {df_clean['ContextTokens'].min()}")
print(f"maximal: {df_clean['ContextTokens'].max()}")
print(f"nonzerorequest: {len(df_clean[df_clean['ContextTokens'] > 0]):,}")

ax = sns.histplot(df_clean["ContextTokens"], bins=50, color="#66b3ff", edgecolor="white")
ax.set_xscale("log")
ax.set_yscale("log")

plt.xlabel("Context Tokens (log scale)", fontsize=14)
plt.ylabel("Number of Requests (log scale)", fontsize=14)
plt.title("Context Tokens Distribution (Log-Log Scale)", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig(f"{SAVE_PATH}02_context_tokens_dist.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 2 saved: 02_context_tokens_dist.png")

# 4.3 Core Chart 3: Number of Images Distribution Bar Plot (Only ≤10 for clarity)
plt.figure(figsize=(12, 6))
# Only plot NumImages ≤10 (After 99th percentile filtering, most requests are ≤10)
sns.countplot(x="NumImages", data=df_clean[df_clean["NumImages"] <= 10], color="#ff9999", edgecolor="white")
plt.xlabel("Number of Images per Request", fontsize=14)
plt.ylabel("Number of Requests", fontsize=14)
plt.title("Number of Images Distribution (≤10)", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.7, axis="y")
plt.savefig(f"{SAVE_PATH}03_num_images_dist.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 3 saved: 03_num_images_dist.png")

# 4.4 Core Chart 4: 7-Day Hourly Request Volume Time-Series Plot (Validates Traffic Burstiness)
plt.figure(figsize=(14, 6))
hourly_requests.plot(color="#66b3ff", linewidth=2)
plt.xlabel("Time (Hourly)", fontsize=14)
plt.ylabel("Number of Requests per Hour", fontsize=14)
plt.title("7-Day Hourly LMM Inference Request Volume", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig(f"{SAVE_PATH}04_hourly_traffic.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 4 saved: 04_hourly_traffic.png")

# ---------------------- 5. Save Cleaned Dataset ----------------------
# 【MUST REPLACE】Change to your own save path
CLEAN_DATA_PATH = f"{SAVE_PATH}azure_lmm_trace_clean.csv.gz"
df_clean.to_csv(CLEAN_DATA_PATH, index=False, compression="gzip")
print(f"\n✅ Cleaned dataset saved to: {CLEAN_DATA_PATH}")

print("\n🎉 Stage1 complete! All 4 charts and cleaned data are saved.")