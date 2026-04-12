# ---------------------- Stage2 Step 2: Correlation Analysis (Standalone Script) ----------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Global plot settings
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams["font.size"] = 12
plt.switch_backend('Agg')

print("✅ Libraries loaded successfully!")

# ---------------------- Path Configuration (MUST REPLACE) ----------------------
ENHANCED_DATA_PATH = "/Users/jiangyutang/Desktop/TFM/azure_lmm_trace_enhanced.csv.gz"
OUTPUT_PATH = "/Users/jiangyutang/Desktop/TFM/"

# ---------------------- Load Enhanced Dataset ----------------------
df = pd.read_csv(ENHANCED_DATA_PATH)
print(f"✅ Loaded {len(df):,} enhanced requests")

# ---------------------- 1. Pearson Correlation Calculation ----------------------
# Select core features for correlation analysis
core_features = [
    "NumImages", 
    "ContextTokens", 
    "GeneratedTokens", 
    "latency_est", 
    "carbon_est",
    "gpu_compute_est"
]

# Calculate Pearson correlation matrix
corr_matrix = df[core_features].corr()
print("\n=== Pearson Correlation Matrix (PAPER CORE TABLE) ===")
print(corr_matrix.round(4))

# Save correlation matrix to CSV for paper
corr_matrix.round(4).to_csv(f"{OUTPUT_PATH}stage2_correlation_matrix.csv")
print(f"\n✅ Correlation matrix saved to {OUTPUT_PATH}")

# ---------------------- 2. Correlation Heatmap Visualization ----------------------
plt.figure(figsize=(10, 8))
# Plot heatmap with annotations, 4 decimal places
ax = sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".4f", 
    cmap="coolwarm", 
    vmin=-1, 
    vmax=1,
    linewidths=0.5,
    annot_kws={"fontsize": 10}
)
plt.title("Correlation Matrix of Workload Features and Performance Metrics", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}stage2_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Correlation heatmap saved!")

# ---------------------- 3. Key Driver Ranking ----------------------
# Rank features by correlation with carbon footprint (core target)
carbon_correlation = corr_matrix["carbon_est"].drop(["carbon_est", "latency_est", "gpu_compute_est"]).sort_values(ascending=False)
print("\n=== Feature Ranking by Correlation with Carbon Footprint ===")
for feature, corr_value in carbon_correlation.items():
    print(f"{feature}: {corr_value:.4f}")

print("\n🎉 Stage2 Step 2 complete! Root cause analysis core deliverables are ready.")