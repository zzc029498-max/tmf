
Master's Final Thesis (TFM) repository at Universitat Rovira i Virgili (URV)

- Author: Yutang Jiang
- Research focus: low-latency and low-carbon multimodal LMM inference for SME-scale GPU clusters
- Current repository status: Stage 1, Stage 2, and Stage 3 analytical evaluation completed

## Project Summary

This repository contains the analysis pipeline, intermediate datasets, figures, and evaluation outputs for a thesis on lightweight scheduling optimization for multimodal large language model (LMM) inference.

The core thesis argument is that heterogeneous multimodal workloads create severe queueing inefficiency in resource-constrained SME deployments. A lightweight scheduler that distinguishes request modality can substantially improve user-facing latency for short requests without modifying model kernels or hardware.

## Research Workflow

### Stage 1. Trace Cleaning and Workload Characterization

`stage1.py` preprocesses the Azure multimodal inference trace, removes invalid and extreme requests, derives request-level features, and generates the basic workload characterization figures.

Key outputs:

- `azure_lmm_trace_clean.csv.gz`
- `01_modal_proportion.png`
- `02_context_tokens_dist.png`
- `03_num_images_dist.png`
- `04_hourly_traffic.png`

Main observations:

- 979,123 valid requests remain after cleaning.
- Data retention rate is 97.91%.
- The workload is almost evenly split between text-only and multimodal traffic:
  489,817 text-only requests and 489,306 multimodal requests.
- Average context length is 2,464.09 tokens, with a median of 1,107 tokens.
- The trace spans about 7 days, with hourly arrivals ranging from 1,594 to 11,905 requests and a standard deviation of 2,333.96, confirming bursty traffic.

### Stage 2. Latency, Energy, and Carbon Root-Cause Analysis

`stage2step1.py` adds estimated latency, energy consumption, and carbon footprint to each cleaned request. `stage2step2.py` performs Pearson correlation analysis to identify the dominant drivers.

Key outputs:

- `azure_lmm_trace_enhanced.csv.gz`
- `stage2_core_stats.csv`
- `stage2_modal_comparison.csv`
- `stage2_correlation_matrix.csv`
- `stage2_modal_carbon_boxplot.png`
- `stage2_correlation_heatmap.png`

Main findings:

- Mean estimated latency across all requests is 4.7622 s.
- Mean estimated carbon footprint is 0.03730 g CO2 per request.
- Multimodal requests are the dominant source of system cost:
  average latency is 8.9270 s versus 0.6017 s for text-only requests.
- Multimodal requests emit 34.2162 kg CO2 in total, versus 2.3087 kg CO2 for text-only traffic over the trace.
- `NumImages` is the strongest predictor of latency and carbon footprint, with Pearson correlation 0.9992.
- `ContextTokens` is also important, but clearly weaker than image count, with correlation 0.8943.

### Stage 3. Trace-Driven Scheduling Evaluation

`stage3.py` evaluates a lightweight non-preemptive modality-aware priority scheduler against FCFS using the enhanced trace. Requests are classified into three classes:

- Text-only: `NumImages == 0`
- Light multimodal: `NumImages == 1`
- Heavy multimodal: `NumImages >= 2`

Key outputs:

- `stage3_policy_summary.csv`
- `stage3_ftl_by_type.csv`
- `stage3_request_results.csv.gz`

Main findings:

- Average total latency drops from 1,591,291.84 s under FCFS to 356,640.25 s under modality-aware priority.
- This corresponds to a 77.59% reduction in average total latency.
- Average first-token latency (FTL) also drops by 77.59%.
- P99 total latency improves only slightly, from 4,036,065.57 s to 3,976,676.57 s, showing that tail behavior remains constrained by heavy requests.
- Text-only average FTL improves by 99.9993%.
- Light multimodal average FTL improves by 95.4413%.
- Heavy multimodal requests become slower, with average FTL increasing by 15.8469%, which exposes a fairness tradeoff that must be discussed explicitly in the thesis.

## Repository Structure

### Core scripts

- `stage1.py`: preprocessing and workload characterization
- `stage2step1.py`: latency, energy, and carbon estimation
- `stage2step2.py`: correlation analysis and bottleneck identification
- `stage3.py`: FCFS versus modality-aware scheduling evaluation

### Data artifacts

- `azure_lmm_trace_clean.csv.gz`: cleaned trace after Stage 1
- `azure_lmm_trace_enhanced.csv.gz`: enhanced trace with Stage 2 metrics
- `stage3_request_results.csv.gz`: per-request scheduling results for both policies

### Tables and figures

- `stage2_core_stats.csv`
- `stage2_modal_comparison.csv`
- `stage2_correlation_matrix.csv`
- `stage3_policy_summary.csv`
- `stage3_ftl_by_type.csv`
- `01_modal_proportion.png`
- `02_context_tokens_dist.png`
- `03_num_images_dist.png`
- `04_hourly_traffic.png`
- `stage2_modal_carbon_boxplot.png`
- `stage2_correlation_heatmap.png`

## How To Reproduce

Environment:

- Python 3.12+
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scipy
```

Run order:

```bash
python3 stage1.py
python3 stage2step1.py
python3 stage2step2.py
python3 stage3.py
```

## Important Reproducibility Note

`stage1.py`, `stage2step1.py`, and `stage2step2.py` still contain machine-specific absolute paths. They reflect the environment used during the analysis and should be adjusted before rerunning on a different machine. `stage3.py` already uses repository-relative paths and is directly reproducible inside this repository.

## Thesis-Level Takeaways

- Multimodal heterogeneity is the main reason SME-scale LMM serving suffers from poor latency and elevated emissions.
- Image count is the dominant bottleneck variable and is a strong basis for lightweight scheduling decisions.
- A simple modality-aware scheduler can dramatically improve service for short requests.
- The current priority policy improves average performance but shifts delay to heavy multimodal traffic, so the final thesis should treat fairness and starvation control as a design consideration rather than claim universal benefit.

