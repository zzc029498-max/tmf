# TFM: Lightweight Low-Latency Green LMM Inference Optimization for SME Clusters
**Master's Final Thesis (TFM)** | Universitat Rovira i Virgili (URV)
**Author**: Yutang Jiang
**Supervisor**: Prof. Pedro
**Current Progress**: ✅ Stage 1 & 2 Completed | ⏳ Stage 3 In Preparation

---

## 1. Project Overview
This repository contains all the code, data, and analysis results for my TFM thesis. The research focuses on addressing the core pain points of multimodal large language model (LMM) inference in resource-constrained small and medium-sized enterprise (SME) clusters.

The core goal is to design a lightweight, zero-intrusion modality-aware scheduling framework to reduce end-to-end inference latency, tail latency, and carbon footprint of LMM inference, without modifying the model kernel or underlying hardware.

---

## 2. Completed Work & Progress
### ✅ Stage 1: Dataset Preprocessing & Workload Characterization
- Completed rigorous cleaning of the public Azure LMM Inference Trace dataset, retaining **979,123 valid requests with a 97.91% data retention rate**;
- Completed full workload characterization, including modal proportion, context token length distribution, image count distribution, and hourly traffic trend analysis.

### ✅ Stage 2: Root Cause Analysis of Latency & Carbon Footprint
- Established a citable end-to-end quantitative model to estimate per-request inference latency, GPU energy consumption, and carbon footprint, based on official NVIDIA A10 hardware specifications and 2026 Spanish grid carbon intensity data;
- Quantified the core heterogeneous gap: multimodal requests have a **14.8x higher average latency and carbon footprint** than text-only requests, contributing 93.7% of total emissions despite accounting for only 50% of total traffic;
- Identified the dominant bottleneck: the number of images in a request has a **near-perfect 0.9992 positive correlation** with latency and carbon emissions;
- Defined the key limitations of state-of-the-art inference frameworks (e.g., vLLM) in SME scenarios, and derived clear optimization directions for the scheduling framework.

### ⏳ Upcoming Stage 3: Scheduling Framework Design & Evaluation
- Design the lightweight modality-aware priority scheduling scheme;
- Validate the optimization effect via trace-driven simulation or analytical formula-based quantification;
- Complete the thesis writing and final revision.

---

## 3. Repository File Description
### Core Analysis Code
| File Name | Description |
|-----------|-------------|
| `stage1.py` | Stage 1 code: dataset cleaning, preprocessing, and preliminary workload characterization |
| `stage2step1.py` | Stage 2 code: end-to-end latency, energy consumption, and carbon footprint quantification |
| `stage2step2.py` | Stage 2 code: Pearson correlation analysis and root cause identification |

### Dataset Files
| File Name | Description |
|-----------|-------------|
| `azure_lmm_trace_clean.csv.gz` | Cleaned raw Azure LMM Inference Trace dataset |
| `azure_lmm_trace_enhanced.csv.gz` | Enhanced dataset with calculated latency, energy, and carbon footprint per request |

### Visualization Figures
| File Name | Description |
|-----------|-------------|
| `01_modal_proportion.png` | Distribution of text-only and multimodal requests |
| `02_context_tokens_dist.png` | Input context token length distribution across all requests |
| `03_num_images_dist.png` | Image count distribution for multimodal requests |
| `04_hourly_traffic.png` | Hourly request arrival rate trend over the trace period |
| `stage2_modal_carbon_boxplot.png` | Carbon footprint comparison between text-only and multimodal requests |
| `stage2_correlation_heatmap.png` | Correlation heatmap of core workload features, latency, and carbon footprint |

### Statistical Result Tables
| File Name | Description |
|-----------|-------------|
| `stage2_core_stats.csv` | Overall core metrics statistics for the full dataset |
| `stage2_modal_comparison.csv` | Key metrics comparison between text-only and multimodal requests |
| `stage2_correlation_matrix.csv` | Full Pearson correlation matrix of core features |

---

## 4. Core Key Findings
1.  Multimodal requests are the dominant driver of latency and carbon emissions in SME LMM clusters, with a 14.8x performance gap compared to text-only requests;
2.  The number of images per request is the core root cause of performance degradation, with an almost linear positive correlation with latency and carbon emissions;
3.  The default FCFS scheduling policy in mainstream frameworks cannot adapt to this heterogeneous workload, causing severe head-of-line blocking in resource-constrained SME clusters.

---

## 5. Environment Requirements
All code is written in Python 3.12+, with the following core dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy