# Top-Tier Upgrade Roadmap for Multimodal LMM Scheduling Thesis

## 1. 项目现状评估（Current Status）

### 1.1 项目定位

该仓库当前更接近**研究原型（research prototype）+ 离线分析管线**，而不是可投稿级系统论文原型，也不是可复用框架。

- `stage1.py`：原始 Azure 多模态推理 trace 清洗、特征构造、分布可视化。
- `stage2step1.py`：基于手工常数的延迟/能耗/碳排估计。
- `stage2step2.py`：对估计值与输入特征做相关性分析。
- `stage3.py`：基于增强后的 trace 做单服务器、非抢占、三优先级调度仿真。

从结构上看，代码是**线性串行脚本集合**，没有明确的模块边界、配置层、实验层、评估层和系统实现层。研究资产主要体现在离线 CSV、图表和 summary 表，而不是方法学或系统实现本身。

### 1.2 技术亮点

- 使用了**接近百万级请求规模**的真实 trace，数据规模足以支撑 workload characterization。
- 研究问题聚焦明确：**多模态异构性是否导致 SME 级 GPU 集群出现延迟与碳排失衡**。
- 提出了一个**轻量级 modality-aware priority scheduler** 的初步思路，论点直接、工程实现简单。
- 已有结果能支持一个基础 thesis narrative：
  workload heterogeneity -> queueing inefficiency -> lightweight scheduling helps short jobs.

### 1.3 主要问题（科研视角）

#### A. 方法创新性不足

- 当前调度策略本质上是**基于 `NumImages` 的静态优先级队列**，属于经典 priority scheduling 的直接实例化，不构成顶刊层面的 novelty。
- Stage 2 的“核心分析”主要建立在**人工设定的线性估计式**上：
  `latency_est`, `energy_est`, `carbon_est` 都是 `total_token_equiv` 的线性变换，因此 `NumImages` 与这些指标的超高 Pearson 相关性在很大程度上是**建模定义导致的 tautology**，不是独立实证发现。

#### B. 技术深度不足

- 没有真实在线 serving system、没有多 GPU/多 server/异构 GPU 拓扑、没有 admission control、没有 batching、没有 prefill/decode 分离。
- `stage3.py` 采用的是**单服务器重放仿真**；以当前数据计算，单服务器利用率约为 `7.71`，系统处于严重过载区，导致平均等待时间在百万秒量级。这说明当前结果更像是**极端饱和假设下的队列放大效应**，而不是现实部署结论。
- FTL 定义是 `wait_time + prefill_component`，但没有真实 first-token instrumentation，也没有对 prefill/decode overlap、KV cache、batch scheduling 的建模。

#### C. 实验严谨性不足

- baseline 只有 FCFS，一个基线远不足以支撑顶刊比较。
- 没有统计显著性分析、置信区间、bootstrap、敏感性分析。
- 没有 cross-trace/generalization 测试，没有 workload shift、peak/off-peak、不同硬件规模实验。
- 没有公平性指标，尽管结果已显示重多模态请求被明显牺牲。

#### D. 可复现性和工程化不足

- `stage1.py`、`stage2step1.py`、`stage2step2.py` 仍含机器相关绝对路径。
- 无 `requirements.txt` / `pyproject.toml` / lockfile / Dockerfile / CI。
- 无自动测试、无参数配置文件、无实验脚本、无随机种子管理、无数据版本管理。
- 仓库只保存中间产物，不保存原始数据获取脚本、校验和、数据 schema、版本元数据。

#### E. 论文表达层面不足

- README 已能陈述 thesis story，但仍偏“结果罗列”，缺少 threat model、假设边界、研究问题分解、对比方法动机、统计方法说明。
- 顶刊要求的是**可辩护的研究设计**，不是“能跑通的分析脚本”。

---

## 2. 与顶刊标准差距（Gap Analysis）

### 2.1 Novelty（创新性）

**现状**

- 方法为规则型三分类优先级调度。
- 分类依据仅为 `NumImages`，没有预测模型、在线学习、反馈控制或理论新结论。

**差距**

- IEEE/ACM/NeurIPS/ICSE 级投稿通常需要至少一项成立：
  1. 新算法；
  2. 新系统机制；
  3. 新 benchmark；
  4. 新理论分析；
  5. 大规模真实系统实证且发现不可替代。

**结论**

- 目前 novelty 仅停留在“将 modality awareness 引入调度规则”，不足以单独支撑顶刊。

### 2.2 Technical Depth（技术深度）

**现状**

- 核心 pipeline 是 pandas 脚本。
- Stage 2 使用固定常数估算延迟/能耗/碳排。
- Stage 3 是单机、非抢占、无 batching 的离散事件模拟。

**差距**

- 顶刊系统论文通常要求：
  - 完整 system design；
  - 关键模块实现；
  - 与生产机制接近的执行模型；
  - 复杂 trade-off 处理（latency/fairness/throughput/energy）。

**结论**

- 目前技术深度更接近硕士 thesis 初稿，而不是成熟研究系统。

### 2.3 Experimental Rigor（实验严谨性）

**现状**

- 主要结果为均值、P99、分组均值。
- 对比方法仅 FCFS。
- 相关性分析未区分“定义驱动相关”和“经验观测相关”。

**差距**

- 顶刊要求：
  - 强 baseline；
  - 多场景评估；
  - 统计显著性；
  - 消融实验；
  - 失败案例与公平性分析；
  - sensitivity/generalization/robustness。

**结论**

- 实验设计当前不足以支撑强 claim。

### 2.4 Reproducibility（可复现性）

**现状**

- 运行顺序可理解，但环境不可锁定，路径不可移植，数据来源不可自动恢复。

**差距**

- 顶刊 increasingly 要求 artifact review-ready：
  - 一键环境构建；
  - 固定依赖；
  - 明确数据版本；
  - 自动复现实验；
  - 输出可核验。

**结论**

- 当前复现门槛高，artifact 质量不达标。

### 2.5 Writing & Documentation

**现状**

- README 能覆盖主要阶段与结果。
- 缺少 problem formulation、hypothesis、formal objective、notation、threats to validity。

**差距**

- 顶刊写作需要：
  - 明确 research questions；
  - formal method section；
  - 实验协议；
  - 限制讨论；
  - artifact appendix。

**结论**

- 当前文档可以支撑 thesis repo，但不足以直接转化为 paper-ready package。

---

## 3. 升级路线（Upgrade Roadmap）

### 3.1 短期（1–2周）

| 改什么 | 为什么（对应顶刊要求） | 如何实现（具体技术方案） |
| --- | --- | --- |
| 重构为包化仓库 | 需要可维护、可复现实验框架，而不是一次性脚本 | 建立 `src/`、`configs/`、`experiments/`、`tests/`、`artifacts/` 结构；拆分为 `data`, `models`, `sim`, `metrics`, `plots` 模块 |
| 去除绝对路径并参数化 | 复现性是 artifact 最低门槛 | 用 `pathlib` + `argparse`/`typer`；配置写入 `yaml`；所有输入输出由 CLI 指定 |
| 固定环境与依赖 | 顶刊 artifact 需要 deterministic setup | 添加 `pyproject.toml` 或 `requirements.txt` + lockfile；记录 Python 版本；加入 `Makefile` |
| 补齐实验脚本 | 需要一键复现实验 | 增加 `make stage1`, `make stage2`, `make stage3`, `make reproduce-all` |
| 建立单元测试与回归测试 | 证明模拟器与指标计算正确 | 为队列逻辑、FTL 计算、分类边界、数据 schema 编写 `pytest`；加入小型 toy trace 金标准 |
| 重写 Stage 2 claim | 当前相关性分析有 tautology 风险 | 区分“model-defined metric”与“trace-observed variable”；新增 partial correlation、nonparametric ranking、feature importance on held-out prediction |
| 增加强 baseline | 单一 FCFS 无法发表 | 至少加入 SJF/STF、SRPT 近似、LAS、WFQ/DRR、aging priority、size-based with misprediction |

### 3.2 中期（1–2月）

| 改什么 | 为什么（对应顶刊要求） | 如何实现（具体技术方案） |
| --- | --- | --- |
| 构建更真实的 serving simulator | 当前单服务器模型过于简化 | 建立多服务器/多 GPU 离散事件模拟器，支持 heterogeneous GPUs、batching、prefill/decode 分离、queue admission |
| 将静态规则升级为学习型或预测型调度 | 增强 novelty 和方法深度 | 用轻量回归模型预测 job size / slowdown / carbon cost；调度目标转为 minimizing weighted slowdown 或 tail latency |
| 引入公平性机制 | 当前结果明显牺牲 heavy multimodal | 实现 aging、bounded slowdown、max-wait guarantee、class quota、deficit accounting |
| 用真实测量校准 Stage 2 模型 | 顶刊不接受纯拍脑袋能耗模型 | 在 A10/L4/A100 等设备上做 microbenchmark，拟合 `latency = f(tokens, images, batch, resolution)` 和 `energy = g(load, duration)` |
| 扩展数据集 | 单 trace 泛化不足 | 引入多时间段、多工作负载、合成 trace、不同模型族（LLaVA/Qwen-VL/Phi-3.5-Vision）数据 |
| 做系统性消融 | 证明每个模块有必要 | 消融特征、分类粒度、预测误差、batch size、GPU 数量、公平性参数、负载强度 |

### 3.3 长期（3–6月）

| 改什么 | 为什么（对应顶刊要求） | 如何实现（具体技术方案） |
| --- | --- | --- |
| 实现原型 serving system | 从“离线分析”升级到“系统论文” | 在 vLLM / SGLang / TGI / custom simulator 上集成 scheduler hook，支持在线请求重放 |
| 提出正式算法与理论分析 | 顶刊通常需要方法可解释且可证明 | 定义优化目标（e.g., weighted response time under fairness constraint）；给出近似保证、competitive analysis 或 stability condition |
| 建立公开 benchmark | benchmark 论文本身也可投稿 | 发布 multimodal scheduling benchmark：trace、schema、replayer、metrics、leaderboard |
| 联合优化 latency-energy-carbon-fairness | 单目标优化不足 | 做多目标优化或 constrained optimization，输出 Pareto frontier |
| 引入 agentic autotuning | 增强研究方向前沿性 | 使用 Codex/agent 自动搜索调度超参数、在线切换策略、生成实验配置并验证异常 case |

---

## 4. 核心研究方向建议（Research Directions）

### 方向 A：预测驱动的多模态作业大小估计与调度

**核心问题**

能否基于请求特征在线预测 prefill/decode 成本，再据此做 size-aware scheduling，而不是仅靠 `NumImages` 粗粒度分类。

**可能贡献**

- 设计轻量预测器，输入 `ContextTokens`, `GeneratedTokens` proxy, `NumImages`, image metadata, model type, queue state。
- 证明在预测误差存在时仍优于 FCFS 和静态 priority。
- 兼顾 tail latency 与 fairness。

### 方向 B：面向 SME GPU 集群的碳感知多目标调度

**核心问题**

能否同时优化 latency、energy、carbon 和公平性，而不是只优化平均延迟。

**可能贡献**

- 将 carbon-aware scheduling 与 request heterogeneity 联合建模。
- 在不同电网碳强度、GPU 类型和负载强度下给出 Pareto 分析。
- 适合 IEEE TPDS / ACM EuroSys / SoCC 类系统研究叙事。

### 方向 C：多模态推理服务 benchmark 与 evaluation framework

**核心问题**

当前领域缺少标准化的 multimodal serving trace benchmark、统一评价协议与可重放工具链。

**可能贡献**

- 发布 trace schema、重放器、系统负载配置、指标协议。
- 统一 baseline：FCFS、SJF、LAS、SRPT-like、fair schedulers、carbon-aware schedulers。
- 这一路线对 ICSE artifact/baseline infrastructure、benchmark track 或 systems venues 都有价值。

### 方向 D：AI/Agent 增强的调度策略自动发现

**核心问题**

能否使用 LLM/Codex/agent 自动生成、验证、筛选调度策略与超参数，而不是人工枚举 heuristic。

**可能贡献**

- agent 根据 workload 特征自动构造策略空间。
- 自动运行仿真、过滤违反 fairness constraint 的策略、输出 Pareto-optimal 候选。
- 适合与 AutoML / agentic systems / software engineering 方向结合。

---

## 5. 实验设计建议（Experiment Design）

### 5.1 Baseline

至少应包含以下基线，而不是只保留 FCFS：

- `FCFS`
- `SJF` / `Shortest-Estimated-Job-First`
- `SRPT-approx` / `Shortest-Remaining-Processing-Time` 近似
- `LAS` / Least Attained Service
- `WFQ` 或 `DRR` 类公平调度
- `Static Priority`（当前方法，作为弱基线）
- `Aging Priority`（缓解饥饿）
- `Learned Scheduler` 或 `Predict-then-Schedule`（若投稿方法为学习型）

### 5.2 Dataset

- 真实 trace：当前 Azure multimodal trace。
- 多切片评估：按日期、时段、峰值/非峰值、负载等级分层。
- 合成 trace：控制 `NumImages`、token 长度、burstiness、heavy-tail 程度。
- 硬件 trace：不同 GPU（A10/L4/A100/H100）上的 microbenchmark 数据。
- 若可能，引入第二个真实来源以避免单数据集依赖。

### 5.3 Evaluation Metrics

- Latency：mean, median, P90, P95, P99 response time
- FTL/TTFT：mean, P95, P99
- Throughput：requests/s, tokens/s, images/s
- Fairness：max slowdown, Jain’s fairness index, bounded slowdown violation rate, starvation rate
- Efficiency：GPU utilization, queue occupancy, batch efficiency
- Sustainability：energy/request, carbon/request, total energy, total carbon
- Robustness：performance variance under workload shift

### 5.4 Ablation Study

必须做以下消融：

- 不同分类粒度：`2-class`, `3-class`, `fine-grained bins`
- 不同预测器：线性回归、GBDT、轻量神经网络
- 有/无公平性约束
- 有/无 batching
- 单 GPU vs 多 GPU vs 异构 GPU
- 不同负载强度：`rho < 1`, `rho ~= 1`, `rho > 1`
- 不同 image-token equivalence 假设
- 不同电网碳强度

### 5.5 对比方法与统计协议

- 每组实验至少重复多个 seed 或多个 trace slice。
- 使用 bootstrap 置信区间报告核心指标。
- 对关键差异做显著性检验或非参数检验。
- 单独展示 failure cases：heavy multimodal starvation、tail collapse、fairness regression。

---

## 6. 可复现性与工程化改造

### 6.1 CI/CD

- 使用 GitHub Actions 执行 `lint + unit tests + smoke experiment`。
- PR 必须通过最小 toy trace 的 end-to-end 回归。
- 自动构建 Docker 镜像并发布到 registry。

### 6.2 自动测试

- `tests/test_metrics.py`：延迟/能耗/碳排公式校验。
- `tests/test_scheduler_fcfs.py`：FCFS 正确性。
- `tests/test_scheduler_priority.py`：优先级顺序、公平性机制边界条件。
- `tests/test_reproducibility.py`：固定输入得到固定输出 hash。
- `tests/test_schema.py`：输入 trace schema 验证。

### 6.3 数据版本管理

- 使用 `DVC` 或 `git-lfs + manifest` 管理大文件。
- 为每个数据文件记录：
  - 来源
  - 下载时间
  - checksum
  - 预处理版本
  - schema version
- 将原始数据、清洗数据、增强数据、实验输出分层存储。

### 6.4 Docker / 环境配置

- 提供 `Dockerfile` 和 `docker-compose.yml` 或 `uv`/`conda` 环境文件。
- 提供 `make docker-reproduce` 以一键跑全流程。
- 固定 Python 与核心包版本，避免 pandas / matplotlib 行为漂移。

### 6.5 实验配置管理

- 所有实验参数进入 `configs/*.yaml`。
- 建议使用 `hydra` 或简化版 YAML loader。
- 每次实验输出自动保存 config snapshot、git commit hash、时间戳和运行机器信息。

### 6.6 Artifact Packaging

- 提供 `artifact/README.md`：
  - hardware assumptions
  - expected runtime
  - expected outputs
  - checksum
  - common failure modes

---

## 7. 论文结构建议（Paper Outline）

### Abstract

- 问题：multimodal LMM serving 在 SME GPU 集群中的延迟/碳排问题。
- 方法：提出预测/公平/碳感知调度框架。
- 结果：在多个真实和合成 workload 上相对强 baseline 显著降低 tail latency/FTL/energy，同时控制 fairness。

### Introduction

- 背景：多模态服务负载异构性更强，SME 资源更受限。
- 痛点：现有 serving scheduler 假设 job size 同质或只针对文本。
- 研究问题：如何在资源受限场景下联合优化 latency、fairness 与 sustainability。
- 贡献列表：算法、系统、benchmark、实证。

### Related Work

- LLM serving / continuous batching / KV cache scheduling
- Size-based scheduling / queueing theory / fairness scheduling
- Carbon-aware computing / green AI
- Multimodal inference systems and benchmarking

### Method

- 问题定义与目标函数
- 请求成本建模或预测模型
- 调度算法设计
- 公平性/碳约束机制
- 复杂度分析与理论性质

### Experiments

- 实验设置：硬件、trace、baselines、metrics
- 主结果：整体性能
- 分层结果：不同请求类型、不同负载、不同 GPU
- 消融：每个模块必要性
- 鲁棒性：预测误差、工作负载迁移
- 失败案例与局限性

### Conclusion

- 总结发现
- 明确限制
- 对实际部署与未来工作给出边界清晰的结论

---

## 8. TODO 列表（可直接执行）

### P0：必须先做

- [ ] 新建 `src/` 包结构，拆分 `data`, `metrics`, `scheduler`, `simulator`, `plots` 模块
- [ ] 将所有绝对路径替换为相对路径 + CLI 参数
- [ ] 添加 `pyproject.toml` 或 `requirements.txt` + lockfile
- [ ] 增加 `configs/default.yaml`
- [ ] 写 `Makefile` 和统一复现实验入口
- [ ] 添加 `pytest` 与最小 toy trace 回归测试
- [ ] 在 README 中明确数据来源、假设、局限与复现步骤

### P1：尽快补齐的科研工作

- [ ] 加入 `SJF`, `LAS`, `Aging Priority`, `WFQ/DRR` baseline
- [ ] 报告 `P50/P95/P99`, slowdown, fairness, starvation 指标
- [ ] 做不同负载强度实验，避免只在 `rho >> 1` 下得出结论
- [ ] 用 bootstrap 给核心结果加 95% CI
- [ ] 将 Stage 2 从线性推导相关性改为校准模型 + 预测评估
- [ ] 增加 sensitivity analysis：image-token equivalence、GPU throughput、carbon intensity

### P2：中期研究升级

- [ ] 构建多 GPU/异构 GPU 离散事件模拟器
- [ ] 引入 batching 与 prefill/decode 分离模型
- [ ] 设计成本预测器并接入调度器
- [ ] 实现 bounded slowdown 或 aging 机制，控制重作业饥饿
- [ ] 增加第二数据源或合成 trace 生成器

### P3：面向投稿的高阶工作

- [ ] 在真实 serving stack 上实现 prototype
- [ ] 做 microbenchmark 校准 latency/energy 模型
- [ ] 发布 benchmark + artifact package
- [ ] 完成 paper-ready 图表与 appendix
- [ ] 准备开源 artifact review 文档

---

## 总结判断

该仓库已经具备一个**合理但初级的 thesis 研究骨架**：真实 trace、明确问题、初步调度策略、可解释的阶段性结果。真正的短板不是“代码不够漂亮”，而是**方法层级、实验设计、系统真实性和可复现性都还停留在原型阶段**。

如果目标是达到 IEEE/ACM/NeurIPS/ICSE 级投稿标准，最关键的升级不是继续堆图表，而是完成三件事：

1. 将静态 heuristic 升级为**有实质方法贡献**的调度框架。
2. 将单脚本离线分析升级为**可信的实验平台或系统原型**。
3. 将结果展示升级为**可审稿、可复现、可辩护的研究设计**。

没有这三步，当前项目更适合硕士 thesis；完成这三步之后，才有进入顶刊评审语境的可能。
