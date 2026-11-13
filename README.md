# GZNL-flu-forecast-models

广州国家实验室（GZNL Group）的中国流感周度阳性率预测模型，面向
[China Infectious Disease Forecast Hub](https://github.com/dailypartita/China-Infectious-Disease-Forecast-Hub)
的目标 **`wk inc flu prop ili`**。

方法改编自 EpiETL `modeling` 套件（`backtest.py`、`ensemble_conformal.py`、
`forecast_final.py`、`prepare_data.py`）。原套件预测的是全国**月度流感病例数**；
本仓库把它改写为对 CNIC 全国流感监测网络 ILI 标本**实验室阳性率**的周度预测，
单位是百分点（0–100，例如 13.5 表示 13.5%），地点仅为 `CN`。

本目录提交的模型是 **GZNL-FluLGBMCQR**（LightGBM 分位数回归 + 保形校准）。
`designated_model` 设为 `false`：GZNL 在该 hub 已有两个指定模型（NextWave、
VincentAvg），一队最多指定两个。

## 预测目标

| 字段 | 取值 |
|------|------|
| `target` | `wk inc flu prop ili` |
| `location` | `CN` |
| 单位 | 百分点，区间 [0, 100] |
| 提交步长 | 整数 `-1` 到 `6` 周 |
| 分位数 | hub 规定的 23 个水平（0.01 … 0.99） |
| `reference_date` | 监测周周一；`target_end_date = reference_date + horizon × 7 天` |

## 训练集

训练数据来自官方 hub 文件 `target-data/time-series.csv`，筛选
`target == "wk inc flu prop ili"` 且 `location == "CN"`。本仓库
`data/time-series.csv` 是生成 2026-11-09 提交时使用的快照。

该快照中，流感阳性率序列一共有 **884 个周一观测**，时间范围是
**2009-09-07 至 2026-10-26**。按日历跨度共 895 周，中间有 6 处缺口、合计缺 **11 周**
（最长一处是 2013-05-27 至 2013-07-08）；2015-01-19 之后的序列是连续的。
所有 884 个观测都是有限数值，单位已经是百分点。

这不是一次固定的“前 70% 训练、后 30% 测试”划分。每个预测起点都会重新拟合，
并且只使用当时已经发布的观测：按当前仓库的发布节奏，约为
`reference_date` 之前 2 周（例如 2026-11-09 这一轮，最后一次可用观测是 2026-10-26）。

LightGBM 需要 12 周滞后才能构成完整特征，因此设计矩阵从序列第 12 个点
（0-based 下标 11）开始；目标是 `y[t + k]`，其中 `k` 是从最后一次观测到目标周的周数，
且 `t + k` 不超过当时可见的最后一周。样本不足 80 行时，该起点回退到气候态模型。
在 2026-11-09 这次提交中，各步长的训练样本数为：

| horizon | 从最后观测算起的步数 k | 训练样本数 |
|---------|-------------------------|------------|
| -1 | 1 | 872 |
| 0 | 2 | 871 |
| 1 | 3 | 870 |
| 2 | 4 | 869 |
| 3 | 5 | 868 |
| 4 | 6 | 867 |
| 5 | 7 | 866 |
| 6 | 8 | 865 |

回测第一个起点 2024-01-01（当时可见到 2023-12-18）的训练样本约为 716–723 行，
随后随历史拉长逐渐增加到 800 行以上。

## 算法原理

提交模型 **FluLGBMCQR** 分两步：先做直接多步 LightGBM 分位数回归，再用滑动窗口
[CQR](https://arxiv.org/abs/1905.03222)（Romano, Patterson, Candès, 2019）校准区间。

**1. LightGBM 分位数回归。** 对每个预测步长、每个分位数水平单独训练一个模型
（直接多步，而不是迭代滚动一步预测）。损失函数是分位数（pinball）损失，
水平 `α` 取 hub 规定的 23 个分位数。树模型超参数为：`n_estimators=100`，
`learning_rate=0.05`，`num_leaves=15`，`min_child_samples=10`，
`subsample=0.9`，`colsample_bytree=0.9`。

特征在最后一次可见观测处构造，预测目标周时只增加该周已知的日历项：

- 滞后 1、2、3、4、6、8、12 周
- 4 周与 8 周滚动均值
- 一阶差分（lag1 − lag2）
- 目标周的年周期 Fourier 项（ISO 周序的 1、2 阶正弦和余弦）

拟合后的 23 个分位数会裁剪到 [0, 100]，并做保序处理，避免分位数交叉。

**2. 保形分位数回归（CQR）。** 对每个对称分位数对 `(τ, 1−τ)`，用最近最多 26 个
已实现起点的预测，计算一致性得分 `S = max(q_lo − y, y − q_hi)`，再按
`(1−α)(1+1/n)` 分位数（`α = 2τ`）把区间**单向外扩**，负的调整量记为零，中位数不动。
也就是只加宽、不收缩：在本序列上，收缩会损害尾部覆盖率。最终提交前再次裁剪并保序。

回测中还保留了两个对照成员，不进入本次提交：

1. **季节朴素**：用 52 周前的同周观测作点预测，在原始百分点尺度上用历史残差分位数构造区间。
2. **气候态 + 距平持续**（`clim_persist`）：目标周的多年周气候态，加上半衰期约 4 周的距平持续。作为原套件中 Prophet 的稳健替代（本仓库不依赖 Prophet / cmdstanpy）。
3. **分位数等权集成**：LightGBM 与气候态成员在 23 个分位数上取平均；季节朴素只作基线，不进入集成。

## 如何确保模型可靠

可靠性主要靠「不泄露未来信息」的滚动起点回测、与简单基线比较，以及区间校准来约束。

- **无前瞻。** 起点 `R` 只能看到 `target_end_date ≤ R − 2` 周的观测；特征、训练标签和气候态都在这条截断线之内。
- **滚动起点回测。** `reference_date` 从 2024-01-01 到 2026-09-14，共 142 个周一，步长 −1…6，报告滞后 2 周。这样评估的是近几年、包含完整季节的真实提交情景，而不是随机打乱的样本内拟合。
- **公平比较窗口。** CQR 需要先积累校准样本：每个步长前 16 个起点不输出 CQR 结果。指标在校准窗已就绪的子集上计算（每步长 126 个起点，合计 1008 个预测点），避免把尚未校准的早期点混进来。
- **多模型对照与事先规则。** 同一套回测里同时评估季节朴素、气候态、LightGBM 及其 CQR / 集成版本。提交方法按平均加权区间分数（WIS，23 个分位数得分的平均）选择，结果是 lightgbm_cqr。
- **覆盖率检查。** 除 MAE、WIS 外，报告 50%、80%、95% 区间的经验覆盖。CQR 后 80%/95% 覆盖从 0.79/0.92 提到 0.85/0.96，更接近名义水平；50% 覆盖约为 0.57，略偏宽。点预测的 MAE 与未校准 LightGBM 相同（中位数不被 CQR 移动）。
- **输出约束。** 预测值限制在流行病学上合理的 [0, 100] 范围内，并强制分位数单调，减少无效区间。
- **格式校验。** 提交前检查列名、行数、分位数集合、单调性、[0, 100] 范围和日期关系；hub 仓库的 CI 会在 pull request 上运行 hubValidations。

需要注意的限制：训练使用了 2009 年以来的全部可见历史（包括新冠期间），没有单独做疫情年份剔除或外部协变量（如疫苗、优势株）；50% 区间仍略宽于名义水平；早期 11 个缺失周按可用观测的滞后构造特征，不做插值补全。

## 回测摘要

指标在「CQR 校准窗已就绪」的公平子集上计算（1008 个预测点）。WIS 为 23 个分位数得分的平均。

| 方法 | MAE | WIS | 50% 覆盖 | 80% 覆盖 | 95% 覆盖 |
|------|-----|-----|----------|----------|----------|
| **lightgbm_cqr（提交）** | 3.35 | **2.20** | 0.57 | 0.85 | 0.96 |
| lightgbm | 3.35 | 2.21 | 0.49 | 0.79 | 0.92 |
| ensemble | 3.86 | 2.57 | 0.53 | 0.85 | 0.96 |
| ensemble_cqr | 3.86 | 2.60 | 0.66 | 0.90 | 0.98 |
| clim_persist_cqr | 4.90 | 3.30 | 0.64 | 0.90 | 0.97 |
| clim_persist | 4.90 | 3.39 | 0.47 | 0.82 | 0.93 |
| seasonal_naive | 9.60 | 5.67 | 0.52 | 0.91 | 0.94 |

按 WIS 选择 LightGBM+CQR。详细分步长指标见 `results/backtest_metrics_by_horizon.csv`。

## 本次提交（reference_date = 2026-11-09）

最后一次观测是 2026-10-26，阳性率 **12.4%**。中位数轨迹：

| horizon | target_end_date | 中位数 (%) |
|---------|-----------------|-----------|
| -1 | 2026-11-02 | 12.10 |
| 0 | 2026-11-09 | 11.45 |
| 1 | 2026-11-16 | 11.75 |
| 2 | 2026-11-23 | 12.43 |
| 3 | 2026-11-30 | 12.77 |
| 4 | 2026-12-07 | 15.01 |
| 5 | 2026-12-14 | 18.41 |
| 6 | 2026-12-21 | 22.00 |

2026-11-09 是距今最近的周一，且是 `hub-config/tasks.json` 中的合法轮次。
该周提交截止为北京时间周三 23:59，首次提交略晚于截止时间。

## 如何运行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_forecast.py --reference-date 2026-11-09
```

默认会从 hub 仓库下载最新的 `target-data/time-series.csv`。若要使用本地文件：

```bash
.venv/bin/python run_forecast.py --data data/time-series.csv --reference-date 2026-11-09
```

已有回测缓存时，可用 `--skip-backtest` 只重做最终预测。
输出：

- `output/GZNL-FluLGBMCQR/<reference_date>-GZNL-FluLGBMCQR.csv` — hubverse 格式预测
- `metadata/GZNL-FluLGBMCQR.yaml` — 模型注册元数据（`designated_model: false`）
- `results/backtest_metrics.csv` — 回测汇总

本机未安装 R / hubValidations，提交前做了列名、行数、分位数集合、单调性、
[0, 100] 范围和日期关系的格式检查。仓库 CI 会在 pull request 上运行 hubValidations。

## 目录

```
flu_model.py          特征、成员模型、CQR、回测与导出
run_forecast.py       训练 / 回测 / 写出 CSV 与 YAML 的入口
requirements.txt
metadata/             模型注册 YAML 副本
output/               hubverse 预测 CSV
data/time-series.csv  本次运行使用的官方 time-series 快照
results/              回测指标（及可选的逐点预测）
```
