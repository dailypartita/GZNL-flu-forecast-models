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

训练数据来自官方 hub 文件 `target-data/time-series.csv`，筛选
`target == "wk inc flu prop ili"`。每个起点只使用当时已发布的观测
（按当前仓库的发布节奏，约为 reference date 之前 2 周）。

## 模型说明

成员模型（用于回测比较；提交的是其中最优的一个）：

1. **季节朴素**：用 52 周前的同周观测作点预测，在原始百分点尺度上用历史残差分位数构造区间。
2. **气候态 + 距平持续**（`clim_persist`）：目标周的多年周气候态，加上半衰期约 4 周的距平持续。作为原套件中 Prophet 的稳健替代（本仓库不依赖 Prophet / cmdstanpy）。
3. **LightGBM 分位数回归**：对每个步长、每个分位数水平单独训练（直接多步）。特征为滞后 1–4、6、8、12 周，4 周与 8 周滚动均值，一阶差分，以及目标周已知的年周期 Fourier 项。
4. **分位数等权集成**：LightGBM 与气候态成员在 23 个分位数上取平均；季节朴素只作基线，不进入集成（与原 EpiETL 协议一致）。

提交模型 **FluLGBMCQR** 在 LightGBM 分位数预测之上做在线滑动窗口
[CQR](https://arxiv.org/abs/1905.03222) 校准：对每个对称分位数对，用最近 26 个起点的保形得分
`S = max(q_lo − y, y − q_hi)`，将区间按 `(1−α)(1+1/n)` 分位数**单向外扩**（不收缩），中位数不动。
预测值裁剪到 [0, 100]，并做保序处理以防分位数交叉。

## 回测摘要

滚动起点：`reference_date` 从 2024-01-01 到 2026-09-14，共 142 个周一，
报告滞后 2 周，步长 −1…6。指标在「CQR 校准窗已就绪」的公平子集上计算
（每步长 126 个起点，合计 1008 个预测点）。WIS 为 23 个分位数得分的平均。

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
