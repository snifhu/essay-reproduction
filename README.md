# 论文复现共享记录

本仓库用于在多台电脑之间共享复现代码、实际运行产物和分析所需数据。

## 本次跑通的产物（优先入口）

**HydEquivNet / 01022500 / Daymet / seed 42 / 60 epoch**

目录：[artifacts/hydEquivNet/01022500_daymet_seed42_60ep](artifacts/hydEquivNet/01022500_daymet_seed42_60ep/)

| 文件 | 用途 |
| --- | --- |
| `test_predictions.csv` | 1462 天测试观测与预测，方便绘图、残差及事件分析 |
| `checkpoints/01022500_HydEquivNet_Causal.pth` | 由最佳验证 NSE 选出的本次训练权重（epoch 49），不是作者权重 |
| `summary.csv` | 源码口径测试 NSE、KGE、Pearson r、降水缩放误差和 EUH 摘要 |
| `run.json` | 实际命令、执行时 Git 提交、配置、环境、总耗时和 PyTorch 显存峰值 |
| `train.log` | 全部 60 轮 loss 和验证 NSE |
| `gpu.csv` | 每约 5 秒采样的整卡显存、利用率、功率 |
| `raw_Q_metrics.json` | 将观测改回原始 Q 后的补充指标，未改变训练或预测输出 |
| `unit_hydrographs/` | 模型脉冲实验导出的 EUH CSV |
| `figures/` | 学习到的核及 EUH 图 |

结果：最佳验证 NSE **0.8364**；测试 NSE **0.73397571**、KGE **0.66848391**、Pearson r **0.88071448**。总耗时 **364.8 s**；显存整卡采样峰值 **1984 MiB**。RTX 5060 Laptop GPU 8 GB；Python 3.11.9 / torch 2.7.1+cu128。

**已跑通单流域流程，尚未精确复现论文数值。** 作者表中该流域 HEN NSE 为 0.77775714。没有运行其他流域、LSTM 或 Mamba。

## 在另一台电脑分析

```powershell
git clone https://github.com/snifhu/essay-reproduction.git
cd essay-reproduction
```

分析 CSV 不需要安装 PyTorch；使用现有 pandas/绘图环境即可。`.pth` 是 `state_dict`，需要本仓库的模型定义；以 `build_model('hen', 6, 5, ExperimentConfig(data_dir=''))` 构造模型，再用 `torch.load(path, map_location='cpu', weights_only=True)` 加载，调用 `model.eval()` 后推理。

分析时请保留以下口径：

- 日期配置为训练 1980–1999、验证 2000–2003、测试 2004–2008。源码各分段单独构造 365 天序列且少计最后一个样本；实际测试预测日期 **2004-12-30 至 2008-12-30**，不是完整 2004–2008 每天。
- 逐日 CSV 中 `observed_Q_mm_day` 是原始 Q（由 float32 log 标签逆变换恢复，存在舍入误差）；`simulated_exp_logQ_mm_day` 是原样 `exp(model_output)`，未减 0.01。源码指标用的观测为 **Q+0.01**；因此直接对 CSV 的原始 Q 计算 NSE，应接近补充指标 **0.73489737**，而不是 summary 中 0.73397571。
- `summary.csv` 的 PR 误差由源码取测试集首个 batch 计算，不是整个测试集均值，也不是可加性检验。
- 输入顺序为 `[Prcp,Srad,Tmax,Tmin,Vp,Swe]`，按训练集最大绝对值归一化；静态向量为 `[1,620.38,92.68,17.79072,0.9232]`，未归一化。面积换算沿用气象文件头 **587.675987 km²**，静态表面积为 **620.38 km²**。
- 式（6）检验降水放大后的比例响应，不证明两场降雨响应可加。EUH 是模型脉冲实验曲线，不是实测单位线；模型预测总流量，未显式拆分有效降雨、直接径流和基流。

完整说明：[SINGLE_BASIN_RESULT.md](SINGLE_BASIN_RESULT.md)。其中 E 盘绝对路径是历史记录，另一台电脑不需要同名盘符。`scripts/run_single_recorded.py` 等初次运行辅助脚本保留了原机路径，**暂不是跨电脑即用训练入口**；本次共享优先服务结果分析。作者 `scripts/train_camels.py` 支持显式 `--data-dir` 和 `--basin-list`；未经核实不要默认训练全部流域。

## 数据、参考结果与来源

- [data/01022500_daymet](data/01022500_daymet/)：本次使用的单流域 Daymet 气象、观测流量和原静态属性表。静态表保留原下载文件，包含其他流域行；仅 01022500 被用于训练。没有上传完整 CAMELS 压缩包、Python 运行时或论文 PDF。
- [reference/author_v1.0](reference/author_v1.0/)：**作者发布的参考表**，含 531 流域指标、EUH 及汇总。这些不是本次重新计算的结果，Mamba 列不代表已复现 Mamba。
- [published_checkpoint_eval.json](published_checkpoint_eval.json)：作者该流域权重在不同驱动、静态输入与归一化区间下的实测对比，不是新模型结果。其 Daymet + 截至 2003 年归一化 + 全零静态输入的 NSE 约 0.77772，支持驱动选择；论文正文未明确驱动版本，尚不能称完全同源。
- [preflight.json](preflight.json)、[provenance.json](provenance.json)、[input_manifest.json](input_manifest.json)、[archive_hashes.json](archive_hashes.json)：环境、真实静态值、原始全零行为、源码一致性及输入哈希证据。
- 531 个发布权重 ID 与参考表 ID 相同；尚未独立重建论文的筛选名单。

上游来源：[HydEquivNet v1.0](https://github.com/Zafkiellll/HydEquivNet/releases/tag/v1.0)、[CAMELS 下载记录](https://zenodo.org/records/15529996)。保留上游作者与数据来源署名；本仓库未为上游作品另行授予许可。

## 版本与后续同步

原始源码基线 `57d9f21`；本次训练执行提交 `45e1b1a0cfbb4290ae82cd26702cc082d033a630`；原结果归档提交 `d3e22a8`。静态读取及文件名适配、环境与结果均有 Git 记录。环境锁定在 `requirements-lock.txt`（CUDA 版 PyTorch 从官方 cu128 索引获取）。

以后把新的复现结果写入独立运行目录，记录参数、修改、代码版本及数据来源，提交并同步到本仓库；保留本次历史产物，避免覆盖。
