# HydEquivNet 单流域复现记录

已完成流域 01022500 的数据读取、60 epoch 训练、逐轮验证、最佳验证权重选择及独立测试。没有启动全流域训练、LSTM 或 Mamba。

## 结果

- 训练/验证/测试：1980–1999 / 2000–2003 / 2004–2008；seed 42；序列长 365；batch size 64，未降低；FP32，无 AMP。
- 最佳验证 NSE：0.8364，epoch 49（日志保留四位小数）。
- 源码口径测试 NSE：0.73397571；KGE：0.66848391；Pearson r：0.88071448。
- 以原始 Q 作为观测的补充 NSE：0.73489737。源码指标的观测是 exp(log(Q+0.01))，即 Q+0.01；补充口径未改变训练或模型输出。
- 总运行耗时（含训练、逐轮验证、测试和图表）：364.8 秒。
- PyTorch 峰值 allocated/reserved：481.7/522.0 MiB；每 5 秒采样的整卡显存最高 1984 MiB（含桌面及其他程序，可能漏掉瞬时峰值）。
- Python 3.11.9；PyTorch 2.7.1+cu128；CUDA runtime 12.8；NVIDIA GeForce RTX 5060 Laptop GPU；驱动 596.13。实际 CUDA 矩阵运算、模型训练和反向传播均通过。

## 忠实设置与适配

保留原网络、6 个气象输入、五项静态输入、损失、AdamW、学习率 0.001、weight decay 0.0001、余弦调度、梯度裁剪 1.0 及日期/随机种子/训练轮数。气象特征顺序为 Prcp、Srad、Tmax、Tmin、Vp、Swe，按训练期最大绝对值归一化。

适配修改：支持 CAMELS 原生 forcing/streamflow 文件名和 basin_physical_characteristics.txt；流域 ID 以字符串保留前导零；静态属性缺失/非有限值立即报错；气象文件匹配不唯一时拒绝运行；使用独立 Daymet 单流域目录。设置 4 个 CPU 线程并加入显存、耗时、版本日志。这些修改已提交 Git。

实际静态向量：[1, 620.3800049, 92.6800003, 17.7907200, 0.9232000]。原始加载器对本流域返回全零，即使仅把属性文件重命名也返回全零；测试已记录。森林字段虽标为 percent，但原数据是小数比例，未自行乘 100。

气象面积头为 587.675987 km²，静态面积为 620.38 km²；流量单位换算保留原源码的气象头面积。1980–2008 共 10593 天，连续且无负流量缺测标记；所选气象驱动均为有限值。

保留源码序列边界行为：各分段单独构造序列，首 364 天不评价，__len__ 还略去最后一个可能样本。训练/验证/测试样本数 6940/1096/1462；实际测试目标日期 2004-12-30 至 2008-12-30。未暗中将其修改为完整 2004–2008 每日评分。

## 与发布结果对比

发布表本流域 HEN NSE 为 0.77775714，本次为 0.73397571，差值 -0.04378143。这是单次、单流域的条件复现，不能据此判断整篇论文优劣或宣称精确数值复现。

驱动版本为 CAMELS basin mean forcing v1p2 / Daymet。论文可检索正文未明确指明 Daymet/Maurer/NLDAS。对同一发布权重进行三驱动与归一化/静态输入对比：Daymet + 截至 2003 年的归一化 + 旧全零静态输入得到 NSE 约 0.77771547，接近参考表；Daymet + 截至 1999 年归一化 + 正确物理属性得到约 0.76323617。该现象支持 Daymet 为本权重匹配驱动，但不是完整数据来源证明。

发布验证脚本另有 train_end=2003-12-31 的尺度计算设置，与默认训练/论文划分不同。新训练遵守 1999 年训练截止。不要为追近参考分数而用验证期计算新训练归一化。

本地 17 个 Python 源文件与 HEN_verified.zip 对应文件完全一致；531 个权重文件 ID 与结果表 531 个唯一 ID 完全相同。尚未独立重建论文的面积 <=2000 km²、面积差异 <=10% 筛选规则，因此扩大实验须以发布表 ID 为候选，并逐个核对数据可用性及筛选依据，不能直接调用所有流域。

仅发现 HEN 和可选 LSTM 的训练实现；Mamba 只有发布结果，未复现。NSE/KGE/Pearson 按源码输出，未将另一个近似高低流偏差函数混称为论文 FHV/FLV。

## 科研解释边界

式（6）检验放大降水后的比例响应，不验证两场降雨响应的可加性。EUH 是全模型脉冲实验导出的曲线，不是实测单位线。模型从原始气象预测总流量，没有显式分离有效降雨、直接径流和基流。输出 EUH 保留源码默认脉冲与背景设置，只作为模型诊断。

## 路径、命令与版本

- 原始资料：E:\HydEquivNet_精读资料，未修改。
- 独立解释器：E:\HydEquivNet_reproduction\runtime\python.exe
- Git 仓库：E:\HydEquivNet_reproduction\repo
- 单流域输入：E:\HydEquivNet_reproduction\data_single_daymet
- 运行产物：E:\HydEquivNet_reproduction\runs\single_60ep
- 训练日志：E:\HydEquivNet_reproduction\work\train_60ep.log
- 原始基线提交：57d9f21；本次执行提交：45e1b1a0cfbb4290ae82cd26702cc082d033a630。后续证据及报告提交可用 git log 查询。

实际训练命令：

```powershell
E:\HydEquivNet_reproduction\runtime\python.exe E:\HydEquivNet_reproduction\repo\scripts\run_single_recorded.py --epochs 60 --batch-size 64 --run-name single_60ep
```

日志重定向至上述 train_60ep.log。运行目录包含最佳权重、summary.csv、run.json、gpu.csv、测试逐日预测以及模型核/EUH 图和 CSV。环境依赖锁定在仓库 requirements-lock.txt，输入哈希在 input_manifest.json。

## 下一阶段建议

先核实 Daymet 来源和发布权重的训练/归一化历史；用发布表名单选择 3–5 个数据完整、气候不同的流域做同一配置试验，逐流域记录成本及差值，再决定扩大到 30 或 531。暂不训练 Mamba；需要其真实训练实现和配置后再另行复现。
