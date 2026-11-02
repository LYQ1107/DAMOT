# HF 原生漂移能量与遗忘恢复

只读 HF 已有特征与已保存基底，纯 CPU；全部新增文件在本目录。

每格顺序为 **漂移能量 E% / 恢复遗忘 R%；η=R/E**。Random 的 E、R 各报告十个固定基底的均值 ± 总体 SD（ddof=0）；主表 η 是 R 均值除以 E 均值。

| 基底 | k=2 | k=8 | k=16 |
|---|---:|---:|---:|
| Intervention-sensitive | 5.613 / 0.769；η=0.137 | 19.731 / 36.243；η=1.837 | 31.826 / 62.704；η=1.970 |
| Random | 0.846 ± 0.146 / 1.167 ± 2.501；η=1.380 | 3.173 ± 0.520 / 5.707 ± 3.761；η=1.799 | 6.336 ± 0.415 / 12.489 ± 4.859；η=1.971 |
| Identity oracle | 6.897 / 39.605；η=5.743 | 19.425 / 48.043；η=2.473 | 30.631 / 53.406；η=1.743 |

**η 是两种归一化比例的商，不是因果效应或绝对 mAP 的物理导数。**恢复遗忘仍是跨阶段身份检索 mAP 损失的代理。Random 每个配对 replicate 的 η、以及十个 η 的均值/SD 另存 CSV；它们与主表的均值之商不混用。

## 计算口径与配对

沿用 [v3 split](../ics_figs_v3/split.json) 的测试视频和 car 类别，每个 stage 的 video/frame/instance/image/bbox/category/weather 均逐行严格相等。仅使用三个旧域的 **8,117 个一一对应 RoI**；不把检索 query cap 当作能量采样。检索任务哈希也与 v3 完全一致。

| 旧域 | 旧阶段 → query 阶段 | 测试 RoI 对 | 测试视频 | 漂移平方和 | 原检索 queries |
|---|---|---:|---:|---:|---:|
| overcast | 1 → 4 | 1813 | 8 | 871.325495674422 | 1142 |
| rainy | 2 → 4 | 2326 | 9 | 879.255399826818 | 1837 |
| dawn_dusk | 3 → 4 | 3978 | 17 | 1116.307007163235 | 2866 |

`d = z_stage4 - z_stage_j`，两侧均用与 v3 相同的 L2 归一化 clean 特征。

`E% = 100 × Σ三个旧域ΣRoI ||d @ E_basis||² / Σ三个旧域ΣRoI ||d||²`。先合并分子、分母再相除；不平均逐域或逐 RoI 的比例。

HF 原生 Intervention-sensitive 与 Identity oracle 完全复用 [v3/bases](../ics_figs_v3/bases/) 的六个 stage1 基底；随机使用 [A2/HF 已保存基底](../idapr/A/HF/bases/) 的三十个 `.npy`。没有重新估计、重新抽样、QR 或改写基底。全部二十个 HF 特征 NPZ 的哈希与 v3 原记录一致。

恢复比例直接取自 [v3 原始 CSV](../ics_figs_v3/drift_restoration_HF_native.csv)，不重新推理或重算检索指标。

## 随机理论值自检

均匀随机 k 维正交子空间的期望能量比例为 k/256：**0.78125% / 3.125% / 6.25%**。十次基底的实测均值不会严格等于期望。预先采用宽松的数值诊断：理论值在十基底均值 ±2SD 内；这不是显著性检验。

| k | 合并随机能量均值 ± SD (%) | k/256 (%) | 实测−理论 (pp) | 相对差 (%) | ±2SD 诊断 |
|---:|---:|---:|---:|---:|---|
| 2 | 0.845872 ± 0.145735 | 0.781250 | +0.064622 | +8.272 | PASS |
| 8 | 3.172936 ± 0.519579 | 3.125000 | +0.047936 | +1.534 | PASS |
| 16 | 6.335820 ± 0.414962 | 6.250000 | +0.085820 | +1.373 | PASS |

所有逐域随机诊断也保留在 [random_theory_selfcheck.csv](random_theory_selfcheck.csv)。该诊断不会导致重新选基底或更改数值。

## 文件与复现

[drift_energy_HF_native.csv](drift_energy_HF_native.csv) 包含 156 行：27 个逐域均值/确定性结果、9 个合并结果、90 个逐域随机 replicate 和 30 个合并随机 replicate。投影平方和、总漂移平方和、恢复比例、效率、随机 SD/min/max 全部保留。

输入与源码哈希见 [input_manifest.json](input_manifest.json)，配对计数和索引/任务哈希见 [paired_cohorts.csv](paired_cohorts.csv)，汇总表原始数据见 [energy_summary.csv](energy_summary.csv)。详细定义见 [protocol.json](protocol.json)。

使用 [README.txt](README.txt) 的 Python 与四线程设置运行 [scripts/run_all.sh](scripts/run_all.sh)。无 GPU、无训练，历史实验目录不改动。
