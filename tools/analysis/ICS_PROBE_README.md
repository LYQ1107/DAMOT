# ICS 子空间探针：检验被收缩子空间中的天气与身份信息

目标：用已训练好的 checkpoint（不重新训练）检验三件事，不预设结果必须支持假设

1. **对得上**：ICS 子空间 $E$（m=2，D=256）装下了大部分真实天气变化和 FAP 残差；
2. **不含身份**：$E$ 里几乎没有区分不同目标的信息；
3. **有因果效果**：收缩 $E$ 后天气信息下降，而身份检索不降。

DAMOT 模型保持冻结，仅做前向推理。GT 框只用于放置 RoI 和给分析打标签，不参与 DAMOT 参数更新；天气线性分类器仅在按视频划分的 fit 部分拟合。

本地执行修订：
- `DarthQDTrack.eval()` 返回 `None`，模型移动设备与 `eval()` 必须分开调用。
- 干跑检查全部抽样帧；默认使用 4 个独立 CPU 进程预取视图。`--workers 0` 可使用串行路径，`--view-batch-size 2` 可降低显存。
- 同图各档 FAP 共用随机数，保证掩码随强度嵌套；默认 FAP 与 0.90 档特征一致。photo/student 共用光照参数。
- P1 指定 `--ref-domain` 时计算相对 Clear 的偏移。P4 同帧身份对按视频区分局部 COCO image id。
- R2 保留原配置的 `causal_min_samples=96`，解析后的配置仅 `causal_pca_mode` 不同。
- BDD 的跨域均值差异还包含场景、相机和车辆组成变化，不能单独视为纯天气的因果证据。P3 每档重新拟合线性探针，部分可逆收缩不一定减少可解码信息。
- student 探针固定几何，只保留颜色扰动与 FAP；训练时的完整两视图还包含几何增强。

---

## 0. 文件

| 文件 | 作用 | 运行位置 |
|---|---|---|
| `tools/analysis/ics_probe_extract.py` | 对每帧生成多种视图（clean / FAP / photometric / student / 剂量梯度），取每个 GT 框的关联嵌入 | GPU |
| `tools/analysis/ics_probe_analyze.py` | P1–P6 全部指标、LaTeX 表、图 | CPU |
| `tools/analysis/ics_probe_selftest.py` | 合成数据自检（植入已知天气子空间，检查分析代码是否正确） | CPU，约 1 分钟 |
| `tools/analysis/run_ics_probe_bdd.sh` | BDD100K 一键流程 | GPU+CPU |
| `configs/.../darth_spaP1_causalPairRes_roiV1Weak_4domain_5e.py` | 对照组 R2：ICS 改用成对残差估计（唯一改动） | 训练服务器 |

环境：沿用 rebuttal 的 V100 环境（`$RUN/env`，torch 1.12 / mmcv-full 1.6 / mmdet 2.25 / sklearn 0.22）。脚本均兼容 Python 3.7。

---

## 1. 准备（只做一次）

```bash
# V100 服务器
export RUN=/data1/liuyeqiang/damot_rebuttal_20261026
export PY=$RUN/env/bin/python
export OUT=$RUN/results/ics_probe
export GPU=0                                   # 空闲的卡
export MKL_THREADING_LAYER=GNU                  # 沿用隔离V100环境的OpenMP设置

# 把本 patch 里的新文件放进 rebuttal 使用的代码快照
# （在任意 DAMOT 仓库克隆里 git apply ics_probe.patch 后执行）
cp tools/analysis/ics_probe_*.py tools/analysis/run_ics_probe_bdd.sh $RUN/code/tools/analysis/
cd $RUN/code

# 检查所需输入是否齐全
ls $RUN/cache/weights/{F_stage1_e5,F_stage2_e5,F_stage3_e5,F_stage4_e5}.pth
ls $RUN/data/annotations/{clear_holdout_45v,overcast,rainy,dawn_dusk,night}.json
ls -d $RUN/data/views/{clear_holdout_45v,overcast,rainy,dawn_dusk,night}
```

若 2.2 干跑报 `image not found`，说明 `data/views/<domain>/` 下没有图片。此时设置 `IMG=<目录>`，使 `<目录>/<domain>/<标注里的 file_name>` 能找到图片（例如对原始 BDD 图片目录 `.../DA-MOT/BDD100K_<domain>/` 建一层 `<domain>` 软链接），再重新干跑。

---

## 2. BDD100K 主实验（约 1–1.5 小时，单张 V100）

```bash
cd $RUN/code

# 2.1 自检：分析代码在合成数据上必须全部 PASS
bash tools/analysis/run_ics_probe_bdd.sh selftest

# 2.2 干跑：只检查标注和图片路径，打印每个域抽到的视频/帧数
bash tools/analysis/run_ics_probe_bdd.sh dryrun
#   期望输出 5 行 [plan] ... missing(sampled)=0

# 2.3 主 checkpoint（F_stage4_e5）：clean/fap/photo/student + 剂量梯度视图
bash tools/analysis/run_ics_probe_bdd.sh extract
#   确认 log 里 [model] 的 bank_initialized=true、bank_num_updates>0
#   每个域最后一行 [done] <domain> rois=... dose_rois=...

# 2.4 前三个阶段的 checkpoint（只要 clean+fap，用于 P5 阶段演化）
bash tools/analysis/run_ics_probe_bdd.sh stages

# 2.5 分析（CPU）
bash tools/analysis/run_ics_probe_bdd.sh analyze
```

也可以一次跑完：`bash tools/analysis/run_ics_probe_bdd.sh all`。

抽样默认值：每域最多 40 个视频、每视频 30 帧（步长 2）、每帧最多 40 个 GT 框、只取 car（APR 也只用 car）。如需全部类别：在 extract 命令加 `--classes all`，analyze 加 `--classes 0,1,2,3,4,5`。

---

## 3. 对照组 R2：成对残差估计的 ICS（需训练，4 卡约与原 F 相同时长）

当前最终配置的 ICS 用的是 `proto_shrinkage_cov_ema`（类中心化特征的方差主方向），它不一定对准天气。R2 只把估计方式换成成对残差（`pairwise_residual`），用来比较两种估计谁更像"天气子空间"。

```bash
# 训练服务器（darth-main 根目录）
cd /data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main
cp <patch 中的>configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalPairRes_roiV1Weak_4domain_5e.py \
   configs/adapters/darth/bdd_4domain/module_combinations/
PYTHON=/home/lwr/anaconda3/envs/DARTH/bin/python
CFG=configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalPairRes_roiV1Weak_4domain_5e.py
ROOT=work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalPairRes_roiV1Weak_4domain_5e
mkdir -p $ROOT/logs $ROOT/episode && cp $CFG $ROOT/
CUDA_VISIBLE_DEVICES=9,8,7,6 $PYTHON -m torch.distributed.launch --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29631 tools/run/train_seq3_episode.py \
  --base-config $CFG \
  --source-ckpt work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/epoch_20.pth \
  --target-order overcast rainy dawn_dusk night --eval-domains overcast rainy dawn_dusk night \
  --work-dir $ROOT/episode --epochs-per-stage 5 --samples-per-gpu 16 --workers-per-gpu 2 \
  --seed 1790355234 --launcher pytorch --no-validate > $ROOT/logs/train_episode.log 2>&1

# 确认 ICS 真的在更新（应看到 pairwise_residual 且 update_ok=True）
grep -m5 "pairwise_residual" $ROOT/logs/train_episode.log
grep -c "update_ok=True" $ROOT/logs/train_episode.log
```

训练完后回到 V100 做探针：

```bash
cp <训练服务器>/$ROOT/episode/adapt/stage04_after_night/epoch_5.pth $RUN/cache/weights/R2_stage4_e5.pth
cd $RUN/code
# 抽特征（全部视图，与主实验相同）
MAIN=R2_stage4_e5 bash tools/analysis/run_ics_probe_bdd.sh extract
# 以 R2 为主分析
CUDA_VISIBLE_DEVICES= $PY tools/analysis/ics_probe_analyze.py --feat-root $OUT/feats \
  --label R2_stage4_e5 --domains clear_holdout_45v overcast rainy dawn_dusk night \
  --ref-domain clear_holdout_45v --classes 1 --montage --out $OUT/analysis_R2_stage4_e5
# 在 F 的特征上同时比较两个 bank（表 P1 会多一行 "ICS (R2_stage4_e5)"）
CUDA_VISIBLE_DEVICES= $PY tools/analysis/ics_probe_analyze.py --feat-root $OUT/feats \
  --label F_stage4_e5 --compare-labels R2_stage4_e5 \
  --domains clear_holdout_45v overcast rainy dawn_dusk night \
  --ref-domain clear_holdout_45v --classes 1 --out $OUT/analysis_F_vs_R2
```

---

跨模型比较注意：F 与 R2 的嵌入坐标可能随训练改变。主要比较应使用相同 RoI 在各模型自身嵌入空间中的结果；`--compare-labels` 把另一模型的 bank 直接投到当前模型特征上，只能作为补充诊断，未检查或对齐坐标时不能视为严格的估计器对照。

## 4. VKITTI2 成对实验（最强证据，可选；需要 VKITTI2 上训练好的 DAMOT checkpoint）

VKITTI2 同一场景、同一帧、同一辆车有 clone/fog/rain/morning/sunset/overcast 多个版本，可以直接得到"只差天气"的特征差。

```bash
export VK_CKPT=<VKITTI2 上 DAMOT 最后阶段 checkpoint>
export VK_ANN=<VKITTI2 的 COCO-video 标注 json>
export VK_IMG=<对应 img_prefix>

# 4.1 先看视频名格式，写出能解析 scene 与 variant 的正则
$PY -c "import json;d=json.load(open('$VK_ANN'));print([v['name'] for v in d['videos']][:30])"
#   例如名字是 Scene01_fog 则用 '(?P<scene>Scene\d+)[_/-](?P<variant>[A-Za-z0-9-]+)'

# 4.2 抽特征（所有视频都取，保证各天气版本抽到同样的帧）
CUDA_VISIBLE_DEVICES=$GPU $PY tools/analysis/ics_probe_extract.py \
  --checkpoint $VK_CKPT --label VK_final \
  --domain vkitti2 $VK_ANN $VK_IMG \
  --video-regex '(?P<scene>Scene\d+)[_/-](?P<variant>[A-Za-z0-9-]+)' \
  --max-videos 100000 --frames-per-video 60 --frame-stride 3 \
  --views clean,fap,photo --dose-frames-per-video 3 \
  --classes Car --out-dir $OUT/feats
#   --classes 用该标注里的类名（VKITTI2 常为 Car）；若标注按天气拆成多个 json，
#   就写多个 --domain <variant> <json> <img_prefix>，并去掉 --video-regex。

# 4.3 分析（--paired：按 scene+frame+instance 跨天气配对）
CUDA_VISIBLE_DEVICES= $PY tools/analysis/ics_probe_analyze.py --feat-root $OUT/feats \
  --label VK_final --domains vkitti2 --paired --ref-variant clone \
  --classes 1 --montage --out $OUT/analysis_VK_final
```

注意：
- 抽特征时 `--classes` 写标注里的类名；分析时 `--classes` 写类别下标（`CLASSES=('pedestrian','car',...)`，不区分大小写，Car→1）。类名不在该列表里的记为 -1，分析时改用 `--classes -1`。
- 日志里 `[paired] N matched (object, frame) pairs` 的 N 应为几千以上。若接近 0，说明转换 COCO 时 instance_id 被按视频重新编号，需要用 VKITTI2 原始 trackID 重新生成标注。

---

## 5. 输出与判读

`$OUT/analysis_<label>/` 下：

| 文件 | 内容 | 支持结论的判据 |
|---|---|---|
| `csv/P1_energy.csv`、`tables/tab_P1_energy.tex` | 各子空间装下的能量占比：真实天气偏移、FAP 残差、身份差异 | ICS 行的天气占比 ≥ 10×chance（m/D≈0.008），且明显高于 Random；身份占比接近 chance；Weather/Identity ≫ 1 |
| `csv/P1_angles.csv` | ICS 与各子空间的主角 | 与 "Oracle weather" 的角度小，与 "Oracle identity" 的角度接近 90° |
| `csv/P2_dissociation.csv`、`tab_P2_dissociation.tex` | 天气线性探针（平衡准确率）与身份检索 mAP，在 $E$ / $E^\perp$ / 全空间 / 随机 2 维 / 身份 oracle 2 维上 | $E$ 上天气准确率高于随机 2 维，而 $E$ 上身份 mAP 低；$E^\perp$ 身份 mAP ≈ 全空间 |
| `csv/P3_intervention.csv`、`tab_P3_intervention.tex` | 收缩 β∈{0,…,1} 后：天气准确率、天气类间距、身份 mAP（VKITTI2 另有跨天气 mAP） | 收缩 ICS：天气准确率与类间距下降、身份 mAP 基本不变（VKITTI2 跨天气 mAP 上升）；收缩随机子空间几乎无变化；收缩身份 oracle 子空间身份 mAP 明显下降（负对照） |
| `csv/P4_dose.csv`、`figs/fig4_dose_response` | 逐级变暗 / 加雾 / FAP 强度时，特征变化落在 $E$ 内外的大小 | $E$ 内变化随强度单调增大，且比 $E^\perp$ 内变化大；黑线（换成另一辆车）的变化主要在 $E^\perp$ |
| `csv/P5_*`、`figs/fig6_stage_heatmap` | 每个阶段结束时的 ICS 子空间装下各域相对 clear 的偏移占比，以及相邻阶段子空间夹角 | 刚适应完的那个域（对角线）占比最高；子空间随新域转动 |
| `figs/fig1_quadrant` | 2×2：$E$ 平面 / 补空间 t-SNE × 按天气着色 / 按身份着色 | 左上天气分团、左下身份混在一起；右上天气混合、右下身份成簇 |
| `figs/fig2_arrows` | 同一目标 clean→FAP（VKITTI2 为 clone→其他天气）的特征位移箭头 | 箭头大致平行、沿 $e_1$ |
| `figs/fig3_montage_e*.png` | $e_1$、$e_2$ 投影两端各 10 个 RoI 截图 | 一端偏暗/雾/低对比，另一端偏亮；两端车型混杂 |
| `figs/fig5_energy_bars` | P1 柱状图 | — |
| `summary.json` | 以上全部数值 | — |
| `figs/all_figures.pdf` | 按顺序汇总全部图，保留矢量文字和图形 | 用于逐页检查可读性 |

回传给作者：`$OUT/analysis_*` 整个目录、`$OUT/logs/`、`$OUT/feats/*/meta.json`。`feats/*.npz` 较大，除非需要重画图，否则不用传。

---

## 6. 常见问题

- **`checkpoint lacks required keys`**：checkpoint 与配置不匹配，加 `--config $RUN/configs/<label>_embedded.py`。
- **`has no initialized ICS bank`**：该 checkpoint 训练时 ICS 未启用或从未更新（`meta.json` 里 `bank_num_updates=0`），对应结果不能用来分析 ICS。
- **显存不足**：`ics_probe_extract.py` 每次前向 4 个视图，可加 `--view-batch-size 2`。
- **t-SNE 太慢**：只影响 Fig. 1，可把 `ics_probe_analyze.py` 中 `figures()` 里的 4000（抽样点数）改小。
- **截图拼图（Fig. 3）为空**：分析机器读不到原图路径。去掉 `--montage` 或在能访问图片的机器上运行 analyze。
- 所有随机性固定（`--seed`），重复运行结果一致；可换 `--seed 1/2` 再跑一次 analyze 观察划分稳定性。
