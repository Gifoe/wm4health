<!-- cycle31-current-start -->
# 最新 cycle31：按用户纠正研究范围

UTC 2026-10-16T19:51:23.870858+00:00。通用时间序列 world-model 框架，以 WorldTS 为研究层次参照；不强制零样本共享 checkpoint、因果/物理解释或某种结构。NormWear-2 原生预测执行、三处条件接口诊断和普通修复完成，独立复核通过；不是新方法或论文性能复现。接受方向0，整体 ACTIVE_UNACHIEVED。

见[范围纠正](cycle31/SCOPE_CORRECTION_20261017.md)、[实际证据](cycle31/experimental_results.md)与[当前研究判断](cycle31/final_research_direction.md)。以下各轮历史保留，其“当前”仅指当时。
<!-- cycle31-current-end -->

<!-- cycle29-index-start -->
# 最新证据cycle29

UTC 2026-10-16T19:21:51.067123+00:00。原始12/12及全量验证完成；等更新对照已复核=True，接受方向0。见[cycle29/baseline_registry.md](cycle29/baseline_registry.md)；以下历史保留。
<!-- cycle29-index-end -->

<!-- latest-cycle28 -->
# 最新记录：cycle28，整体研究未完成

最新完整内容见[cycle28/baseline_registry.md](cycle28/baseline_registry.md)和[CURRENT_RESEARCH_STATUS.md](CURRENT_RESEARCH_STATUS.md)。目标是通用时间序列world model；本轮跑通新发布PCFM，并完成真实CED同信息比较。普通glue开关已修复大量退化，已有概率ICL较强；最终接受方向仍0。以下早期历史材料中的“当前/下一步”只指当时。

---

# Baseline 与复现登记

所有新增文件位于独立目录。旧项目、数据与 checkpoint 只读使用，未删除或修改。以下三组不能混成排行榜。

## A. 历史原论文复现与负面结果（本轮重新查阅报告）

旧紧凑证据在 `_longhorizon_loop_dependency_audit_worktree/experiments/`，其 HEAD 为 `3849f8add3a546d4598229c0a1d19444b1a04ce8`；服务器原目录继续保留。

| 旧实验 | 状态/结论 | 本轮用途 |
|---|---|---|
| baseline_reproduction_seed2021_v1 | 24/24 历史 cells；22 MATCHED、2 CLOSE；21 fresh、3 verified reuse | 确认已有 LatentTSF/DAG/WorldTS 复现，避免重做；这些不是本轮新增结果 |
| conditional_latent_loop_seed2021_v1 | NO-GO | 不重启泛化的条件 loop 路线 |
| worldts_future_response_loop | INCONCLUSIVE | 不把不足证据改写成正面路线 |
| worldts_readout_alignment_feasibility_seed2021_v1 | NO-GO complex loss | 不重复 readout-aware loss 微调 |
| worldts_stage1_state_selection_audit_seed2021_v1 | NO_GO_NEW_STATE_SELECTION_METHOD | 不重复 epoch-1/state selection 假设 |
| neso_worldts_online_v1 | NO-GO persistent updater | 普通线性修正已解释大部分收益；真实在线 vintage 可用性仍有边界 |

## B. 本轮执行

| 对象 | 数据/信息 | 状态与公平性 |
|---|---|---|
| 4 个旧 WorldTS checkpoint：NP/PJM × H24/H360 | 各 8 个 TRAIN context，具名列恢复与训练分区标准化 | 冻结权重的有限差分/精确梯度审计；不是新增 forecast accuracy 复现，更不是电力数据上的因果干预 |
| WorldTS native two-stage / identical-core e2e / ordinary FIR | 新随机受控标量系统；history64、future actions48 | 各 3 seeds，新诊断迁移；同 train/val、验证选 ckpt；two-stage 训练阶段更多，不宣称严格 compute match |
| GRU | 同上，同输入、验证选择 | 14,881 参数，比 native 39,361 更小；普通强基线 |
| ARX / ridge | 同上训练 transition，验证选 ridge | 非神经系统辨识对照；系数从数据估计，没有使用真参数 |
| Neural Gaussian SSM | 独立 2D 非线性随机系统，12,288 train transitions | 3 seeds，验证均值 MSE 选 ckpt，3 个冻结模型用于独立 dev/confirmation 与后续 fresh cohort；自建诊断模型，不是 RP-GSSM 复现 |
| Affine Gaussian regression | 256/2,048/16,384 transitions，各 64 独立训练重复 | OLS 全局最优，排除神经训练不充分解释；两套独立实验各 192 fits |
| 区间对照 | same fitted kernel / confidence set / test trajectories | exact gauge-only、fixed-noise statistics-only、separate/paired norm tubes、shared-parameter Taylor tube、无 remainder 消融、oracle-known-kernel family |

RP-GSSM、PRO-DYN、TSWM、EpiCF、TIME、CoMP、CSM/APID、Lally IMDP 本轮只作代码/协议审查，没有伪报执行或复现成功。DoTime 的版本/配对/指标代码已查，但当前新模拟数据不是其官方 suite；不把合成系统成绩写到 DoTime 排行榜。

## C. 源码固定版本

| Repository | Exact commit | Download scope |
|---|---|---|
| Verified-Intelligence/auto_LiRPA | e3930353a3589f7d10879f1e71223d1eefb57196 | exact source archive |
| BenevolentAI/CoMP | 0da27c1795fefa6208ce2107f23fa534efdab833 | exact source archive |
| Valentyn1997/CSM-APID | a95c6d434c45f6c73413b864795055cfe527da24 | exact source archive |
| thummd/dotime | 8f227406a5b17c2040cfdcc680b493729ef47d00 | exact source archive |
| complex-ai-lab/epi-cf-benchmark | 8f7df27507dc8b01422bb373417c8d0d3efffae2 | metadata/readme/tree only |
| Muyiiiii/LatentTSF | 7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2 | exact source archive |
| ARBrachet/PRO-DYN | 733d986d349e9e71ded5d9cdeeeed8d3323d83b4 | exact source archive |
| jessicalally/robust-cf-inference-beyond-mdps | 2d1577efa3f28593d6af929d3dcedff45ef34588 | exact source archive |
| gatsby-sahani/rp-gssm | a75225d5edceb972bb872dc9a335608733567dd4 | exact source archive |
| zqiao11/TIME | 912291c69045d369ac45fcd5e96ecf60398c092a | metadata/readme/tree only |
| HaochenZhang717/TSWM-Official-Repo | 2e5e48540ff0da48e8192163969a6c58df1d5a3e | exact source archive |
| decisionintelligence/WorldTS | 269426ea6217d7152a244da0e209d922acd46d4f | exact source archive |

WorldTS 在本轮服务器的 `upstream/WorldTS` 中只读导入。其余近邻代码主要用于核对创新边界。权重/数据哈希见 `outputs/existing_future_support.json`、`outputs/provenance.json`。依赖版本、GPU、脚本哈希均已记录；新环境只通过只读路径复用旧依赖，没有写入旧环境。


CROWN新增执行状态：17个worker（H8 smoke一个；H32/H64各8个matched cases），smoke另有alpha-CROWN。通用库未修改，CPU单独环境，精确同一L2系数球与SO(2)全局角度集合。若最终worker数少于计划，以 `outputs/lirpa/registry.json` 为准，不将未完成条目计为成功。
