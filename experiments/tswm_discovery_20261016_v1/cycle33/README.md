# Native WorldTS frozen-readout attribution

这是通用时间序列world-model研究中的机制控制实验，不是新模型、单数据集调参目标或已验证Research Gap。实验问题是：固定原生learned encoder/predictor时，普通decoder监督、容量和信息聚合能否解释此前怀疑的target/readout缺陷？

## Protocol and evidence

- [冻结协议](NATIVE_READOUT_PLAN.md)：NP/PJM H24、同一stage1、四种普通head、三个decoder seeds、20 epochs；先锁定所有fit/选模，后统一TEST。
- [执行与debug记录](OPERATIONS.md)：科学训练未重启；独立复核器日期解析预检修正保留原版本。
- `outputs/native_cache_v1/`：float32 TRAIN/VAL特征及完整manifest；所有来源数据、源码、旧checkpoint SHA和未来target不影响预测分支的检查。
- `outputs/readout_fits_v1/`：24组完整曲线、checkpoint、共同样本顺序和训练seal。
- `outputs/test_evaluation_v1/`：预列56组原始预测、所有缓存、完整指标、独立复核及provenance闭合。
- `outputs/analysis_v1/analysis.json`：逐seed配对下降、选模控制及168连续origin区块敏感性。
- `outputs/figures_v1/`：PNG/SVG/PDF与绘图原值。误差条只表decoder seed变化，stage1 seed固定。

训练/评估和证据验证脚本位于[scripts](scripts)。`verify_readouts33.py`是保留的未执行原复核器，`verify_readouts33_v2.py`是实际修正版本。原科学脚本、预算、数据和选择标准均未改动。

## Interpretation

native prediction-only、加宽decoder和global MLP都是普通控制，任何收益优先归为已有修复。两种电力任务都是此前看过的旧数据，只能支持本轮归因；它们不能单独证明跨领域能力。低读出分数也不能证明信息丢失，仍需优化与同输入观测空间模型控制。

最终方向、负证据和下一步判断以本目录七份研究报告为准。整体发现目标未因本轮训练完成而完成。未经文献区别和独立机制验证，不将本轮提升命名为新world-model框架。

## Reproducibility scope

服务器保留原始数据、旧模型、全数组、实际state_dict与日志；本地只镜像紧凑报告、源码和图，完整SHA见`COMPACT_MIRROR_MANIFEST.json`。旧data/checkpoint根目录由`native_common33.py`声明，公开核心revision由`frozen_source_manifest.json`锁定；这不是一个已公开发布的自包含数据包。重跑应使用新的独立输出目录，现有worker会拒绝覆盖已存在的fit/evaluation目录。
