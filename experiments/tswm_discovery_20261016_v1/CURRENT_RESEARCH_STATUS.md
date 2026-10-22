# 当前研究状态 — 2026-10-23 V9

整体研究目标ACTIVE/UNACHIEVED，最终接受新颖方向0。本轮两项具体执行已闭合，旧报告/错误/快照全部保留。

## DALI迁移资格：停止该配置的机制归因

原生完整核心完成2000 intended更新；加原保存失败前500次，实际2500次调用。52个指标独立复算、110个源文件重验；65步filter/32步imagination与逐步调用、无提前响应的11组数组核验均完全一致。时钟资格通过。

最终两个族MSE0.5111/0.1581，仍是普通因果参考的2.33/1.92倍，未过预注册factual门槛。保留完整状态，停止当前配置追加训练和物理机制编辑。该结果未唯一归因预算、目标或迁移误差，不能证明需要新架构。[闭合报告](mechanism_edit_20261023_v1/factual_transfer_closure_20261023_v1/experimental_results.md)。

## Meta-iLaD当前发布版：真实原生复现与失败归因

固定2026-10-21新源码，MoCAP作者refined版：60个公开原始文件、24个数据张量指纹全部精确；9个现成checkpoint、9个benchmark单元、2个输入/重放资格和7个诊断共18次评估，0新fit。CPU主表Meta-iLaD ID23.249±4.025，OOD78.990±3.131；IDF ID26.780±4.215，原生信息协议不同，未制造IDF OOD比较。

Meta-iLaD18个参考指标均通过1%范围，CPU IDF4个未过。相同seed的CPU/GPU抽样使主案例MSE变化12.02%；固定全部251组Gaussian样本后相对预测RMS小于8e-7。GPU三个固定seed匹配全部6个IDF参考指标。原CPU表与GPU诊断分别保留，外部完整运行环境未被唯一恢复。普通随机解释排除了这个新增方法动机。

[原生源码/数据/评估与归因报告](meta_ilad_native_20261023_v1/experimental_results.md) · [独立指标核验](meta_ilad_native_20261023_v1/native_verification_v1/VERIFICATION_V1.json) · [噪声/边界独立核验](meta_ilad_native_20261023_v1/native_verification_v1/ATTRIBUTION_VERIFICATION_V1.json)。这不是原论文Table2或整篇论文复现。

## 三篇主线与下一项反证

模型主线现在有WorldTS、LatentTSF、Meta-iLaD三项独立模型工作的实际执行；TSWM另列机制benchmark/recipe。各自任务/输入和复现范围分组，不能称为同任务三种完整SOTA的统一榜单。[注册表](meta_ilad_native_20261023_v1/baseline_registry.md)。

候选仍是跨root/action的固定机制语义，尚无合格的物理编辑失败证据或新方法。下一步应核查原生物理配方和现有辨识理论，再用可控真值反证是否已被已有方法解决。当前不从随机性、普通loss修复或不足factual能力中强立新方向。

数据、权重、噪声bank、预测与环境留服务器G，未同步本地。GitHub保存紧凑代码、配置、CSV、SHA、回执与报告；旧证据不覆盖。
