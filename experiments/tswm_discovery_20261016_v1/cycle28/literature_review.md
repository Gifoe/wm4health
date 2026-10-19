# 文献与创新边界：cycle28

研究对象仍是可复用的数值时间序列 world model。文献窗口冻结为2026-04-16至2026-10-16；实验在北京时间10月17日完成。三个独立视角分别保存在 research/numeric_frontier.md、capability_critic.md、adjacent_efficiency.md，完整 primary 链接和版本在笔记中。旧组件先例并不自动否定新组合；有限数据/计算的收益可以构成贡献，但必须先测出具体残差。

| 能力组 | 直接先例 | 本轮应如何解释 |
|---|---|---|
| 任意变量、图和未来条件预测 | [PCFM 官方源码](https://github.com/software-competence-center-hagenberg/PCFM)、[出版社记录](https://doi.org/10.1007/978-3-032-34896-8_15) | 完整发布新底座；9个边际分位数、masked reconstruction，不等于 joint path sampler |
| 系统级上下文识别与概率预测 | [概率 ICL-SysID](https://arxiv.org/abs/2610.13411)、[已缓存源码](https://github.com/PuraFilippo/sysid_benchmarks) | cycle16 已 materialize，本轮首次在本项目完成真实 CED 质量比较；不是新发现该论文 |
| TSFM 输入响应与适配保留 | [Attenuated in-context identification](https://arxiv.org/abs/2610.08118) | 已有合成适配、频率与遗忘诊断；普通 LoRA/分支等修复尚需排除，不能泛提 adapter 新意 |
| 可更新的低维模型动态 | [AURA](https://arxiv.org/abs/2609.31559)、[KAFFEE](https://arxiv.org/abs/2605.31547) | latent Bayes tracking 和过滤目标已有；任务与假设不同，不能声称已解决所有数值模拟 |
| 物理时间概率生成 | [VSF](https://arxiv.org/abs/2610.00976) | 原文 stochastic 数据明确是每轨迹参数不确定性；不能误读为过程噪声。当前官方树只有README/LICENSE/gitignore，未执行模型 |

与只用论文关键词相比，源码明确了 PCFM 默认 glue 是加性平移、双向时间注意力、8192窗口、CP仅历史估图；这些是契约边界，不能自身当失败证据。本轮已做实际发布推理与实测数据比较。其默认设置的问题有公开普通开关可修复；较小的原生 ICL 也表现较强，尚无支持新框架的残差。

KUMA/PRO-DYN/RP-GSSM、DynaBase/FIM 与旧 SysID 文献只作历史刷新，未重复下载或训练。缺代码/缺checkpoint只影响执行成本，不是科学Gap。没有把预测误差直接当物理忠实性，也没有把独立分位数采样制造的坏路径当作者失败。
