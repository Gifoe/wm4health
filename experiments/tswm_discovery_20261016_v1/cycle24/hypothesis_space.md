# 竞争解释与实证状态

| 解释 | 区分证据 | 状态 |
|---|---|---|
| 未来数据/状态初值/随机漏历史污染 | 原因果源、完整前缀、独立query/probe reset | 排除本轮已知混淆 |
| 数值积分伪影 | uniform grid/refinement、全量尾界、SciPy/native重放 | 主结果通过，原失败尝试保留 |
| q encoder更新不等于其decoder联合律的条件化 | model-generated probe tower +同decoder精确重加权 | 支持，且有真实冷启动后果 |
| 普通likelihood/decoder也有误差 | 完整Bayes未全条件优于native；physical oracle更好 | 支持，不能只优化coherence |
| 只有复杂新模型才能修复 | defensive importance +Gaussian矩投影 | 本toy反证：旧方法已修复大部分 |
| 缺陷普遍且新框架必要 | 新现代模型/高维系统/复杂信息结构比较 | 尚未执行、不能宣称 |

原生先验的不足、后验近似误差和数值误差是不同解释。不能将后两者排除后就默认产生新的研究方向；还要证明问题未被已有方法充分解决，并明确实际用途。
