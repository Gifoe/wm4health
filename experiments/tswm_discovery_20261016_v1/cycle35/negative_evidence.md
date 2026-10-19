# C35 反证与限制

未通过事先定义的双任务screen；停止当前具体分支保护组合的扩训，不改loss/字段/阈值挽救。

- solar：PZ相对PY / JZ_N / PZ_NC的VAL MAE下降分别为 2.366% / -1.627% / 9.549%；预列screen passes=False。
- jena：PZ相对PY / JZ_N / PZ_NC的VAL MAE下降分别为 0.000% / 0.000% / 0.000%；预列screen passes=False。

正/负判断仅限这套单seed、小配置、预算有限的TRAIN/VAL screen；不是所有latent innovations或全部world models的判决。0/12 arms仍有cap改善标记。两个任务不是广泛多域确认，VAL重复使用更不能宣称TEST泛化或统计显著。

V1数值非确定性被实现断言拦下，不能当作研究失败：默认CUDA首步约4e-9梯度差，foreachFalse单独不足，deterministic控制恢复12步相同。V2正式两任务21检查step的保护P状态与numerical分数完全相同。没有通过放宽科学一致性容差掩盖它。

PZ/PY对比捆绑latent目标和表示位置；PZ_NC只表示相对已学习numerical representation的条件价值，不证明给定全部raw历史仍有独立信息。梯度余弦只是一固定TRAIN batch诊断，不是总体冲突结论。教师latent选模并不保证forecast充分。

当前不重启旧NP/PJM读出、Loop或IS wrapper路线，也不因原线索失败而自动接受替代架构。
