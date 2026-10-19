# C41 竞争假设与反证

| 假设 | 本轮证据 | 下一裁决 |
|---|---|---|
| 新环境缺的是可确定的信息 | 对齐反例完全同train law、不同held-out law | 正确的anchor/history coverage必须提供给所有模型；不以更大network解决信息不足 |
| 失败是latent坐标/弱probe | exact observable-law gauge control通过 | learned state用future law/excess proper risk；弱probe差不等于信息丢失 |
| 校准损伤来自错误transition的补偿 | exact4-state控制支持这一可能性，ordinary修复为0 | 不冻结错误Q当强baseline；共同坐标下区分posterior与transition |
| 双接口复用有有限数据交互收益 | 目前只有合法setting和ordinary positive control，无方法效果 | 同数据/推断/损失/预算2×2 + joint hierarchical SSM，检验held-out交互 |
| 所谓收益只是普通factorization | 是当前最强替代解释 | 若joint普通模型吸收全部收益，不新增模块换名 |

状态、推断和transition误差一般不独立相加。优先比较同一信息/同一generative model的inference，再检查observable law。新primitive无覆盖是压力条件，不承诺其一定泛化。
