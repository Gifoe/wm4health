# 竞争假设及下一项反证

| 假设 | 可区分实验 | 当前状态 |
|---|---|---|
| 只有最后坐标标准化引起前缀变化 | 内部-only vs完整native | 否定 |
| 系统内部使用分支未来统计 | 固定参数/输入前缀、后缀重采；独立参考控制 | 128系统支持 |
| 自定义重放/数值误差 | 原生WHDataset精确重放、NumPy递推 | 排除 |
| 强共享latent模型的证据编码已经足够一致 | 原生物理query moments及真实后验预测 | 尚未执行 |
| 原生重新编码与其隐含联合律偏离 | no-update / 普通likelihood reweighting / exact physical posterior | 尚未执行 |
| 即便偏离也不影响用途 | 真证据proper scores/信息动作，与MC数值误差分开 | 尚未执行 |

下一工作不是再训练一个弱CNP制造缺陷。先执行NEXT_EXECUTABLE_GATE.md中六个已发布toy模型的低成本核查；其context上限10、参数维度1允许可信数值积分。两个actuation variant共享同一plant，不能算跨多个真实系统成功。未知结果不能提前写入推荐。
