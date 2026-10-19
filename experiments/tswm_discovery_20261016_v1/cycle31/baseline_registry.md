更新 UTC 2026-10-16T19:51:23.870858+00:00。整体目标 ACTIVE_UNACHIEVED，已接受最终方向 0。此为过程报告，不能冒充最终已验证创新。

# 模型、协议与复现状态

| 底座 | 本轮实际执行 | 不能据此声称 |
|---|---|---|
| NormWear-2 core | 公开 safetensors 严格加载全部 277 keys，186,714,897 参数；原 card predict、simulate、原 no-action Bayesian；action Bayesian 原错误和普通修复 | 原论文生理数据性能复现、预训练从头复现或新方法 |
| WorldTS / C29 对照 | 复用历史证据，不在本轮重复训练 | 本轮新复现 |
| AdaJEPA | 官方论文/源码/权重目录核查，未下载权重或运行 | 已在单 5090 复现 |
| HiP-RSSM / MTS3 / MOCA / APE | 本轮仅核查机制及公开代码 | 已执行或已被新方法击败 |

NormWear-2：[HF 固定 revision](https://huggingface.co/mosaic-laboratory/normwear2/tree/3d1281c9994140056edfbbdedab57f2e6493dfaa)。文件 model.safetensors 748,201,812 bytes；SHA256 `44ddf4118b4b027a5b726ce7368439e0fa28fd3c331909149befac2087daad11`；全部权重留服务器。官方 demo 的纯代码 cells 已读取，未执行其私有数据/旧 pickle 加载 cell。

服务器 Python 3.10、torch 2.8.0+cu128、numpy 1.26.4、RTX 5090。依官方 demo 的 core 构造方式，使用固定 config；未安装 transformers、未改全局环境。公开 safetensors 直接使用 core 参数名；没有参数缺失、非严格加载或随机初始化替代。

v1 runner 对参数前缀的错误假设、v2 action Bayesian 越界均原样保留。v3 的 ordinary 修复在独立运行时函数与目录，发布源码/权重保持原样。
