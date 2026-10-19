# C36 模型、来源与执行登记

更新 UTC 2026-10-16T22:15:52.350331+00:00。C36 是已完成的文献/源码及数据资格审计，整体目标 ACTIVE_UNACHIEVED，接受最终方向 0。本轮没有模型训练、预测误差或方法效果实验。

目标为可复用时间序列 world-model 的状态学习、条件转移与观测生成框架。数据集只是证据载体；允许分别训练，不强制共享零样本 checkpoint、action、因果/物理识别、Loop 或 LLM。

| 对象 | 固定来源 | 本轮实际状态 |
|---|---|---|
| WorldTS | `269426ea6217d7152a244da0e209d922acd46d4f`，此前原生执行与权重保留 | 本轮读论文/旧登记，不重训；纠正“未找到 code”的视角检索遗漏 |
| KITE，ICML 2026 | `3140ee824cbd80c5ec7fdf2b54666210519d0b39` | 保存 README、requirements、7 Python source 与 repo/commit/tree；SHA/Git blob 核查。未导入、下载权重或训练 |
| Chronos-2 / Cross-Unet / DSSMF / PBC / joint models | 原文与作者代码链接见文献报告/三份 notes | 机制边界核查，不报本轮复现成绩 |
| OpenSTEF/openstef | `ca716897a5b9e5ae767a4c02e5efafb24425a654` | 保存消费 loader/as-of 版本类与示例；没有 weather 构造链的完整证据 |
| OpenSTEF Liander2024 HF | `dce7fe9bbae0d62288986fa97fa1ee7e9d3b7044` | metadata/card/target yaml；未下载约 650MB signal files。当前 Python3.12+ source 不能冒充已在服务器3.10原样运行 |
| Elexon WINDFOR/FUELHH | 官方接口一日固定 JSON，整文件 SHA、URL、下载 UTC 保留 | 本地公共 HTTP200 下载后整文件 SHA 上传；服务器独立选择/原生频率审计执行成功 |

source_audit 共20个 source/card/config 文件、277946 bytes；它不包含模型权重，也不是20次实验。元数据与数据的文件数分开登记。服务器直接访问 Elexon swagger 的 V1 为403，保留原源码和输出目录；V2使用透明的本地公开下载传输，未绕过TLS或权限。三种研究视角建议的6arms×2tasks×2seeds没有运行。

资格 probe 无模型、无随机 seed、无 train/val/test，没有以它选超参数。全部历史成果只读保留。数据/源码整文件回读校验记录与科学结果分别位于 manifest、receipts 和 outputs；不会把传输成功当模型复现。
