# C36 实际执行结果

更新 UTC 2026-10-16T22:15:52.350331+00:00。C36 是已完成的文献/源码及数据资格审计，整体目标 ACTIVE_UNACHIEVED，接受最终方向 0。本轮没有模型训练、预测误差或方法效果实验。

## 执行与传输

服务器执行 `scripts/probe_elexon36_v2.py` 与 `scripts/audit_cadence36.py`；输出分别在 `outputs/elexon_probe_v2/result.json`、`outputs/cadence_audit_v1.json`。V1直接访问官方 swagger 为HTTP403，原版本保留；三个原始文件由本地正常TLS公共HTTP200取得，再上传服务器整文件SHA回读核验。不是远端直接下载成功，没有登录/TLS绕过。

固定日期为2026-01-01发布的预报、2026-01-02实测；日期与三个 origins 在读取信号前冻结，不按误差挑日。原始 schema/forecast/actual 文件分别为1479300、66002、8737 bytes。服务器验证数据类型、finite、时区与自身 dataset/fuelType；没有计算 forecast errors。

## 数据计数与独立 as-of 选择

预报584行、73个独立 startTime，(publishTime,startTime) 重复0；发布时间02:30..22:30Z，目标范围2025-12-31 21:00Z至2026-01-03 21:00Z。实测48行、48个半小时 startTime，(publishTime,startTime,fuelType) 重复0。实测值只作后验 label 文件资格，不参与预报版本选择。

对每个目标时刻选择 publishTime≤origin 且 startTime>origin 的最新版本；bucket-max 与独立逐目标逐行扫描完全一致。结果如下：

| origin UTC | 可用未来半小时轴点 | 精确匹配预报 | 可用原生小时轴点 | 原生小时轴完整 |
|---|---:|---:|---:|---|
| 2026-01-01T12:00:00Z | 48 | 24 | 24 | True |
| 2026-01-01T18:00:00Z | 48 | 24 | 24 | True |
| 2026-01-02T00:00:00Z | 47 | 23 | 23 | True |

## 必须保留的频率解释

预报73个独立时点间隔为3600秒，共72个间隔；实测48个时点间隔1800秒，共47个间隔。24个匹配点是完整次日24小时的小时预报，并非只覆盖12小时，也不是缺24条应有的小时预报。midnight origin 使用严格 startTime>origin，故00:00点排除，剩23个。

[官方页面](https://bmrs.elexon.co.uk/wind-generation)也定义小时预报。V2原输出按48个半小时轴报告coverage，原字节保留；补充审计明确原生频率。没有插值、复制、平均或评分。未来实验必须先冻结预报值与实测区间的验证定义；同timestamp不自动保证区间平均/点值语义相同。

## 可复核文件与结论范围

V2 result SHA256 `de62454e8c634971658dfcaf489de05827f9d94f876adb0c9f9581f0876c186b`。原始 bytes/SHA/URL 与脚本/计划 SHA 均在输出和 receipts。补充 cadence audit 引用相同 result SHA。整文件本地镜像另外核验。

本轮没有模型 fits、预测质量、概率校准、long-horizon simulation 或新方法优势。只有一个发布日/验证日，不能证明多年接口完整性、跨领域机制或固定 fleet/curtailment/revision 口径。官方 publishTime 是发行标签，不是独立历史网络接收记录；足以构建声明清楚的 as-of 协议，不要求额外的生产部署保证。
