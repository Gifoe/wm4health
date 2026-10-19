# 官方forecast vintage的小型服务器资格检查

检索发现Elexon BMRS官方schema明示publishTime和startTime，dataset endpoint按published range查历史。选定一次固定探针：WINDFOR发布日2026-01-01，FUELHH/WIND交付日2026-01-02。此选择在下载前固定；不训练、评估模型或选择有利日期，不是新框架实验。

使用官方 `https://data.elexon.co.uk/swagger/v1/swagger.json` 与 `https://data.elexon.co.uk/bmrs/api/v1/datasets/WINDFOR?publishDateTimeFrom=2026-01-01T00:00:00Z&publishDateTimeTo=2026-01-02T00:00:00Z&format=json`，以及FUELHH settlementDateFrom/To=2026-01-02、fuelType=WIND。

原始JSON在服务器按字节保存并SHA；记录URL、HTTP状态、UTC获取时间、参数schema、字段、行数、重复key、时间范围。正常TLS，不绕过验证。每次响应最多12MiB；只取这一发布日/交付日。

对Jan1 12:00/18:00及Jan2 00:00三个origin，按同一Jan2目标半小时时间轴选择latest publishTime≤origin且startTime>origin的forecast。记录coverage与所有选中key，不以future actual补缺。producer分组选择与独立Python逐行max选择必须一致。FUELHH只做对象/时间/availability资格，actual values不参与forecast选择，不计算forecast误差。不建立正式TRAIN/VAL/TEST或方法效用结论。

官方图表wind-generation使用generation-by-fuel-type outturn；两者均为ESO可见operational-metered风机，排除embedded/无operational meter。这里仅检查该公开合同与字段，不承诺零curtailment差异、fleet coverage永不变、settlement revisions或网络接收时间完全已知。publishTime是官方发行标签，区别于NWP初始化；不等于独立历史网络receipt。

如果覆盖不完整、endpoint失败或对象定义仍有疑问，原样报告；不虚构vintages、不改为/latest、不批量下载其它日期寻找通过案例。三探针不是统计独立系统，资格成功也不证明Research Gap。
