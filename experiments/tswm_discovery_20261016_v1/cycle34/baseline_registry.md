# C34 模型、数据与执行登记

| 对象 | 固定来源与状态 | 实际边界 |
|---|---|---|
| Sundial 128M | HF thuml/sundial-base-128m，revision3212e42564493f520593e5414af4367fc4b49226；128329680参数；513341448B权重 | 源码/API执行资格检查，非原论文性能复现、非新模型 |
| FEV solar_with_weather/15T | autogluon/fev_datasets revision f71c0fff4cf81283a2c43e7f3a73aa4f9826aef8；2301980B | 只选 target 的最早连续有限672点；未用未来真值/天气 |
| FEV jena_weather/10T | 同revision；2342073B；预列 target_0 | 原字段物理单位不猜测；只用最早连续有限672点 |
| PathFM | 论文可读，作者可运行发布未确认 | 没有原生运行或复现 |
| HEPA | 来源与复现说明审计 | 未运行；实现/数据差异不是架构缺口 |

权重与两份raw parquet仅在服务器。全部下载经正常TLS、官方元数据/SHA验证。依赖隔离在cycle34/vendor：transformers4.40.1、tokenizers0.19.1、HFhub0.23.5、regex2024.5.15；不修改既有环境。源码直接导入已读文件，不执行trust_remote_code。

完整URL、文件SHA、依赖wheel、版本与下载receipt在outputs/asset_setup_v1/manifest.json和upstream_source/Sundial/source_manifest.json。权重SHA256414435b508391f92afadd2aaeec418c806776aeccbce12e638d73a139ca5ca78。
