# 执行资格检查 v2：公开权重直接使用 core 参数名

v1 在 strict_load 阶段停止：runner 假定 safetensors 的 277 个参数名带 `normwear.` 外层前缀。随后仅用 safe_open 读取实际 key，确认发布文件直接保存 `encoder_blocks.*`、`decoder_blocks.*`、`patch_embed.*` 等核心模型参数名。

v2 改为先检查发布权重 key 集合与官方 core 模型的完整 key 集合严格相等，再 `load_state_dict(strict=True)`。没有放宽缺失参数、形状或数值检查，没有修改发布源码/权重，也没有安装新依赖。

v1 脚本与 outputs/normwear_qualification_v1/failure.json 保留；v2 使用独立 outputs/normwear_qualification_v2。其余资格与接口检查计划不变。原始失败属于 runner 加载命名假设错误，不是科学负结果，也不是 NormWear 性能缺陷。
