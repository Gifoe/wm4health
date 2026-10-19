# 执行记录

- cycle27为有效PROGRESS，整体研究仍未完成。本轮新增cycle28独立目录，未改旧权重/数据/环境，未公开上传。
- PCFM原生6源码文件逐个Gitblob/SHA核验；211MB权重直接在远端HF镜像经正常TLS取得并计算完整LFS SHA，未经不安全反序列化。
- 官方test_release.py主要用随机权重做API测试，本轮未执行它，也没有宣称其suite通过。实际执行的是完整原生example.main的5组demo，源码不改；外层仅记录forecast/what_if输入输出。所有5组原始图和stdout保留。
- 原生28次调用与真实CED22组预测完成且独立复核。原生28神经调用重放为新进程同源数值重放，不伪称独立重写网络；物理/数据/ARX/分数有独立代数实现。
- CED数据遵循nonlinear_benchmarks1.0.1：两条500点，前400训练、测试前10初始化、剩90全free forecast。原ICL notebook的CED注释写96 horizon，但本数据剩90；本轮按真实官方全剩余长度90调用native网络，明确是协议一致screen，不是原论文完整表格复现。
- ICL checkpoint已在cycle16缓存，受限weights_only+argparse.Namespace加载；未重下、未训练。future-y均零占位。native-padding30和direct10全部保留，未在测试选最佳。
- PCFM默认glue造成显著可修复误差；关掉的结果单列，不能称新架构。CP图只读允许历史，手工graph只给SISO工程方向与lag0，不给隐藏状态/真实机制系数。
- CED最初一次只读metadata命令受Windows原生命令引号处理影响，python-c报SyntaxError；改用保存的inspect_ced_data.py文件后成功。没有改变任何科学设置或重跑模型job。
- 未认证GitHubAPI在独立检索中发生shared-IP rate limit，停止重复API；公开raw/PyPI/HF镜像仍可读，无外部阻塞。VSF目前官方树仅README/LICENSE/gitignore；不把缺实现当Gap，也未启动重建。
- fetch/sync保留完整SHA；getall已排除.safetensors，避免把211MB原始权重当紧凑镜像反复传输。所有真实raw NPZ/权重留服务器；紧凑报告/数据来源/源码/图/复核镜像。
- 前向时长的计时边界在PCFM封装、ICL裸模型、ARX验证拟合之间不同，本轮不据此排名速度。两个预训练prior不同，不把参数量对比当控制了离线compute。
- 最终接受方向0。下一轮不继续调这两条CED或已解决toy；需要新的系统级能力残差及匹配强修复的证据，不能靠扩默认配置失败维持路线。
