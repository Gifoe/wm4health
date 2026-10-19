# 操作记录

第一次 acquire_native_loss.py 使用 urllib 默认网络设置，下载前报 ConnectionRefusedError WinError10061；没有进入科学诊断，没有 source mismatch 或数值失败。服务器此前已有环境代理失效的情况。

保留原脚本。改用独立 acquire_native_loss_direct.py 的 process-local ProxyHandler({})，对同一个公开 GitHub pinned URL 使用正常证书校验/TLS；不改服务器代理或环境设置，不关闭验证。科学计划/脚本/容差均不改，结果单独记录。

直接连接 raw.githubusercontent.com 在服务器同样 ConnectionRefused，本地直接 raw 请求也以 RemoteDisconnected 结束；没有科学运行产生失败结果。因此不能认定默认代理是唯一原因。随后通过正常 TLS 的公开 GitHub Git blob API 获取同一 pinned tree 的三个文件，Git blob、SHA256、与已有源码规范化文本均核对，再用 SFTP 整文件 SHA 上传服务器。全部数值诊断和独立 verifier 在服务器执行，未关闭证书验证、未改系统网络。
