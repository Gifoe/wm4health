$ErrorActionPreference='Stop'
$env:CUDA_LAUNCH_BLOCKING='1'
Start-Process -FilePath 'E:\Anaconda\envs\salt_env\python.exe' -ArgumentList @('-I','-B','-X','faulthandler','-u','D:\tswm_discovery_20261016_v1\cycle71\scripts\capture_runtime_replay71_v1.py') -WorkingDirectory 'D:\tswm_discovery_20261016_v1\cycle71' -WindowStyle Hidden
