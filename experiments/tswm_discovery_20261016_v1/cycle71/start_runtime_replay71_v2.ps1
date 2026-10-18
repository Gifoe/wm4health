$ErrorActionPreference='Stop'
$out71='D:\tswm_discovery_20261016_v1\cycle71\outputs'
if(Test-Path -LiteralPath "$out71\runtime_replay71_v1_unrestricted_launch.json"){throw 'Existing replay launch; inspect state'}
$live71=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*scripts\replay_runtime71_v1.py*' -or $_.CommandLine -like '*scripts\capture_runtime_replay71_v1.py*'})
if($live71.Count -ne 0){throw 'A replay worker is already live'}
$body71=@'
$rt71='E:\Anaconda\envs\salt_env'
$env:CONDA_PREFIX=$rt71
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:CUDA_LAUNCH_BLOCKING='1'
$env:PATH="$rt71;$rt71\Scripts;$rt71\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rt71\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle71\scripts\capture_runtime_replay71_v1.py' *> 'D:\tswm_discovery_20261016_v1\cycle71\outputs\runtime_replay_capture71_v1_stdout.log'
exit $LASTEXITCODE
'@
$enc71=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body71))
$r71=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc71);CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle71'}
$r71|ConvertTo-Json -Compress
if($r71.ReturnValue -ne 0){throw 'Bounded replay launch failed'}
