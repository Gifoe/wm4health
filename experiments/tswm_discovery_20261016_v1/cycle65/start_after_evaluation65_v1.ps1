$ErrorActionPreference='Stop'
$out65='D:\tswm_discovery_20261016_v1\cycle65\outputs'
if(Test-Path -LiteralPath "$out65\after_evaluation_state65_v1.json"){throw 'Replay coordinator exists; inspect actual process.'}
$workers65=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*scripts\after_train65_v1.py*' -and $_.ExecutablePath -eq 'E:\Anaconda\envs\persist_stable_251\python.exe'})
if($workers65.Count -ne 1){throw 'Need one verified actual evaluation coordinator'}
$pid65=$workers65[0].ProcessId
$command65='"D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe" -B -u "D:\tswm_discovery_20261016_v1\cycle65\scripts\after_evaluation65_v1.py" '+$pid65
$launch65=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command65;CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle65'}
$launch65 | ConvertTo-Json -Compress
if($launch65.ReturnValue -ne 0){throw 'WMI replay dependency launch failed'}
