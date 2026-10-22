$ErrorActionPreference='Stop'
$root3='D:\tswm_discovery_20261016_v1\three_baseline_failures_20261023_v1'
if(Test-Path -LiteralPath ($root3+'\outputs\dense_hourly_v1_launch.json')){throw 'Dense launch exists; inspect its handle.'}
if(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*three_baseline_failures_20261023_v1*'}){throw 'A campaign worker is still live.'}
$j3=Get-Content -LiteralPath ($root3+'\outputs\native_replays_v3\RESULTS_V3.json') -Raw | ConvertFrom-Json
if($j3.status -ne 'ALL_SIX_NATIVE_REPLAY_ARTIFACTS_VERIFIED'){throw 'Incomplete native replay matrix.'}
$body3=@'
$rt3='E:\Anaconda\envs\persist_stable_251'
$env:CONDA_PREFIX=$rt3
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:PATH="$rt3;$rt3\Scripts;$rt3\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& 'D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe' -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\three_baseline_failures_20261023_v1\scripts\capture_dense_hourly_v1.py'
exit $LASTEXITCODE
'@
$enc3=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body3))
$startup3=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$r3=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc3);CurrentDirectory=$root3;ProcessStartupInformation=$startup3}
$r3 | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($r3.ReturnValue -ne 0){throw 'Dense audit launch failed.'}
