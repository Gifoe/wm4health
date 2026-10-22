$ErrorActionPreference='Stop'
$root71='D:\tswm_discovery_20261016_v1\cycle71'
$out71=Join-Path $root71 'outputs'
if(Test-Path -LiteralPath (Join-Path $out71 'free_simulation71_recovery_20261023_v1_launch.json')){throw 'Scientific child already has a launch receipt; inspect its handle and exit.'}
if(Test-Path -LiteralPath (Join-Path $out71 'ASSESSMENT_EXECUTION_RECOVERY71_20261023_V1.json')){throw 'Recovery passed startup before; inspect artifacts before resuming.'}
$active71=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle71*'})
if($active71.Count -ne 0){throw 'A C71 Python process is live; do not duplicate.'}
$body71=@'
$rt71='E:\Anaconda\envs\salt_env'
$env:CONDA_PREFIX=$rt71
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:CUDA_LAUNCH_BLOCKING='1'
$env:PATH="$rt71;$rt71\Scripts;$rt71\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rt71\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle71\scripts\capture_assessment71_recovery_20261023_v1.py'
exit $LASTEXITCODE
'@
$enc71=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body71))
$startup71=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$p71=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc71);CurrentDirectory=$root71;ProcessStartupInformation=$startup71}
if($p71.ReturnValue -ne 0){throw 'CIM assessment launch failed.'}
$record71=@{pid=$p71.ProcessId;return_value=$p71.ReturnValue;utc=[DateTime]::UtcNow.ToString('o');transport='previously qualified CIM isolated-runtime launcher';prior_attempt='Start-Process child handle absent, no scientific outputs'}
$record71 | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out71 'assessment_capture71_recovery_20261023_v2_launcher.json') -Encoding UTF8
$record71 | ConvertTo-Json -Compress
