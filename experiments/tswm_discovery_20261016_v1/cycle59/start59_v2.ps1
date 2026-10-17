$ErrorActionPreference='Stop'
$cycleRoot59=$PSScriptRoot
$pythonRoot59=Join-Path (Split-Path -Parent $cycleRoot59) '.venv\Scripts\python.exe'
$existing59=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle59*'}
if($existing59){throw 'A C59 process is live; inspect it instead of launching.'}
if(Test-Path -LiteralPath "$cycleRoot59\outputs\state59_v1.json"){throw 'Prior execution state exists; inspect before any restart.'}
if(Test-Path -LiteralPath "$cycleRoot59\outputs\launcher59_v2.json"){throw 'V2 launch receipt already exists.'}
$cmd59='"'+$pythonRoot59+'" -B -u "'+$cycleRoot59+'\scripts\capture59_v1.py"'
$launch59=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd59;CurrentDirectory=$cycleRoot59}
if($launch59.ReturnValue -ne 0){throw 'Independent CIM launch failed'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_CIM_OWNER_AFTER_VERIFIED_ABSENT_SSH_JOB';stub_process_id=$launch59.ProcessId;previous_start_process_pid=24924;prior_owner_verified_absent=$true;prior_execution_state_absent=$true} | ConvertTo-Json | Set-Content -LiteralPath "$cycleRoot59\outputs\launcher59_v2.json" -Encoding UTF8
Get-Content -Raw -LiteralPath "$cycleRoot59\outputs\launcher59_v2.json"
