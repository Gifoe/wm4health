$ErrorActionPreference='Stop'
$root60=$PSScriptRoot
$python60=Join-Path (Split-Path -Parent $root60) '.venv\Scripts\python.exe'
$live60=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle60*'}
if($live60){throw 'C60 owner exists; inspect it instead of launching.'}
if(Test-Path -LiteralPath "$root60\outputs\preflight_state60_v2.json"){throw 'Prior C60 preflight state exists.'}
$cmd60='"'+$python60+'" -B -u "'+$root60+'\scripts\capture_preflight60_v2.py"'
$launched60=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd60;CurrentDirectory=$root60}
if($launched60.ReturnValue -ne 0){throw 'C60 CIM launch failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_C60_PREFLIGHT_V2_AFTER_VERIFIED_TERMINAL_PATH_COLLISION';stub_process_id=$launched60.ProcessId} | ConvertTo-Json -Compress
