$ErrorActionPreference='Stop'
$rootPost60=$PSScriptRoot
$pythonPost60=Join-Path (Split-Path -Parent $rootPost60) '.venv\Scripts\python.exe'
$livePost60=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle60*' -and $_.CommandLine -like '*capture_post*'}
if($livePost60){throw 'C60 scorer owner exists; inspect it.'}
if(Test-Path -LiteralPath "$rootPost60\outputs\post_state60_v1.json"){throw 'C60 postprocess state exists; do not restart.'}
$cmdPost60='"'+$pythonPost60+'" -B -u "'+$rootPost60+'\scripts\capture_post60_v1.py"'
$launchPost60=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmdPost60;CurrentDirectory=$rootPost60}
if($launchPost60.ReturnValue -ne 0){throw 'C60 scorer launch failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_C60_SEAL_WAITER';stub_process_id=$launchPost60.ProcessId} | ConvertTo-Json -Compress
