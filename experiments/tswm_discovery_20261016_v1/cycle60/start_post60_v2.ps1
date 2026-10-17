$ErrorActionPreference='Stop'
$rootPostV260=$PSScriptRoot
$pythonPostV260=Join-Path (Split-Path -Parent $rootPostV260) '.venv\Scripts\python.exe'
$livePostV260=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle60*' -and $_.CommandLine -like '*capture_post*'}
if($livePostV260){throw 'A C60 postprocess owner exists; inspect it.'}
if(Test-Path -LiteralPath "$rootPostV260\outputs\post_state60_v2.json"){throw 'C60 V2 postprocess state exists.'}
$cmdPostV260='"'+$pythonPostV260+'" -B -u "'+$rootPostV260+'\scripts\capture_post60_v2.py"'
$launchPostV260=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmdPostV260;CurrentDirectory=$rootPostV260}
if($launchPostV260.ReturnValue -ne 0){throw 'C60 V2 postprocess launch failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_C60_COMBINED_CAMPAIGN_SEAL_WAITER';stub_process_id=$launchPostV260.ProcessId} | ConvertTo-Json -Compress
