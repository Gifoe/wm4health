$ErrorActionPreference='Stop'
$root61=$PSScriptRoot
$py61=Join-Path (Split-Path -Parent $root61) '.venv\Scripts\python.exe'
if(Test-Path -LiteralPath "$root61\outputs\post_state61_v1.json"){throw 'Postprocess already attempted; inspect it.'}
$existing61=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*capture_post61_v1.py*'}
if($existing61){throw 'An existing postprocess owner must be inspected.'}
$command61='"'+$py61+'" -B -u "'+$root61+'\scripts\capture_post61_v1.py"'
$result61=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command61;CurrentDirectory=$root61}
if($result61.ReturnValue -ne 0){throw 'Process creation failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');stub_pid=$result61.ProcessId;status='VERIFIED_OWNER_WAITER_LAUNCHED'} | ConvertTo-Json -Compress
