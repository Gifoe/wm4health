$ErrorActionPreference='Stop'
$root61=$PSScriptRoot
$py61=Join-Path (Split-Path -Parent $root61) '.venv\Scripts\python.exe'
if(Test-Path -LiteralPath "$root61\outputs\preflight_state61_v2.json"){throw 'Preflight already attempted; inspect retained evidence.'}
$existing61=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle61*'}
if($existing61){throw 'An existing C61 process must be inspected.'}
$command61='"'+$py61+'" -B -u "'+$root61+'\scripts\capture_preflight61_v2.py"'
$result61=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command61;CurrentDirectory=$root61}
if($result61.ReturnValue -ne 0){throw 'Process creation failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');stub_pid=$result61.ProcessId;status='NATIVE_QUALIFICATION_LAUNCHED'} | ConvertTo-Json -Compress
