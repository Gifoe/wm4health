$ErrorActionPreference='Stop'
$root63=$PSScriptRoot
$py63=Join-Path (Split-Path -Parent $root63) '.venv\Scripts\python.exe'
if(Test-Path -LiteralPath "$root63\outputs\audit_launch63_v1.json"){throw 'Audit already launched; inspect live process and evidence.'}
$command63='"'+$py63+'" -B -u "'+$root63+'\scripts\capture63_v1.py"'
$result63=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command63;CurrentDirectory=$root63}
if($result63.ReturnValue -ne 0){throw 'Process creation failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');stub_pid=$result63.ProcessId;status='NATIVE_PROBABILITY_CONTRACT_AUDIT_LAUNCHED'} | ConvertTo-Json -Compress
