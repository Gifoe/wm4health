$ErrorActionPreference='Stop'
$root57=$PSScriptRoot
$project57=Split-Path -Parent $root57
$python57=Join-Path $project57 '.venv\Scripts\python.exe'
if(-not (Test-Path -LiteralPath $python57)){throw 'Existing CUDA runtime required'}
$existing57=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle57*' -and $_.CommandLine -like '*capture*'}
if($existing57){throw 'C57 owner exists; inspect before launching'}
$command57='"'+$python57+'" -B -u "'+$root57+'\scripts\capture_preflight57_v1.py"'
$launch57=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command57;CurrentDirectory=$root57}
if($launch57.ReturnValue -ne 0){throw 'C57 preflight launch failed'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_NATIVE_CUDA_QUALIFICATION';stub_process_id=$launch57.ProcessId;formal_updates=0} | ConvertTo-Json -Compress
