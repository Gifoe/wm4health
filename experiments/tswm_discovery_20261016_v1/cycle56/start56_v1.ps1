$ErrorActionPreference='Stop'
$root56=$PSScriptRoot
$project56=Split-Path -Parent $root56
$python56=Join-Path $project56 'cycle2\.venv_rpm\Scripts\python.exe'
if(-not (Test-Path -LiteralPath $python56)){throw 'Existing isolated CPU runtime required'}
$existing56=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle56*' -and $_.CommandLine -like '*capture56*'}
if($existing56){throw 'Existing C56 owner: inspect it instead of restarting'}
$command56='"'+$python56+'" -B -u "'+$root56+'\scripts\capture56_v1.py"'
$launch56=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command56;CurrentDirectory=$root56}
if($launch56.ReturnValue -ne 0){throw 'C56 launch failed'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_FINITE_REFERENCE_GRADIENT_AUDIT';stub_process_id=$launch56.ProcessId;new_training_updates=0} | ConvertTo-Json -Compress
