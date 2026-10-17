$ErrorActionPreference='Stop'
$root55=$PSScriptRoot
$project55=Split-Path -Parent $root55
$python55=Join-Path $project55 'cycle2\.venv_rpm\Scripts\python.exe'
if(-not (Test-Path -LiteralPath $python55)){throw 'C55 requires the existing isolated native runtime'}
$existing55=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle55*' -and $_.CommandLine -like '*capture55_v1.py*'}
if($existing55){throw 'The C55 capture already exists; inspect it instead of restarting'}
$command55='"'+$python55+'" -B -u "'+$root55+'\scripts\capture55_v1.py"'
$launch55=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command55;CurrentDirectory=$root55}
if($launch55.ReturnValue -ne 0){throw 'C55 launch failed'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_SCALAR_POPULATION_CONTROL';stub_process_id=$launch55.ProcessId;native_training_updates=0} | ConvertTo-Json -Compress
