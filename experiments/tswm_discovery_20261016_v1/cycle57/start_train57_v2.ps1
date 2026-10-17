$ErrorActionPreference='Stop'
$root57=$PSScriptRoot
$project57=Split-Path -Parent $root57
$python57=Join-Path $project57 '.venv\Scripts\python.exe'
$existing57=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle57*' -and $_.CommandLine -like '*capture_train*'}
if($existing57){throw 'C57 training owner exists; do not restart'}
$command57='"'+$python57+'" -B -u "'+$root57+'\scripts\capture_train57_v2.py"'
$launch57=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command57;CurrentDirectory=$root57}
if($launch57.ReturnValue -ne 0){throw 'C57 V2 training launch failed'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_CENTERED_CONTROL_TRAINING';stub_process_id=$launch57.ProcessId;heldout_opened=$false} | ConvertTo-Json -Compress
