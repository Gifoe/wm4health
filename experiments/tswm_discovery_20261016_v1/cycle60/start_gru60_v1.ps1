$ErrorActionPreference='Stop'
$rootGru60=$PSScriptRoot
$pythonGru60=Join-Path (Split-Path -Parent $rootGru60) '.venv\Scripts\python.exe'
$liveGru60=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle60*' -and $_.CommandLine -like '*gru60*'}
if($liveGru60){throw 'A C60 GRU process is live; inspect it.'}
if(Test-Path -LiteralPath "$rootGru60\outputs\gru_training_state60_v1.json"){throw 'Prior C60 GRU training state exists.'}
$cmdGru60='"'+$pythonGru60+'" -B -u "'+$rootGru60+'\scripts\capture_gru60_v1.py"'
$launchGru60=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmdGru60;CurrentDirectory=$rootGru60}
if($launchGru60.ReturnValue -ne 0){throw 'C60 GRU launch failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_SIX_ORDINARY_GRU_FITS';stub_process_id=$launchGru60.ProcessId} | ConvertTo-Json -Compress
