$ErrorActionPreference='Stop'
$rootTrain60=$PSScriptRoot
$pythonTrain60=Join-Path (Split-Path -Parent $rootTrain60) '.venv\Scripts\python.exe'
$liveTrain60=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle60*'}
if($liveTrain60){throw 'A C60 owner already exists; inspect it.'}
if(Test-Path -LiteralPath "$rootTrain60\outputs\training_state60_v1.json"){throw 'A C60 training attempt already has state; do not restart.'}
$cmdTrain60='"'+$pythonTrain60+'" -B -u "'+$rootTrain60+'\scripts\capture_train60_v1.py"'
$launchTrain60=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmdTrain60;CurrentDirectory=$rootTrain60}
if($launchTrain60.ReturnValue -ne 0){throw 'C60 trainer launch failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_SIX_NATIVE_COMPONENT_NSSM_FITS';stub_process_id=$launchTrain60.ProcessId} | ConvertTo-Json -Compress
