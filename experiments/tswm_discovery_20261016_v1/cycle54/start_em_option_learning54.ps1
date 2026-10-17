$ErrorActionPreference='Stop'
$root='D:\tswm_discovery_20261016_v1\cycle54'
if(Test-Path -LiteralPath "$root\outputs\em_option_learning_v1\queue.stdout.log"){throw 'Existing EM option queue; inspect'}
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$launch=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"D:\tswm_discovery_20261016_v1\cycle2\.venv_rpm\Scripts\python.exe" -B -u "D:\tswm_discovery_20261016_v1\cycle54\scripts\queue_em_option54_v1.py"';CurrentDirectory=$root;ProcessStartupInformation=$startup}
$launch | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($launch.ReturnValue -ne 0){throw 'EM option learning queue launch failed'}
