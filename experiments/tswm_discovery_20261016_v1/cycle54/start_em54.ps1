$ErrorActionPreference='Stop'
$root='D:\tswm_discovery_20261016_v1\cycle54'
if(Test-Path -LiteralPath "$root\outputs\em_queue54_v1.stdout.log"){throw 'Preserve existing EM queue logs'}
if(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'train54.py em|train_em54_affinity|run_em54'}){throw 'Existing EM worker; inspect'}
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$launch=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"D:\tswm_discovery_20261016_v1\cycle2\.venv_rpm\Scripts\python.exe" -B -u "D:\tswm_discovery_20261016_v1\cycle54\scripts\run_em54_v1.py"';CurrentDirectory=$root;ProcessStartupInformation=$startup}
$launch | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($launch.ReturnValue -ne 0){throw 'EM queue launch failed'}
