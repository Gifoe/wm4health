$ErrorActionPreference='Stop'
$root='D:\tswm_discovery_20261016_v1\cycle54'
if (Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'capture54|supervise54|train54|prepare54'}) {throw 'Existing C54 process; inspect before starting'}
if(Test-Path -LiteralPath "$root\outputs\supervisor54_v3.stdout.log"){throw 'Preserve prior capture logs'}
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$launch=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"D:\tswm_discovery_20261016_v1\cycle2\.venv_rpm\Scripts\python.exe" -B -u "D:\tswm_discovery_20261016_v1\cycle54\scripts\capture54_v3.py"';CurrentDirectory=$root;ProcessStartupInformation=$startup}
$launch | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($launch.ReturnValue -ne 0){throw 'Persistent WMI launch failed'}
