$ErrorActionPreference='Stop'
$out65='D:\tswm_discovery_20261016_v1\cycle65\outputs'
if(Test-Path -LiteralPath "$out65\recovery_launch65_v1.json"){throw 'Existing recovery launch; inspect actual process.'}
$live65=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle65*'})
if($live65.Count -ne 0){throw 'Earlier experiment process still live; do not duplicate.'}
$command65='"D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe" -B -X faulthandler -u "D:\tswm_discovery_20261016_v1\cycle65\scripts\capture_recovery65_v1.py"'
$launch65=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command65;CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle65'}
$launch65 | ConvertTo-Json -Compress
if($launch65.ReturnValue -ne 0){throw 'WMI recovery launch failed'}
