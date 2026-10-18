$ErrorActionPreference='Stop'
$out66='D:\tswm_discovery_20261016_v1\cycle66\outputs'
if(Test-Path -LiteralPath "$out66\stage_launch66_v1.json"){throw 'Existing download launch; inspect actual process.'}
$command66='"D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe" -B -u "D:\tswm_discovery_20261016_v1\cycle66\scripts\capture_stage66_v1.py"'
$launch66=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command66;CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle66'}
$launch66 | ConvertTo-Json -Compress
if($launch66.ReturnValue -ne 0){throw 'WMI model staging launch failed'}
