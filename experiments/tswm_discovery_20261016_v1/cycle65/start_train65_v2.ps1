$ErrorActionPreference='Stop'
$out65='D:\tswm_discovery_20261016_v1\cycle65\outputs'
if(Test-Path -LiteralPath "$out65\train_launch65_v1.json"){throw 'Existing launch; inspect its actual process instead of restarting.'}
$command65='"D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe" -B -u "D:\tswm_discovery_20261016_v1\cycle65\scripts\capture_train65_v2.py"'
$launch65=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command65;CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle65'}
$launch65 | ConvertTo-Json -Compress
if($launch65.ReturnValue -ne 0){throw 'WMI launch failed'}
