$ErrorActionPreference='Stop'
$out70='D:\tswm_discovery_20261016_v1\cycle70\outputs'
New-Item -ItemType Directory -Path $out70 -Force|Out-Null
if(Test-Path -LiteralPath "$out70\replay70_v1_launch.json"){throw 'Existing qualification launch; inspect it'}
$live70=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle70*'})
if($live70.Count -ne 0){throw 'C70 process already live'}
$body70=@'
$base70='E:\Anaconda\envs\persist_stable_251'
$rt70='D:\tswm_discovery_20261016_v1\cycle2\.venv_rpm'
$env:CONDA_PREFIX=$base70
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:PATH="$rt70\Scripts;$base70;$base70\Scripts;$base70\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rt70\Scripts\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle70\scripts\capture_replay70_v1.py'
exit $LASTEXITCODE
'@
$enc70=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body70))
$launch70=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc70);CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle70'}
$launch70|ConvertTo-Json -Compress
if($launch70.ReturnValue -ne 0){throw 'C70 launch failed'}
