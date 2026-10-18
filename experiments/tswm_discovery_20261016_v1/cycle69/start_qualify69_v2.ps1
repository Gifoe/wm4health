$ErrorActionPreference='Stop'
$out69='D:\tswm_discovery_20261016_v1\cycle69\outputs'
if(!(Test-Path -LiteralPath "$out69\DATA_SOURCE69_V2.json")){throw 'No verified checkpoint receipt'}
if(Test-Path -LiteralPath "$out69\qualification69_v2_launch.json"){throw 'Existing qualification pipeline launch'}
$live69=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle69*'})
if($live69.Count -ne 0){throw 'C69 process remains live'}
$body69=@'
$rt69='E:\Anaconda\envs\salt_env'
$env:CONDA_PREFIX=$rt69
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:PATH="$rt69;$rt69\Scripts;$rt69\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rt69\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle69\scripts\capture_qualify69_v2.py'
exit $LASTEXITCODE
'@
$enc69=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body69))
$launch69=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc69);CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle69'}
$launch69 | ConvertTo-Json -Compress
if($launch69.ReturnValue -ne 0){throw 'Qualification launch failed'}
