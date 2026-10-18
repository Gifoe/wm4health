$ErrorActionPreference='Stop'
$out68='D:\tswm_discovery_20261016_v1\cycle68\outputs'
New-Item -ItemType Directory -Path $out68 -Force | Out-Null
if(Test-Path -LiteralPath "$out68\stage68_v2_launch.json"){throw 'Existing stage launch'}
$live68=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle68*'})
if($live68.Count -ne 0){throw 'C68 process remains live'}
$body68=@'
$rt68='E:\Anaconda\envs\salt_env'
$env:CONDA_PREFIX=$rt68
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:PATH="$rt68;$rt68\Scripts;$rt68\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rt68\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle68\scripts\capture_stage68_v2.py'
exit $LASTEXITCODE
'@
$enc68=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body68))
$launch68=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc68);CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle68'}
$launch68 | ConvertTo-Json -Compress
if($launch68.ReturnValue -ne 0){throw 'C68 stage launch failed'}
