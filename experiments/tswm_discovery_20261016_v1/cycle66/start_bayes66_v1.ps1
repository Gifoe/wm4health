$ErrorActionPreference='Stop'
$out66='D:\tswm_discovery_20261016_v1\cycle66\outputs'
if(Test-Path -LiteralPath "$out66\bayes_launch66_v1.json"){throw 'Existing native interface launch'}
$live66=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle66*'})
if($live66.Count -ne 0){throw 'Native process remains live'}
$body66=@'
$rt66='E:\Anaconda\envs\salt_env'
$env:CONDA_PREFIX=$rt66
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:PATH="$rt66;$rt66\Scripts;$rt66\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rt66\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle66\scripts\capture_bayes66_v1.py'
exit $LASTEXITCODE
'@
$enc66=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body66))
$launch66=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc66);CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle66'}
$launch66 | ConvertTo-Json -Compress
if($launch66.ReturnValue -ne 0){throw 'WMI native interface launch failed'}
