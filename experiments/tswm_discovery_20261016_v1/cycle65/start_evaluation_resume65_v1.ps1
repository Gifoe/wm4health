$ErrorActionPreference='Stop'
$out65='D:\tswm_discovery_20261016_v1\cycle65\outputs'
if(Test-Path -LiteralPath "$out65\EVALUATION_RESUME65_V1.json"){throw 'Existing resume; inspect actual process.'}
$live65=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle65*'})
if($live65.Count -ne 0){throw 'C65 process remains live'}
$body65=@'
$rtenv65='E:\Anaconda\envs\persist_stable_251'
$env:CONDA_PREFIX=$rtenv65
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:PATH="$rtenv65;$rtenv65\Scripts;$rtenv65\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rtenv65\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle65\scripts\isolated_entry65_v1.py' 'capture_evaluation_resume65_v1.py'
exit $LASTEXITCODE
'@
$enc65=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body65))
$launch65=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc65);CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle65'}
$launch65 | ConvertTo-Json -Compress
if($launch65.ReturnValue -ne 0){throw 'WMI resume launch failed'}
