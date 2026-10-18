$ErrorActionPreference='Stop'
$out71='D:\tswm_discovery_20261016_v1\cycle71\outputs'
if(Test-Path -LiteralPath "$out71\open_loop71_v1_launch.json"){throw 'Existing rollout launch; inspect state'}
$live71=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*open_loop71_v1.py*'})
if($live71.Count -ne 0){throw 'Rollout qualifier already live'}
$body71=@'
$rt71='E:\Anaconda\envs\salt_env'
$env:CONDA_PREFIX=$rt71
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:PATH="$rt71;$rt71\Scripts;$rt71\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rt71\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle71\scripts\capture_open_loop71_v1.py'
exit $LASTEXITCODE
'@
$enc71=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body71))
$r71=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc71);CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle71'}
$r71|ConvertTo-Json -Compress
if($r71.ReturnValue -ne 0){throw 'Rollout qualifier launch failed'}
