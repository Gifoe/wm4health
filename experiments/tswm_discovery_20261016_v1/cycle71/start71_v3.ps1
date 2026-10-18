param([ValidateSet('qualification','campaign')][string]$Mode71)
$ErrorActionPreference='Stop'
$out71='D:\tswm_discovery_20261016_v1\cycle71\outputs'
if(Test-Path -LiteralPath "$out71\${Mode71}71_v3_launch.json"){throw 'Existing launch; inspect terminal/live state'}
$live71=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle71*'})
if($live71.Count -ne 0){throw 'C71 worker live'}
$body71=@'
$rt71='E:\Anaconda\envs\salt_env'
$env:CONDA_PREFIX=$rt71
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:PATH="$rt71;$rt71\Scripts;$rt71\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rt71\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle71\scripts\capture71_v3.py' '__MODE71__'
exit $LASTEXITCODE
'@
$body71=$body71.Replace('__MODE71__',$Mode71)
$enc71=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body71))
$launch71=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc71);CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle71'}
$launch71 | ConvertTo-Json -Compress
if($launch71.ReturnValue -ne 0){throw 'C71 launch failed'}
