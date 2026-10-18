param([ValidateSet('qualify','formal')][string]$Mode='qualify')
$ErrorActionPreference='Stop'
$out71='D:\tswm_discovery_20261016_v1\cycle71\outputs'
if(Test-Path -LiteralPath "$out71\bounded_campaign71_v1_${Mode}_launch.json"){throw 'Existing launch; inspect actual state'}
if($Mode -eq 'formal'){
 $q71=Get-Content -LiteralPath "$out71\BOUNDED_CONTINUATION_QUALIFICATION71_V1.json" -Raw | ConvertFrom-Json
 if($q71.status -ne 'PASS_BOUNDED_PROCESS_CONTINUATION_EXACT_EQUIVALENCE'){throw 'Missing bounded-process qualification'}
}
$live71=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*scripts\bounded_native71_v1.py*' -or $_.CommandLine -like '*scripts\bounded_campaign71_v1.py*' -or $_.CommandLine -like '*scripts\replay_runtime71_v*.py*'})
if($live71.Count -ne 0){throw 'A native or replay worker is live'}
$body71=@'
$rt71='E:\Anaconda\envs\salt_env'
$env:CONDA_PREFIX=$rt71
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:CUDA_LAUNCH_BLOCKING='1'
$env:PATH="$rt71;$rt71\Scripts;$rt71\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rt71\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle71\scripts\capture_bounded_campaign71_v1.py' __MODE__
exit $LASTEXITCODE
'@
$body71=$body71.Replace('__MODE__',$Mode)
$enc71=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body71))
$r71=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc71);CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle71'}
$r71|ConvertTo-Json -Compress
if($r71.ReturnValue -ne 0){throw 'Bounded native campaign launch failed'}
