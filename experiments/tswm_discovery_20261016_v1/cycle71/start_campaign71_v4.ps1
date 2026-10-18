$Mode='campaign'
$ErrorActionPreference='Stop'
$out71='D:\tswm_discovery_20261016_v1\cycle71\outputs'
if(Test-Path -LiteralPath "$out71\campaign71_v4_launch.json"){throw 'Existing recovery launch; inspect state'}
if($Mode -eq 'formal'){
 $q71=Get-Content -LiteralPath "$out71\RESUME_QUALIFICATION71_V2.json" -Raw | ConvertFrom-Json
 if($q71.status -ne 'PASS_CHECKPOINT_CONTINUATION_EQUIVALENCE'){throw 'Recovery qualification is missing'}
}
$live71=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*scripts\train71_v3.py*' -or $_.CommandLine -like '*scripts\resume_native71_v2.py*'})
if($live71.Count -ne 0){throw 'A training/recovery worker is already live'}
$body71=@'
$rt71='E:\Anaconda\envs\salt_env'
$env:CONDA_PREFIX=$rt71
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:CUDA_LAUNCH_BLOCKING='1'
$env:PATH="$rt71;$rt71\Scripts;$rt71\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& "$rt71\python.exe" -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\cycle71\scripts\capture_campaign71_v4.py'
exit $LASTEXITCODE
'@
$body71=$body71.Replace('__MODE__',$Mode)
$enc71=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body71))
$r71=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc71);CurrentDirectory='D:\tswm_discovery_20261016_v1\cycle71'}
$r71|ConvertTo-Json -Compress
if($r71.ReturnValue -ne 0){throw 'Recovery worker launch failed'}
