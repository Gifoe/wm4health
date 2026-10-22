$ErrorActionPreference='Stop'
$r='D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1'
if(Test-Path -LiteralPath ($r+'\postfit_queue_recovery_v1')){throw 'Post-fit queue already exists; inspect its process and terminal receipts.'}
$body=@'
$rt='E:\Anaconda\envs\persist_stable_251'
$env:CONDA_PREFIX=$rt
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:PATH="$rt;$rt\Scripts;$rt\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& 'D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe' -I -B -u 'D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\scripts\postfit_queue_v2.py'
exit $LASTEXITCODE
'@
$enc=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$result=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc);CurrentDirectory=$r;ProcessStartupInformation=$startup}
$result | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($result.ReturnValue -ne 0){throw 'Post-fit queue launch failed.'}
