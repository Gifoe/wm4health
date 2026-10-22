$ErrorActionPreference='Stop'
$root3='D:\tswm_discovery_20261016_v1\action_order_probe_20261023_v1'
if(Test-Path -LiteralPath ($root3+'\execution')){throw 'Execution directory exists; inspect actual handle and receipt.'}
if(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*run_order_probe*'}){throw 'An order probe is live.'}
$body3=@'
$rt3='E:\Anaconda\envs\persist_stable_251'
$env:CONDA_PREFIX=$rt3
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:PATH="$rt3;$rt3\Scripts;$rt3\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& 'D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe' -I -B -X faulthandler -u 'D:\tswm_discovery_20261016_v1\action_order_probe_20261023_v1\scripts\capture_order_probe_v1.py'
exit $LASTEXITCODE
'@
$enc3=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body3))
$startup3=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$r3=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc3);CurrentDirectory=$root3;ProcessStartupInformation=$startup3}
$r3 | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($r3.ReturnValue -ne 0){throw 'Order probe launch failed.'}
