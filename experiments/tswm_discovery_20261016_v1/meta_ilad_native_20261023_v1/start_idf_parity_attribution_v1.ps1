$ErrorActionPreference='Stop'
$r='D:\tswm_discovery_20261016_v1\meta_ilad_native_20261023_v1'
if(Test-Path -LiteralPath ($r+'\inference_parity_v1')){throw 'Parity diagnostic exists; inspect actual state before restarting.'}
$body=@'
$rt='E:\Anaconda\envs\persist_stable_251'
$env:CONDA_PREFIX=$rt
$env:PATH="$rt;$rt\Scripts;$rt\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
$env:PYTHONPATH=''
$env:PYTHONNOUSERSITE='1'
& 'D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe' -I -B -X utf8 -u 'D:\tswm_discovery_20261016_v1\meta_ilad_native_20261023_v1\scripts\run_idf_parity_attribution_v1.py'
exit $LASTEXITCODE
'@
$enc=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body));$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$j=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc);CurrentDirectory=$r;ProcessStartupInformation=$startup}
$j | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($j.ReturnValue -ne 0){throw 'Parity diagnostic launch failed.'}
