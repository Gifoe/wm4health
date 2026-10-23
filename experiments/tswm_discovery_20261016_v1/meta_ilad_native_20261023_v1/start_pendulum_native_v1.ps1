$ErrorActionPreference='Stop'
$r='D:\tswm_discovery_20261016_v1\meta_ilad_native_20261023_v1'
if(Test-Path -LiteralPath ($r+'\pendulum_native_execution_v1')){throw 'Existing data job; inspect actual PID and receipt.'}
$body=@'
$r='D:\tswm_discovery_20261016_v1\meta_ilad_native_20261023_v1'
$env:CONDA_PREFIX='E:\Anaconda\envs\persist_stable_251'
$env:PATH="$env:CONDA_PREFIX;$env:CONDA_PREFIX\Scripts;$env:CONDA_PREFIX\Library\bin;$env:SystemRoot\System32;$env:SystemRoot"
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe' -I -B -X utf8 -u ($r+'\scripts\capture_pendulum_native_v1.py')
exit $LASTEXITCODE
'@
$enc=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$j=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc);CurrentDirectory=$r;ProcessStartupInformation=$startup}
$j | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($j.ReturnValue -ne 0){throw 'Data job launch failed.'}
