$ErrorActionPreference='Stop'
$r='D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1'
if(Test-Path -LiteralPath ($r+'\native_core_profile_execution_recovery_v1')){throw 'Native core profile exists; inspect live PID and terminal receipt.'}
$j=Get-Content -LiteralPath ($r+'\runtime_model_core_v1\READY_V1.json') -Raw | ConvertFrom-Json
if($j.status -ne 'NATIVE_MODEL_CORE_CPU_DEPENDENCIES_READY'){throw 'Native core dependencies not ready.'}
$old=Get-Content -LiteralPath ($r+'\native_core_profile_execution_v1\exit.json') -Raw | ConvertFrom-Json
if($old.exit_code -ne 1 -or $old.timeout){throw 'Expected preserved metric-domain failure receipt.'}
$body=@'
$rt='E:\Anaconda\envs\persist_stable_251'
$env:CONDA_PREFIX=$rt
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:JAX_PLATFORMS='cpu'
$env:PATH="$rt;$rt\Scripts;$rt\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& 'D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\.venv_editor_cpu\Scripts\python.exe' -I -B -u 'D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\scripts\capture_native_core_profile_v3.py'
exit $LASTEXITCODE
'@
$enc=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$result=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc);CurrentDirectory=$r;ProcessStartupInformation=$startup}
$result | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($result.ReturnValue -ne 0){throw 'Core profile launch failed.'}
