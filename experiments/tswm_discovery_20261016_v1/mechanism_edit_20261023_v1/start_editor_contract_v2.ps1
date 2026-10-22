$ErrorActionPreference='Stop'
$r='D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1'
if(Test-Path -LiteralPath ($r+'\editor_contract_execution_recovery_v1')){throw 'Editor execution exists; inspect its PID and receipt.'}
$j=Get-Content -LiteralPath ($r+'\runtime_setup_v1\RUNTIME_RECEIPT_V1.json') -Raw | ConvertFrom-Json
if($j.status -ne 'ISOLATED_JAX_CPU_READY'){throw 'CPU runtime not qualified.'}
$old=Get-Content -LiteralPath ($r+'\editor_contract_execution_v1\exit.json') -Raw | ConvertFrom-Json
if($old.exit_code -ne 1){throw 'Expected preserved incomplete preflight receipt.'}
$body=@'
$rt='E:\Anaconda\envs\persist_stable_251'
$env:CONDA_PREFIX=$rt
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:JAX_PLATFORMS='cpu'
$env:PATH="$rt;$rt\Scripts;$rt\Library\bin;$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
& 'D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\.venv_editor_cpu\Scripts\python.exe' -I -B -u 'D:\tswm_discovery_20261016_v1\mechanism_edit_20261023_v1\scripts\capture_editor_contract_v2.py'
exit $LASTEXITCODE
'@
$enc=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$result=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc);CurrentDirectory=$r;ProcessStartupInformation=$startup}
$result | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($result.ReturnValue -ne 0){throw 'Editor contract launch failed.'}
