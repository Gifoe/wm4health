$ErrorActionPreference='Stop'
$r='D:\tswm_discovery_20261016_v1\meta_ilad_native_20261023_v1'
if(Test-Path -LiteralPath ($r+'\recipe_execution_v1')){throw 'Native recipe exists; inspect actual PID and receipts.'}
$body=@'
$r='D:\tswm_discovery_20261016_v1\meta_ilad_native_20261023_v1'
$ready=Get-Content -LiteralPath ($r+'\runtime_cpu_recovery_v1\READY_V1.json') -Raw | ConvertFrom-Json
if($ready.status -ne 'NATIVE_META_ILAD_CPU_RUNTIME_READY'){throw 'Qualified CPU runtime missing.'}
$py=$ready.python_path
$base=& $py -I -c 'import sys;print(sys.base_prefix)'
$venv=Split-Path -Parent $py
$env:PATH="$venv;$base;$base\DLLs;$env:SystemRoot\System32;$env:SystemRoot"
Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
Remove-Item Env:CONDA_DEFAULT_ENV -ErrorAction SilentlyContinue
$env:PYTHONPATH=''
$env:PYTHONNOUSERSITE='1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& $py -I -B -X utf8 -u ($r+'\scripts\capture_native_recipe_v1.py')
exit $LASTEXITCODE
'@
$enc=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$j=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc);CurrentDirectory=$r;ProcessStartupInformation=$startup}
$j | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($j.ReturnValue -ne 0){throw 'Native recipe launch failed.'}
