$ErrorActionPreference='Stop'
$root64=$PSScriptRoot
$py64=Join-Path (Split-Path -Parent $root64) '.venv\Scripts\python.exe'
if(Test-Path -LiteralPath "$root64\outputs\sampler_launch64_v1.json"){throw 'Sampler job already attempted; inspect actual state.'}
$command64='"'+$py64+'" -B -u "'+$root64+'\scripts\capture_sampler64_v1.py"'
$result64=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command64;CurrentDirectory=$root64}
if($result64.ReturnValue -ne 0){throw 'Process creation failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');stub_pid=$result64.ProcessId;status='FIXED_SAMPLER_CONTROLS_LAUNCHED'} | ConvertTo-Json -Compress
