$ErrorActionPreference='Stop'
$root62=$PSScriptRoot
$py62=Join-Path (Split-Path -Parent $root62) '.venv\Scripts\python.exe'
if(Test-Path -LiteralPath "$root62\outputs\post_state62_v3.json"){throw 'Post already attempted; inspect state.'}
$command62='"'+$py62+'" -B -u "'+$root62+'\scripts\capture_post62_v3.py"'
$result62=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command62;CurrentDirectory=$root62}
if($result62.ReturnValue -ne 0){throw 'Process creation failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');stub_pid=$result62.ProcessId;status='POST_PROCESS_WAITS_FOR_ALL_TRAINING_BEFORE_TEST'} | ConvertTo-Json -Compress
