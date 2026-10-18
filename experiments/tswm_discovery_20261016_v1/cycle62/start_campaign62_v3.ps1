$ErrorActionPreference='Stop'
$root62=$PSScriptRoot
$py62=Join-Path (Split-Path -Parent $root62) '.venv\Scripts\python.exe'
if(Test-Path -LiteralPath "$root62\outputs\campaign_state62_v3.json"){throw 'Qualified loader campaign already attempted; inspect state.'}
$live62=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle62*'}
if($live62){throw 'C62 live process exists.'}
$command62='"'+$py62+'" -B -u "'+$root62+'\scripts\capture_campaign62_v3.py"'
$result62=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command62;CurrentDirectory=$root62}
if($result62.ReturnValue -ne 0){throw 'Process creation failed.'}
@{utc=[DateTime]::UtcNow.ToString('o');stub_pid=$result62.ProcessId;status='QUALIFIED_LOADER_FIVE_MISSING_FITS'} | ConvertTo-Json -Compress
