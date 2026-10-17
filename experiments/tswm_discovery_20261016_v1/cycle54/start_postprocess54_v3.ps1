$ErrorActionPreference='Stop'
$root='D:\tswm_discovery_20261016_v1\cycle54'
if(Test-Path -LiteralPath "$root\outputs\postprocess54_v3.stdout.log"){throw 'Preserve existing postprocessing logs'}
if(Test-Path -LiteralPath "$root\outputs\SELECTION_SEAL54_V1.json"){throw 'Primary seal exists; inspect scoring before changing gate'}
$prior=Get-Content -LiteralPath "$root\outputs\postprocess54_v2.json" -Raw | ConvertFrom-Json
if($prior.status -ne 'WAITING_FOR_ALL_12_SEALED_FITS'){throw 'Prior watcher is no longer waiting; inspect'}
$targets=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*\cycle54\scripts\postprocess54_v2.py*'})
if(-not $targets.Count){throw 'Prior watcher handles not found; inspect'}
$receipt=@{utc=[DateTime]::UtcNow.ToString('o');reason='Predeclared existing EM-option learning control; freeze all16 models/maps before any heldout score';prior_status=[string]$prior.status;stopped_process_ids=@($targets | ForEach-Object {[int]$_.ProcessId})}
$receipt | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath "$root\outputs\POSTPROCESS_WATCHER_HANDOFF54_V3.json" -Encoding UTF8
foreach($target in ($targets | Sort-Object @{Expression={if($_.ExecutablePath -like 'E:*'){0}else{1}}})){
 if(Get-Process -Id $target.ProcessId -ErrorAction SilentlyContinue){Stop-Process -Id $target.ProcessId -ErrorAction Stop}
}
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$launch=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"D:\tswm_discovery_20261016_v1\cycle2\.venv_rpm\Scripts\python.exe" -B -u "D:\tswm_discovery_20261016_v1\cycle54\scripts\postprocess54_v3.py"';CurrentDirectory=$root;ProcessStartupInformation=$startup}
$launch | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($launch.ReturnValue -ne 0){throw 'All16 postprocessing launch failed'}
