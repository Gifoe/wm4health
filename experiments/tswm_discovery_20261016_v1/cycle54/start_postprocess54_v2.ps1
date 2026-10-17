$ErrorActionPreference='Stop'
$root='D:\tswm_discovery_20261016_v1\cycle54'
if(Test-Path -LiteralPath "$root\outputs\postprocess54_v2.stdout.log"){throw 'Preserve existing postprocessing logs'}
if(Test-Path -LiteralPath "$root\outputs\SELECTION_SEAL54_V1.json"){throw 'Seal already exists; inspect active processing before handoff'}
$targets=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*\cycle54\scripts\postprocess54_v1.py*'})
if(-not $targets.Count){throw 'Expected old waiting watcher not found; inspect'}
if((Get-Content -LiteralPath "$root\outputs\postprocess54_v1.json" -Raw | ConvertFrom-Json).status -ne 'WAITING_FOR_ALL_12_SEALED_FITS'){throw 'Watcher no longer waiting; inspect'}
$receipt=@{utc=[DateTime]::UtcNow.ToString('o');reason='Qualified baseline implementation amendment; replace only waiting postprocessing watcher';priorState=Get-Content -LiteralPath "$root\outputs\postprocess54_v1.json" -Raw;handles=@($targets | Select-Object ProcessId,ExecutablePath,CommandLine)}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$root\outputs\POSTPROCESS_WATCHER_HANDOFF54_V2.json" -Encoding UTF8
foreach($target in ($targets | Sort-Object @{Expression={if($_.ExecutablePath -like 'E:*'){0}else{1}}})){
 if(Get-Process -Id $target.ProcessId -ErrorAction SilentlyContinue){Stop-Process -Id $target.ProcessId -ErrorAction Stop}
}
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$launch=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"D:\tswm_discovery_20261016_v1\cycle2\.venv_rpm\Scripts\python.exe" -B -u "D:\tswm_discovery_20261016_v1\cycle54\scripts\postprocess54_v2.py"';CurrentDirectory=$root;ProcessStartupInformation=$startup}
$launch | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($launch.ReturnValue -ne 0){throw 'Amended postprocessing launch failed'}
