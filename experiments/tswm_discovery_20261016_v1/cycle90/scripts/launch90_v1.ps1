$ErrorActionPreference='Stop'
$root90='D:\tswm_discovery_20261016_v1'
$p90=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -match 'cycle90.*(capture90_v1|train90_v1|screen90_v1)'})
if($p90.Count -gt 0){$p90 | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress; exit 0}
if(Test-Path -LiteralPath ($root90+'\cycle90\outputs\campaign_v1\HANDLE.json')){throw 'Existing campaign handle: inspect actual process before any restart'}
$startup90=New-CimInstance Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('"'+$root90+'\.venv\Scripts\python.exe" -B -X faulthandler -u "'+$root90+'\cycle90\scripts\capture90_v1.py"');CurrentDirectory=$root90;ProcessStartupInformation=$startup90} | Select-Object ProcessId,ReturnValue | ConvertTo-Json -Compress
