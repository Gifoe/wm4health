$ErrorActionPreference='Stop'
$root89='D:\tswm_discovery_20261016_v1'
$p89=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -match 'cycle89.*(capture_csdi_eval89_v1|eval_csdi89_v1)'})
if($p89.Count -gt 0){$p89 | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress; exit 0}
if(Test-Path -LiteralPath ($root89+'\cycle89\outputs\csdi_eval_v1\HANDLE.json')){throw 'Existing CSDI eval handle: inspect before any restart'}
$startup89=New-CimInstance Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('"'+$root89+'\.venv\Scripts\python.exe" -B -X faulthandler -u "'+$root89+'\cycle89\scripts\capture_csdi_eval89_v1.py"');CurrentDirectory=$root89;ProcessStartupInformation=$startup89} | Select-Object ProcessId,ReturnValue | ConvertTo-Json -Compress
