$ErrorActionPreference='Stop'
$r='D:\tswm_discovery_20261016_v1\meta_ilad_native_20261023_v1'
if(Test-Path -LiteralPath ($r+'\acquisition_execution_v1')){throw 'Acquisition exists; inspect actual process and receipts before retry.'}
$body=@'
& 'D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe' -I -B -u 'D:\tswm_discovery_20261016_v1\meta_ilad_native_20261023_v1\scripts\capture_acquisition_v1.py'
exit $LASTEXITCODE
'@
$enc=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($body))
$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}
$j=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=('powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand '+$enc);CurrentDirectory=$r;ProcessStartupInformation=$startup}
$j | Select-Object ReturnValue,ProcessId | ConvertTo-Json -Compress
if($j.ReturnValue -ne 0){throw 'Acquisition launch failed.'}
