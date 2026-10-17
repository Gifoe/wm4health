$ErrorActionPreference='Stop'
$closeRoot57=$PSScriptRoot
$closeProject57=Split-Path -Parent $closeRoot57
$closePython57=Join-Path $closeProject57 '.venv\Scripts\python.exe'
$closeExisting57=Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -like '*cycle57*' -and $_.CommandLine -like '*capture_close*'}
if($closeExisting57){throw 'C57 closure owner exists; inspect it'}
$closeCommand57='"'+$closePython57+'" -B -u "'+$closeRoot57+'\scripts\capture_close57_v1.py"'
$closeLaunch57=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$closeCommand57;CurrentDirectory=$closeRoot57}
if($closeLaunch57.ReturnValue -ne 0){throw 'C57 closure owner launch failed'}
@{utc=[DateTime]::UtcNow.ToString('o');status='LAUNCHED_INTERPRETATION_WAITER';stub_process_id=$closeLaunch57.ProcessId;heldout_model_selection=$false} | ConvertTo-Json -Compress
