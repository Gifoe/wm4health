$ErrorActionPreference='Stop'
$target62=Join-Path $PSScriptRoot 'outputs\WINDOWS_EVENT62_V1.json'
if(Test-Path -LiteralPath $target62){throw 'Event receipt exists; preserve it.'}
$event62=Get-WinEvent -FilterHashtable @{LogName='Application';Id=1000;StartTime=(Get-Date).AddMinutes(-30)} -MaxEvents 20 | Where-Object {$_.Message -like '*persist_stable_251*'} | Select-Object -First 1
if(-not $event62){throw 'Expected application failure event unavailable.'}
$record62=@{event_id=$event62.Id;record_id=$event62.RecordId;utc=$event62.TimeCreated.ToUniversalTime().ToString('o');reported_properties=$event62.Properties.Value;scope='Reported native fault metadata; root cause not established.'}
[IO.File]::WriteAllText($target62,($record62 | ConvertTo-Json -Depth 5),(New-Object Text.UTF8Encoding($false)))
@{status='WINDOWS_EVENT_PRESERVED';record_id=$event62.RecordId;utc=$record62.utc} | ConvertTo-Json -Compress
