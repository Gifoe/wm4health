$ErrorActionPreference='Stop'
$cycleRoot59='D:\tswm_discovery_20261016_v1\cycle59'
New-Item -ItemType Directory -Path "$cycleRoot59\outputs" -Force | Out-Null
if(Test-Path -LiteralPath "$cycleRoot59\outputs\launcher59_v1.json"){throw 'Prior launch receipt exists: inspect the owner instead of restarting.'}
$process59=Start-Process -FilePath 'D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe' -ArgumentList @('-B','-u',"$cycleRoot59\scripts\capture59_v1.py") -WorkingDirectory $cycleRoot59 -WindowStyle Hidden -RedirectStandardOutput "$cycleRoot59\outputs\supervisor59_v1.stdout.log" -RedirectStandardError "$cycleRoot59\outputs\supervisor59_v1.stderr.log" -PassThru
@{utc=[DateTime]::UtcNow.ToString('o');pid=$process59.Id;script='capture59_v1.py'} | ConvertTo-Json | Set-Content -LiteralPath "$cycleRoot59\outputs\launcher59_v1.json" -Encoding UTF8
Get-Content -Raw -LiteralPath "$cycleRoot59\outputs\launcher59_v1.json"
