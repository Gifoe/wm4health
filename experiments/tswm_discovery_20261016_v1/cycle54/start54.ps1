$ErrorActionPreference='Stop'
$root='D:\tswm_discovery_20261016_v1\cycle54'
$python='D:\tswm_discovery_20261016_v1\cycle2\.venv_rpm\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path "$root\outputs" | Out-Null
if (Test-Path -LiteralPath "$root\outputs\supervisor54.stdout.log") { throw 'Preserve existing launcher logs' }
$p=Start-Process -FilePath $python -ArgumentList @('-B','-u',"$root\scripts\supervise54.py") -WorkingDirectory $root -RedirectStandardOutput "$root\outputs\supervisor54.stdout.log" -RedirectStandardError "$root\outputs\supervisor54.stderr.log" -WindowStyle Hidden -PassThru
@{pid=$p.Id;utc=[DateTime]::UtcNow.ToString('o');status='STARTED'} | ConvertTo-Json -Compress
