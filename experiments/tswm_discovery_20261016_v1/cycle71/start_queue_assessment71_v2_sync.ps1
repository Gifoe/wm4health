$ErrorActionPreference='Stop'
$env:CUDA_LAUNCH_BLOCKING='1'
$runtime71=@{utc=[datetime]::UtcNow.ToString('o');CUDA_LAUNCH_BLOCKING=$env:CUDA_LAUNCH_BLOCKING;purpose='Synchronous diagnostic execution after the recorded driver incident; forecast protocol unchanged'}
$runtime71 | ConvertTo-Json | Out-File -LiteralPath 'D:\tswm_discovery_20261016_v1\cycle71\outputs\ASSESSMENT_RUNTIME71_V2.json' -Encoding utf8 -NoClobber
& powershell.exe -NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -File 'D:\tswm_discovery_20261016_v1\cycle71\start_queue_assessment71_v2.ps1'
if($LASTEXITCODE -ne 0){throw 'Assessment queue did not start'}
