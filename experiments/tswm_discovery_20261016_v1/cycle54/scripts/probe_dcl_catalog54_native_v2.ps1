$ErrorActionPreference='Stop'
$root='D:\tswm_discovery_20261016_v1\cycle54'
$output=Join-Path $root 'outputs\dcl_catalog_probe54_native_v2.json'
if(Test-Path -LiteralPath $output){throw 'Retain the existing metadata receipt'}
$requests=@()
foreach($url in @('https://nefeli.helmholtz-munich.de/records/tdfs2-kx054','https://nefeli.helmholtz-munich.de/api/records/tdfs2-kx054')){
  $watch=[Diagnostics.Stopwatch]::StartNew()
  try{
    $response=Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 18
    $content=[string]$response.Content
    if($content.Length -gt 524288){throw 'Metadata exceeds bounded content acceptance; no linked files requested'}
    $record=@{url=$url;status=[int]$response.StatusCode;content_type=[string]$response.Headers['Content-Type'];content_chars=$content.Length;elapsed_seconds=$watch.Elapsed.TotalSeconds}
    if($url -like '*/api/records/*'){
      $parsed=$content | ConvertFrom-Json
      $record.record_id=$parsed.id
      $record.metadata=$parsed.metadata
      $record.files=$parsed.files
      $record.links=$parsed.links
    } else {$record.contains_record_id=$content.Contains('tdfs2-kx054')}
    $requests+=$record
  }catch{
    $requests+=@{url=$url;error_type=$_.Exception.GetType().FullName;error=$_.Exception.Message;elapsed_seconds=$watch.Elapsed.TotalSeconds}
  }
}
$result=@{utc=[DateTime]::UtcNow.ToString('o');status='NATIVE_WINDOWS_TLS_METADATA_PROBE_COMPLETE';transport='Invoke-WebRequest with unchanged Windows default certificate validation';requests=$requests;signals_downloaded=$false;formal_training_changed=$false;script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLower()}
$json=$result | ConvertTo-Json -Depth 35
[IO.File]::WriteAllText($output,$json,(New-Object Text.UTF8Encoding($false)))
Write-Output $json
