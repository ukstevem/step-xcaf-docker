param(
  [string]$Image  = "step-xcaf-docker:latest",
  [string]$OutDir = ".\out"
)

$ErrorActionPreference = "Stop"

function Resolve-OnePath([string]$p) {
  $rp = Resolve-Path -LiteralPath $p
  if ($null -eq $rp) { throw "Resolve-Path failed for: $p" }
  return [string]($rp | Select-Object -First 1).Path
}

$repo = Resolve-OnePath "."
$out  = Resolve-OnePath $OutDir

# Verify required inputs exist on host
$req = @(
  (Join-Path -Path $out -ChildPath "xcaf_instances.json"),
  (Join-Path -Path $out -ChildPath "assets_manifest.json"),
  (Join-Path -Path $out -ChildPath "review\multibody_decisions.json"),
  (Join-Path -Path $out -ChildPath "exploded\exploded_parts.json")
)
foreach($p in $req){
  if(-not (Test-Path -LiteralPath $p)){
    throw "Missing required input: $p"
  }
}

Write-Host "[step6] repo: $repo"
Write-Host "[step6] out : $out"
Write-Host "[step6] img : $Image"

docker run --rm `
  --entrypoint python `
  -v "${repo}:/app" `
  -v "${out}:/out" `
  -w /app `
  $Image `
  /app/step6_merge_derived.py --out /out

if($LASTEXITCODE -ne 0){
  throw "docker failed with exit code $LASTEXITCODE"
}

Write-Host "[step6] DONE"
Write-Host "  out\xcaf_instances_active.json"
Write-Host "  out\assets_manifest_active.json"
Write-Host "  out\derived\step6_merge_log.jsonl"
