param(
  [string]$Image = "step-xcaf-docker:latest",
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

$active = Join-Path -Path $out -ChildPath "assets_manifest_active.json"
if(-not (Test-Path -LiteralPath $active)){
  throw "Missing prerequisite: $active`nRun Step 6 merge first."
}

Write-Host "[step7] repo: $repo"
Write-Host "[step7] out : $out"
Write-Host "[step7] img : $Image"

docker run --rm `
  --entrypoint python `
  -v "${out}:/out" `
  -v "${repo}:/app" `
  -w "/app" `
  $Image `
  /app/step7_add_chirality_to_active_manifest.py `
  --out-dir /out `
  --manifest /out/assets_manifest_active.json

if($LASTEXITCODE -ne 0){
  throw "docker failed with exit code $LASTEXITCODE"
}

Write-Host "[step7] DONE"
Write-Host "  out\assets_manifest_active.json"
