param(
  [string]$Repo = (Get-Location).Path,
  [string]$Image = "step-xcaf-docker:latest"
)

$ErrorActionPreference = "Stop"

$repoPath = (Resolve-Path $Repo).Path
$outPath  = Join-Path $repoPath "out"
if (!(Test-Path $outPath)) { New-Item -ItemType Directory -Path $outPath | Out-Null }

docker run --rm `
  -v "${repoPath}:/app" `
  -v "${outPath}:/out" `
  -w /app `
  $Image `
  /app/step8_enrich_active.py --backup --annotate

if ($LASTEXITCODE -ne 0) { throw "docker failed with exit code $LASTEXITCODE" }
Write-Host "=== DONE Step 8 (enriched active) ==="
