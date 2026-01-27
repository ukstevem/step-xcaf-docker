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
  /app/step9_classify_and_rollup_active.py --backup

if ($LASTEXITCODE -ne 0) { throw "docker failed with exit code $LASTEXITCODE" }
Write-Host "=== DONE Step 9 (classified + rolled up active) ==="
