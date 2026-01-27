param(
  [string]$Repo = (Get-Location).Path,
  [string]$Image = "step-xcaf-docker:latest"
)

$ErrorActionPreference = "Stop"

$repoPath = (Resolve-Path $Repo).Path
$outPath  = Join-Path $repoPath "out"
if (!(Test-Path $outPath)) { New-Item -ItemType Directory -Path $outPath | Out-Null }

# Step 9.2 first (base bbox/vol/name/category)
docker run --rm `
  -v "${repoPath}:/app" `
  -v "${outPath}:/out" `
  -w /app `
  $Image `
  /app/step9_enrich_base_from_xcaf_active.py --backup

if ($LASTEXITCODE -ne 0) { throw "docker failed (step9.2) exit code $LASTEXITCODE" }

# Step 8 (grouping + UI tables)
docker run --rm `
  -v "${repoPath}:/app" `
  -v "${outPath}:/out" `
  -w /app `
  $Image `
  /app/step8_enrich_active.py --backup --annotate

if ($LASTEXITCODE -ne 0) { throw "docker failed (step8) exit code $LASTEXITCODE" }

# Step 9 (subparts classify + rollup tables)
docker run --rm `
  -v "${repoPath}:/app" `
  -v "${outPath}:/out" `
  -w /app `
  $Image `
  /app/step9_classify_and_rollup_active.py --backup

if ($LASTEXITCODE -ne 0) { throw "docker failed (step9) exit code $LASTEXITCODE" }

docker run --rm `
  -v "${repoPath}:/app" `
  -v "${outPath}:/out" `
  -w /app `
  $Image `
  /app/step9_add_occurrence_tree.py --backup --include_xform

if ($LASTEXITCODE -ne 0) { throw "docker failed with exit code $LASTEXITCODE" }

Write-Host "=== DONE Step 9.2 -> Step 8 -> Step 9 -> Step 9.3 (active enriched, UI-ready, occurence tree) ==="
