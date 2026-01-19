# run-step4-dev.ps1
# Step 4 runner for step-xcaf-docker (mounts live repo to /app)

$ErrorActionPreference = "Stop"

$repo  = Split-Path -Parent $MyInvocation.MyCommand.Path
$inDir = Join-Path $repo "in"
$outDir = Join-Path $repo "out"

if (-not (Test-Path $inDir))  { New-Item -ItemType Directory -Path $inDir  | Out-Null }
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$envFile  = Join-Path $repo ".env"
$envLocal = Join-Path $repo ".env.local"

if (-not (Test-Path $envFile)) {
  throw "Missing .env at $envFile"
}

# Build env-file args (use .env.local if present)
$envArgs = @("--env-file", $envFile)
if (Test-Path $envLocal) {
  $envArgs += @("--env-file", $envLocal)
}

# Pass-through any additional CLI args to the script
# Example:
#   .\run-step4-dev.ps1 --plate-thick-max-mm 30 --grating-fill-ratio-max 0.20
$scriptArgs = $args

docker run --rm -it `
  @envArgs `
  -v "${inDir}:/in" `
  -v "${outDir}:/out" `
  -v "${repo}:/app" `
  --entrypoint python `
  step-xcaf-docker `
  /app/step4_make_procurement_csv.py @scriptArgs

if ($LASTEXITCODE -ne 0) {
  throw "docker failed with exit code $LASTEXITCODE"
}

Write-Host "Step 4 complete. Outputs in: $outDir"
