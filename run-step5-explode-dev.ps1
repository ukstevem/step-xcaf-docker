param(
  [Parameter(Mandatory=$true)]
  [string] $StepPath,

  [string] $Image = "step-xcaf-docker:latest",

  [string] $OutDir = ".\out"
)

$ErrorActionPreference = "Stop"

function Require-File([string] $p, [string] $label) {
  if (-not (Test-Path $p -PathType Leaf)) {
    throw ("Missing required {0}: {1}" -f $label, $p)
  }
}

$RepoRoot = (Resolve-Path ".").Path
$OutAbs   = (Resolve-Path $OutDir).Path

# JSON decisions (new canonical)
$dec = Join-Path $OutAbs "review\multibody_decisions.json"

# Inputs that must already exist from Steps 1-4
Require-File $dec "out\review\multibody_decisions.json"
Require-File (Join-Path $OutAbs "xcaf_instances.json") "out\xcaf_instances.json"
Require-File (Join-Path $RepoRoot "step5_explode_multibody.py") "step5_explode_multibody.py"

if ((Get-Item $dec).Length -lt 40) { throw "multibody_decisions.json looks too small: $dec" }

# Resolve STEP path and mount its directory
$StepAbs = (Resolve-Path $StepPath).Path
Require-File $StepAbs "STEP file"

$StepDir  = Split-Path -Parent $StepAbs
$StepName = Split-Path -Leaf   $StepAbs

Write-Host "=== Step 5 (Explode) DEV ==="
Write-Host "repo    : $RepoRoot"
Write-Host "step    : $StepAbs"
Write-Host "out     : $OutAbs"
Write-Host "image   : $Image"
Write-Host ""

# Bind mounts:
# - repo to /app (latest code, no rebuild)
# - out to /out
# - STEP folder to /in_step
docker run --rm `
  --entrypoint python `
  -v "${RepoRoot}:/app" `
  -v "${OutAbs}:/out" `
  -v "${StepDir}:/in_step" `
  -w /app `
  $Image `
  /app/step5_explode_multibody.py `
    --step_path "/in_step/$StepName" `
    --out_dir "/out"

if ($LASTEXITCODE -ne 0) {
  throw "docker failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done."
Write-Host "Outputs:"
Write-Host "  $OutAbs\exploded\exploded_parts.json"
Write-Host "  $OutAbs\exploded\explode_log.jsonl"
Write-Host "  $OutAbs\exploded\stl\..."
