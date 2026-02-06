param(
  [string]$ImageName = "step-xcaf-docker",
  [string]$OutDir = "",
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Ensure-Dir([string]$p) {
  if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

function Assert-Exists([string]$p, [string]$msg) {
  if (-not (Test-Path -LiteralPath $p)) { throw $msg }
}

function Run-Docker([string[]]$DockerArgs) {
  Write-Host ("docker " + ($DockerArgs -join " ")) -ForegroundColor DarkGray
  & docker @DockerArgs
  if ($LASTEXITCODE -ne 0) { throw ("docker failed with exit code " + $LASTEXITCODE) }
}

$repoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = Join-Path $repoRoot "out" }
Ensure-Dir $OutDir

if ($Clean) {
  Remove-Item -Recurse -Force (Join-Path $OutDir "review")    -ErrorAction SilentlyContinue
  Remove-Item -Recurse -Force (Join-Path $OutDir "review_ui") -ErrorAction SilentlyContinue
}

$repoAbs = (Resolve-Path -LiteralPath $repoRoot).Path
$outAbs  = (Resolve-Path -LiteralPath $OutDir).Path

# Inputs
$xcafHost = Join-Path $OutDir "xcaf_instances.json"
Assert-Exists $xcafHost ("Missing input: " + $xcafHost + " (run Step 1 first)")

# Step 4 (Stage 1-3): generate review JSONs
$scriptArgs = @(
  "run","--rm",
  "--entrypoint","python",
  "-v", ($outAbs + ":/out"),
  "-v", ($repoAbs + ":/app"),
  "-w","/app",
  $ImageName,
  "-u","/app/step4_multibody_review.py",
  "--in","/out/xcaf_instances.json",
  "--outdir","/out"
)

Run-Docker $scriptArgs

# Outputs (JSON)
$reviewJson    = Join-Path $OutDir "review\multibody_review.json"
$decisionsJson = Join-Path $OutDir "review\multibody_decisions.json"

Assert-Exists $reviewJson    ("Missing output: " + $reviewJson)
Assert-Exists $decisionsJson ("Missing output: " + $decisionsJson)

Write-Host ""
Write-Host "Step 4 review JSONs written:" -ForegroundColor Cyan
Write-Host ("  " + $reviewJson)
Write-Host ("  " + $decisionsJson)
