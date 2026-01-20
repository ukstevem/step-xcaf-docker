param(
  [string]$ImageName = "step-xcaf-docker",
  [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Ensure-Dir([string]$p) {
  if (-not (Test-Path -LiteralPath $p)) {
    New-Item -ItemType Directory -Force -Path $p | Out-Null
  }
}

function Run-Docker([string[]]$DockerArgs) {
  Write-Host ("docker " + ($DockerArgs -join " ")) -ForegroundColor DarkGray
  & docker @DockerArgs
  if ($LASTEXITCODE -ne 0) { throw ("docker failed with exit code " + $LASTEXITCODE) }
}

$repoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = Join-Path $repoRoot "out" }

# Ensure out dir exists early (used by checks below)
Ensure-Dir $OutDir

# --- Ensure Step 4 review JSONs exist (auto-run Step 4 Stage 1-3 if missing) ---
$reviewJson    = Join-Path $OutDir "review\multibody_review.json"
$decisionsJson = Join-Path $OutDir "review\multibody_decisions.json"

$needStep4 = (-not (Test-Path -LiteralPath $reviewJson)) -or (-not (Test-Path -LiteralPath $decisionsJson))
if ($needStep4) {
  Write-Host ""
  Write-Host "[Step 4 UI BUILD] review JSONs missing -> running Step 4 review generation first" -ForegroundColor Yellow

  $step4Runner = Join-Path $repoRoot "run-step4-dev.ps1"
  if (-not (Test-Path -LiteralPath $step4Runner)) {
    throw ("Missing runner: " + $step4Runner)
  }

  & $step4Runner -OutDir $OutDir -ImageName $ImageName
  if ($LASTEXITCODE -ne 0) { throw ("run-step4-dev.ps1 failed with exit code " + $LASTEXITCODE) }

  if (-not (Test-Path -LiteralPath $reviewJson))    { throw ("Still missing after Step 4 run: " + $reviewJson) }
  if (-not (Test-Path -LiteralPath $decisionsJson)) { throw ("Still missing after Step 4 run: " + $decisionsJson) }
}

# 1) Validate vendor source exists (fail hard if missing)
$vendorSrc = Join-Path $repoRoot "vendor"
if (-not (Test-Path -LiteralPath $vendorSrc)) {
  throw ("Missing vendor directory at repo root: " + $vendorSrc + " (required for offline viewer)")
}

# 2) Ensure output folders exist
$reviewUiDir = Join-Path $OutDir "review_ui"
$vendorDst   = Join-Path $reviewUiDir "vendor"
Ensure-Dir $reviewUiDir
Ensure-Dir $vendorDst

# 3) Copy vendor files into out/review_ui/vendor (overwrite)
Write-Host ""
Write-Host "[Step 4 UI BUILD] copying vendor -> out/review_ui/vendor" -ForegroundColor Green
Copy-Item -Recurse -Force -LiteralPath (Join-Path $vendorSrc "*") -Destination $vendorDst

# 4) Run the python UI build (items.json + stl cache + html/js/css)
$repoAbs = (Resolve-Path -LiteralPath $repoRoot).Path
$outAbs  = (Resolve-Path -LiteralPath $OutDir).Path

Write-Host ""
Write-Host "[Step 4 UI BUILD] generating out/review_ui (items.json + stl cache)" -ForegroundColor Green

$scriptArgs = @(
  "run","--rm",
  "--entrypoint","python",
  "-v", ($outAbs + ":/out"),
  "-v", ($repoAbs + ":/app"),
  "-w","/app",
  $ImageName,
  "-u","/app/step4_multibody_ui_build.py",
  "--outdir","/out"
)

Run-Docker $scriptArgs

# 5) Sanity checks (fail hard if missing)
$itemsJson = Join-Path $reviewUiDir "items.json"
$indexHtml = Join-Path $reviewUiDir "index.html"

if (-not (Test-Path -LiteralPath $itemsJson)) { throw ("Missing output: " + $itemsJson) }
if (-not (Test-Path -LiteralPath $indexHtml)) { throw ("Missing output: " + $indexHtml) }

Write-Host ""
Write-Host "Built out/review_ui successfully." -ForegroundColor Cyan
Write-Host ("  " + $itemsJson)
Write-Host ("  " + $indexHtml)
Write-Host ("  " + $vendorDst)
