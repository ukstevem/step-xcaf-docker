param(
  [Parameter(Mandatory = $true)]
  [string]$StepPath,

  [string]$ImageName = "step-xcaf-docker",

  [string]$OutDir = "",

  [switch]$WithMassprops
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
  if ($LASTEXITCODE -ne 0) { throw "docker failed with exit code $LASTEXITCODE" }
}

$repoRoot = $PSScriptRoot
$inDir    = Join-Path $repoRoot "in"
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = Join-Path $repoRoot "out" }

Ensure-Dir $inDir
Ensure-Dir $OutDir

if (-not (Test-Path -LiteralPath $StepPath)) {
  throw "STEP file not found: $StepPath"
}

$stepAbs = (Resolve-Path -LiteralPath $StepPath).Path
$inAbs   = (Resolve-Path -LiteralPath $inDir).Path
$outAbs  = (Resolve-Path -LiteralPath $OutDir).Path
$repoAbs = (Resolve-Path -LiteralPath $repoRoot).Path

# Copy STEP into ./in (skip if already there)
$leaf      = Split-Path $stepAbs -Leaf
$localStep = Join-Path $inDir $leaf
if ($stepAbs -ne $localStep) {
  Copy-Item -LiteralPath $stepAbs -Destination $localStep -Force
}

$stepIn = "/in/$leaf"

# Ensure image exists (no rebuild; dev workflow expects image already built once)
try {
  Run-Docker @("image","inspect",$ImageName) | Out-Null
} catch {
  throw "Docker image '$ImageName' not found. Build it once: docker build -t $ImageName ."
}

Write-Host "`n[Step 1 DEV] XCAF extract -> xcaf_instances.json" -ForegroundColor Green
Write-Host "Repo mount: $repoAbs -> /app (live code)" -ForegroundColor Cyan

$scriptArgs = @(
  "run","--rm",
  "-v","${inAbs}:/in:ro",
  "-v","${outAbs}:/out",
  "-v","${repoAbs}:/app",
  "-w","/app",
  $ImageName,
  "-u","/app/read_step_xcaf.py",
  $stepIn, "/out"
)

if ($WithMassprops) {
  $scriptArgs += "--with_massprops"
}

Run-Docker $scriptArgs

$outJson = Join-Path $OutDir "xcaf_instances.json"
if (-not (Test-Path -LiteralPath $outJson)) {
  throw "Missing output: $outJson"
}

Write-Host "`nDone." -ForegroundColor Cyan
Write-Host "Output:" -ForegroundColor Cyan
Write-Host "  $outJson"
