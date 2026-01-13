param(
  [Parameter(Mandatory = $true)]
  [string]$StepPath,

  [string]$ImageName = "step-xcaf-docker",

  # Where outputs go (default: <repo>\out)
  [string]$OutDir = "",

  # Step 2 options
  [switch]$OverwriteStl,
  [switch]$AsciiStl,
  [double]$LinearDeflection = 0.25,
  [double]$AngularDeflection = 0.35,

  # Step 3 option
  [switch]$SkipChirality
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

# Check image exists (dev workflow assumes you built it once)
try {
  Run-Docker @("image","inspect",$ImageName) | Out-Null
} catch {
  throw "Docker image '$ImageName' not found. Build it once: docker build -t $ImageName ."
}

# Step 2 expects xcaf_instances.json produced by Step 1
$xcafJson = Join-Path $OutDir "xcaf_instances.json"
if (-not (Test-Path -LiteralPath $xcafJson)) {
  throw "Missing prerequisite: $xcafJson`nRun Step 1 first to produce xcaf_instances.json."
}

Write-Host "`n[Step 2 DEV] Export STLs + stl_manifest.json" -ForegroundColor Green
Write-Host "Repo mount: $repoAbs -> /app (live code)" -ForegroundColor Cyan

$step2Args = @(
  "run","--rm",
  "--entrypoint","python",
  "-v","${inAbs}:/in:ro",
  "-v","${outAbs}:/out",
  "-v","${repoAbs}:/app",
  "-w","/app",
  $ImageName,
  "/app/export_stl_xcaf.py",
  "--step-path",$stepIn,
  "--out-dir","/out",
  "--xcaf-json","/out/xcaf_instances.json",
  "--linear-deflection",$LinearDeflection.ToString("0.############"),
  "--angular-deflection",$AngularDeflection.ToString("0.############")
)

if ($OverwriteStl) { $step2Args += "--overwrite" }
if ($AsciiStl)     { $step2Args += "--ascii-stl" }

Run-Docker $step2Args

$manifestJson = Join-Path $OutDir "stl_manifest.json"
if (-not (Test-Path -LiteralPath $manifestJson)) {
  throw "Missing output: $manifestJson"
}

Write-Host "`n[Step 3 DEV] Add chirality signatures to stl_manifest.json" -ForegroundColor Green

if (-not $SkipChirality) {
  $step3Args = @(
    "run","--rm",
    "--entrypoint","python",
    "-v","${outAbs}:/out",
    "-v","${repoAbs}:/app",
    "-w","/app",
    $ImageName,
    "/app/add_chirality_to_manifest.py",
    "--out-dir","/out",
    "--manifest","/out/stl_manifest.json"
  )


  Run-Docker $step3Args
} else {
  Write-Host "Skipping chirality step (--SkipChirality set)" -ForegroundColor Yellow
}

Write-Host "`nDone." -ForegroundColor Cyan
Write-Host "Outputs:" -ForegroundColor Cyan
Write-Host "  $manifestJson"
Write-Host "  $(Join-Path $OutDir 'stl\')" -ForegroundColor Cyan
