param(
  [Parameter(Mandatory = $true)]
  [string]$StepPath,

  [string]$ImageName = "step-xcaf-docker",

  [string]$OutDir = "",

  [switch]$WithMassprops,

  [switch]$BuildNoCache
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

# Copy STEP into ./in (skip if already there)
$leaf      = Split-Path $stepAbs -Leaf
$localStep = Join-Path $inDir $leaf
if ($stepAbs -ne $localStep) {
  Copy-Item -LiteralPath $stepAbs -Destination $localStep -Force
} else {
  Write-Host "STEP already in ./in, skipping copy: $localStep" -ForegroundColor DarkGray
}

$stepIn = "/in/$leaf"

# Build image if requested (no-cache)
if ($BuildNoCache) {
  Write-Host "Building docker image (no-cache): $ImageName" -ForegroundColor Cyan
  Run-Docker @("build","--no-cache","-t",$ImageName,$repoRoot)
} else {
  # Ensure image exists; if not, build it once (cached)
  try {
    Run-Docker @("image","inspect",$ImageName) | Out-Null
  } catch {
    Write-Host "Docker image not found; building: $ImageName" -ForegroundColor Cyan
    Run-Docker @("build","-t",$ImageName,$repoRoot)
  }
}

Write-Host "`n[1/1] Step 1 XCAF extract -> xcaf_instances.json" -ForegroundColor Green

$scriptArgs = @(
  "run","--rm",
  "-v","${inAbs}:/in:ro",
  "-v","${outAbs}:/out",
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
