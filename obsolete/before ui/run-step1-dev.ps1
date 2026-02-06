param(
  [Parameter(Mandatory = $true)]
  [string]$StepPath,

  # Optional explicit env file; if blank we prefer .env.local then .env
  [string]$EnvFile = "",

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

function Get-DockerfilePath([string]$root) {
  $df1 = Join-Path $root "Dockerfile"
  $df2 = Join-Path $root "dockerfile"
  if (Test-Path -LiteralPath $df1) { return $df1 }
  if (Test-Path -LiteralPath $df2) { return $df2 }
  throw "No Dockerfile found at '$df1' or '$df2'"
}

# Robust repo root:
# - when run as a script, $PSScriptRoot is set
# - when copy/pasted in console, it can be empty -> fall back to script path -> then cwd
$repoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($repoRoot)) {
  try {
    $repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
  } catch {
    $repoRoot = (Get-Location).Path
  }
}

# Pick env file (prefer .env.local then .env) unless explicitly provided
$envMain  = Join-Path $repoRoot ".env"
$envLocal = Join-Path $repoRoot ".env.local"

if ([string]::IsNullOrWhiteSpace($EnvFile)) {
  if (Test-Path -LiteralPath $envLocal) {
    $EnvFile = $envLocal
  } elseif (Test-Path -LiteralPath $envMain) {
    $EnvFile = $envMain
  } else {
    $EnvFile = ""
  }
}

$inDir = Join-Path $repoRoot "in"
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

# Resolve env file (optional)
$envAbs = ""
if (-not [string]::IsNullOrWhiteSpace($EnvFile)) {
  if (Test-Path -LiteralPath $EnvFile) {
    $envAbs = (Resolve-Path -LiteralPath $EnvFile).Path
  } else {
    Write-Host "NOTE: Env file not found at $EnvFile (continuing without --env-file)" -ForegroundColor Yellow
  }
}

# Copy STEP into ./in (skip if already there)
$leaf      = Split-Path $stepAbs -Leaf
$localStep = Join-Path $inDir $leaf
if ($stepAbs -ne $localStep) {
  Copy-Item -LiteralPath $stepAbs -Destination $localStep -Force
}

$stepIn = "/in/$leaf"

# Always rebuild the image from the *current* local repo so code changes are picked up.
# (You still mount /app live, but this guarantees correctness even if mounts are misconfigured.)
$dockerfilePath = Get-DockerfilePath $repoRoot
Write-Host "`nBuilding image '$ImageName' from: $dockerfilePath" -ForegroundColor Cyan
Run-Docker @(
  "build",
  "--pull",
  "-t", $ImageName,
  "-f", $dockerfilePath,
  $repoAbs
)

Write-Host "`n[Step 1 DEV] XCAF extract -> xcaf_instances.json" -ForegroundColor Green
Write-Host "Repo mount: $repoAbs -> /app (live code)" -ForegroundColor Cyan
if ($envAbs) { Write-Host "Env file  : $envAbs -> --env-file" -ForegroundColor Cyan }

# IMPORTANT: force python entrypoint so we always run the mounted /app/read_step_xcaf.py
$scriptArgs = @(
  "run","--rm",
  "--entrypoint","python",
  "-v","${inAbs}:/in:ro",
  "-v","${outAbs}:/out",
  "-v","${repoAbs}:/app",
  "-w","/app"
)

if ($envAbs) {
  $scriptArgs += @("--env-file", $envAbs)
}

$scriptArgs += @(
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
