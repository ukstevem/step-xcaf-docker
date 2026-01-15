param(
  [Parameter(Mandatory = $true)]
  [string]$StepPath,

  [string]$ImageName = "step-xcaf-docker",

  # Where outputs go (default: <repo>\out)
  [string]$OutDir = "",

  # Step 2 options (CLI overrides .env)
  [switch]$OverwriteStl,
  [switch]$AsciiStl,
  [double]$LinearDeflection,
  [double]$AngularDeflection,
  [switch]$UseSigfree,

  # Step 2 strictness
  [switch]$Strict,
  [switch]$StrictSignatureCollisions,

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

function Get-DockerfilePath([string]$root) {
  $df1 = Join-Path $root "Dockerfile"
  $df2 = Join-Path $root "dockerfile"
  if (Test-Path -LiteralPath $df1) { return $df1 }
  if (Test-Path -LiteralPath $df2) { return $df2 }
  throw "No Dockerfile found at '$df1' or '$df2'"
}

function Import-DotEnv([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) { return }

  Get-Content -LiteralPath $path | ForEach-Object {
    $line = $_.Trim()
    if ($line.Length -eq 0) { return }
    if ($line.StartsWith("#")) { return }
    $eq = $line.IndexOf("=")
    if ($eq -lt 1) { return }

    $k = $line.Substring(0, $eq).Trim()
    $v = $line.Substring($eq + 1).Trim()

    # strip optional quotes
    if ($v.StartsWith('"') -and $v.EndsWith('"')) { $v = $v.Substring(1, $v.Length - 2) }
    if ($v.StartsWith("'") -and $v.EndsWith("'")) { $v = $v.Substring(1, $v.Length - 2) }

    Set-Item -Path ("Env:\{0}" -f $k) -Value $v
  }
}

function Env-IsTruthy([string]$v) {
  if ([string]::IsNullOrWhiteSpace($v)) { return $false }
  $t = $v.Trim().ToLowerInvariant()
  return -not ($t -eq "0" -or $t -eq "false" -or $t -eq "no" -or $t -eq "off")
}

# -----------------------------
# Paths + load .env
# -----------------------------
$repoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($repoRoot)) {
  try {
    $repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
  } catch {
    $repoRoot = (Get-Location).Path
  }
}

$inDir    = Join-Path $repoRoot "in"
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = Join-Path $repoRoot "out" }

# Load env (main then local)
$envMain  = Join-Path $repoRoot ".env"
$envLocal = Join-Path $repoRoot ".env.local"
Import-DotEnv $envMain
Import-DotEnv $envLocal

# Choose env file to pass into docker (optional)
$envFileAbs = ""
if (Test-Path -LiteralPath $envLocal) {
  $envFileAbs = (Resolve-Path -LiteralPath $envLocal).Path
} elseif (Test-Path -LiteralPath $envMain) {
  $envFileAbs = (Resolve-Path -LiteralPath $envMain).Path
}

# Apply env defaults ONLY if user did not pass the parameter explicitly
if ($PSBoundParameters.ContainsKey("LinearDeflection") -eq $false -and $env:STL_LINEAR_DEFLECTION) {
  $LinearDeflection = [double]$env:STL_LINEAR_DEFLECTION
}
if ($PSBoundParameters.ContainsKey("AngularDeflection") -eq $false -and $env:STL_ANGULAR_DEFLECTION) {
  $AngularDeflection = [double]$env:STL_ANGULAR_DEFLECTION
}

# If still not set, use hard defaults
if ($PSBoundParameters.ContainsKey("LinearDeflection") -eq $false -and ($null -eq $LinearDeflection -or $LinearDeflection -eq 0)) {
  $LinearDeflection = 0.25
}
if ($PSBoundParameters.ContainsKey("AngularDeflection") -eq $false -and ($null -eq $AngularDeflection -or $AngularDeflection -eq 0)) {
  $AngularDeflection = 0.35
}

if (-not $UseSigfree -and (Env-IsTruthy $env:USE_SIG_FREE)) { $UseSigfree = $true }
if (-not $AsciiStl -and (Env-IsTruthy $env:STL_ASCII))     { $AsciiStl   = $true }
if (-not $SkipChirality -and (Env-IsTruthy $env:SKIP_CHIRALITY)) { $SkipChirality = $true }

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

# Step 2 expects xcaf_instances.json produced by Step 1
$xcafJson = Join-Path $OutDir "xcaf_instances.json"
if (-not (Test-Path -LiteralPath $xcafJson)) {
  throw "Missing prerequisite: $xcafJson`nRun Step 1 first to produce xcaf_instances.json."
}

Write-Host "`n[Step 2 DEV] Export STLs + stl_manifest.json" -ForegroundColor Green
Write-Host "Repo mount: $repoAbs -> /app (live code)" -ForegroundColor Cyan
if ($envFileAbs) { Write-Host ".env passed to container: $envFileAbs" -ForegroundColor Cyan }

Write-Host ("Step 2 settings: linear_deflection={0} angular_deflection={1} use_sig_free={2} overwrite={3} ascii={4}" -f `
  $LinearDeflection, $AngularDeflection, $UseSigfree, $OverwriteStl, $AsciiStl) -ForegroundColor Cyan

$step2Args = @(
  "run","--rm",
  "--entrypoint","python"
)

if ($envFileAbs) {
  $step2Args += @("--env-file", $envFileAbs)
}

$step2Args += @(
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

if ($UseSigfree) { $step2Args += "--use-sig-free" }
if ($OverwriteStl) { $step2Args += "--overwrite" }
if ($AsciiStl)     { $step2Args += "--ascii-stl" }
if ($Strict)       { $step2Args += "--strict" }
if ($StrictSignatureCollisions) { $step2Args += "--strict-signature-collisions" }

Run-Docker $step2Args

$manifestJson = Join-Path $OutDir "stl_manifest.json"
if (-not (Test-Path -LiteralPath $manifestJson)) {
  throw "Missing output: $manifestJson"
}

$manifestObj = Get-Content -LiteralPath $manifestJson -Raw | ConvertFrom-Json

$exported = [int]$manifestObj.meta.counts.exported
$skipped  = [int]$manifestObj.meta.counts.skipped_existing

# STLs available means: either exported now OR already existed and were matched
$stlsAvailable = $exported + $skipped

# Count any items that actually have an stl_path set
$itemsWithStlPath = @(
  $manifestObj.items |
    Where-Object { $_.stl_path -and $_.stl_path.Trim().Length -gt 0 }
).Count

if (($stlsAvailable -le 0) -or ($itemsWithStlPath -le 0)) {
  Write-Host "Skipping chirality step: no STLs available (exported=$exported skipped_existing=$skipped itemsWithStlPath=$itemsWithStlPath)." -ForegroundColor Yellow
  $SkipChirality = $true
} else {
  Write-Host "Chirality step enabled: STLs available (exported=$exported skipped_existing=$skipped itemsWithStlPath=$itemsWithStlPath)." -ForegroundColor Cyan
}

Write-Host "`n[Step 3 DEV] Add chirality signatures to stl_manifest.json" -ForegroundColor Green

if (-not $SkipChirality) {
  $step3Args = @(
    "run","--rm",
    "--entrypoint","python"
  )

  if ($envFileAbs) {
    $step3Args += @("--env-file", $envFileAbs)
  }

  $step3Args += @(
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
  Write-Host "Skipping chirality step (--SkipChirality set or no STLs available)" -ForegroundColor Yellow
}

Write-Host "`nDone." -ForegroundColor Cyan
Write-Host "Outputs:" -ForegroundColor Cyan
Write-Host "  $manifestJson" -ForegroundColor Cyan
Write-Host "  $(Join-Path $OutDir 'stl')" -ForegroundColor Cyan
