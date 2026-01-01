param(
  [Parameter(Mandatory = $true)]
  [string]$StepPath,

  [string]$ImageName = "step-xcaf-docker",

  [string]$OutDir = "",

  [int]$ThumbSize = 512,

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

if ($BuildNoCache) {
  Write-Host "Building docker image (no-cache): $ImageName" -ForegroundColor Cyan
  Run-Docker @("build","--no-cache","-t",$ImageName,$repoRoot)
}

Run-Docker @("image","inspect",$ImageName) | Out-Null

Write-Host "`n[1/4] XCAF extract -> xcaf_instances.json" -ForegroundColor Green
Run-Docker @(
  "run","--rm",
  "-v","${inAbs}:/in:ro",
  "-v","${outAbs}:/out",
  $ImageName,
  "-u","/app/read_step_xcaf.py",
  $stepIn, "/out"
)
if (-not (Test-Path -LiteralPath (Join-Path $OutDir "xcaf_instances.json"))) {
  throw "Missing output: $OutDir\xcaf_instances.json"
}

Write-Host "`n[2/4] Export unique STLs -> out/stl + stl_manifest.json" -ForegroundColor Green
Run-Docker @(
  "run","--rm",
  "-v","${inAbs}:/in:ro",
  "-v","${outAbs}:/out",
  $ImageName,
  "-u","/app/export_stl_xcaf.py",
  $stepIn, "/out"
)
if (-not (Test-Path -LiteralPath (Join-Path $OutDir "stl_manifest.json"))) {
  throw "Missing output: $OutDir\stl_manifest.json"
}

Write-Host "`n[3/4] Render thumbnails -> out/png + update stl_manifest.json" -ForegroundColor Green
Run-Docker @(
  "run","--rm",
  "--entrypoint","sh",
  "-v","${outAbs}:/out",
  $ImageName,
  "-lc",
  "PYVISTA_OFF_SCREEN=true xvfb-run -a -s '-screen 0 1024x768x24' python -u /app/render_thumbnails.py /out $ThumbSize"
)

# Check at least one PNG exists
$pngDir = Join-Path $OutDir "png"
$pngCount = 0
if (Test-Path -LiteralPath $pngDir) {
  $pngCount = (Get-ChildItem -LiteralPath $pngDir -Filter *.png -File -ErrorAction SilentlyContinue | Measure-Object).Count
}
if ($pngCount -eq 0) {
  throw "No PNG thumbnails produced in: $pngDir"
}

Write-Host "`n[4/4] Build BOM CSVs -> bom_from_xcaf_*.csv" -ForegroundColor Green
Run-Docker @(
  "run","--rm",
  "-v","${outAbs}:/out",
  $ImageName,
  "-u","/app/build_bom_from_xcaf.py",
  "/out"
)

if (-not (Test-Path -LiteralPath (Join-Path $OutDir "bom_from_xcaf_all.csv"))) {
  throw "Missing output: $OutDir\bom_from_xcaf_all.csv"
}
if (-not (Test-Path -LiteralPath (Join-Path $OutDir "bom_from_xcaf_leaf.csv"))) {
  throw "Missing output: $OutDir\bom_from_xcaf_leaf.csv"
}

Write-Host "`nDone." -ForegroundColor Cyan
Write-Host "Outputs:" -ForegroundColor Cyan
Write-Host "  $OutDir\xcaf_instances.json"
Write-Host "  $OutDir\stl_manifest.json"
Write-Host "  $OutDir\stl\*.stl"
Write-Host "  $OutDir\png\*.png  (count: $pngCount)"
Write-Host "  $OutDir\bom_from_xcaf_all.csv"
Write-Host "  $OutDir\bom_from_xcaf_leaf.csv"
