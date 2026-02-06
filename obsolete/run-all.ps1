param(
  [Parameter(Mandatory = $true)]
  [string]$StepPath,

  [string]$ImageName = "step-xcaf-docker",

  # Fixed container name (avoids random names)
  [string]$ContainerName = "step-xcaf-run",

  [string]$OutDir = "",

  [int]$ThumbSize = 512,

  # chirality tolerance used by add_chirality_to_manifest.py
  [double]$ChiralTol = 0.5,

  # Export leaf ancillaries from occurrences (plates-on-beams etc.)
  [switch]$ExportLeafAncillaries,

  # Minimum relative volume for ancillaries (filters tiny junk)
  [double]$MinAncRelVol = 0.0001,

  # Run grouping steps (group_ancillaries)
  [switch]$DoGrouping,

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
$repoAbs = (Resolve-Path -LiteralPath $repoRoot).Path

# Copy STEP into ./in (skip if already there)
$leaf      = Split-Path $stepAbs -Leaf
$localStep = Join-Path $inDir $leaf
if ($stepAbs -ne $localStep) {
  Copy-Item -LiteralPath $stepAbs -Destination $localStep -Force
} else {
  Write-Host "STEP already in ./in, skipping copy: $localStep" -ForegroundColor DarkGray
}

$stepIn = "/in/$leaf"

Write-Host ("Flags: ExportLeafAncillaries={0}  DoGrouping={1}" -f $ExportLeafAncillaries.IsPresent, $DoGrouping.IsPresent) -ForegroundColor DarkGray

if ($BuildNoCache) {
  Write-Host "Building docker image (no-cache): $ImageName" -ForegroundColor Cyan
  Run-Docker @("build","--no-cache","-t",$ImageName,$repoRoot)
}

# sanity
Run-Docker @("image","inspect",$ImageName) | Out-Null

# STL export command: choose normal vs leaf-ancillary export
$exportCmd = ""
if ($ExportLeafAncillaries) {
  # CLI pattern: ... /out 0 --export-leaf-ancillaries ...
  $exportCmd = "python -u /app/export_stl_xcaf.py $stepIn /out 0 --export-leaf-ancillaries --min-anc-rel-vol $MinAncRelVol"
} else {
  $exportCmd = "python -u /app/export_stl_xcaf.py $stepIn /out"
}

# Chirality step: only runs if file exists in image
# IMPORTANT: pass the manifest FILE path (not /out)
$chiralityCmd = "if [ -f /app/add_chirality_to_manifest.py ]; then python -u /app/add_chirality_to_manifest.py /out/stl_manifest.json $ChiralTol; else echo 'add_chirality_to_manifest.py missing; skipping chirality step'; fi"

# Grouping step: only if requested AND the exporter produced stl_manifest_ancillary.json
# We only require ancillary_groups.json (no CSVs).
$groupCmd = "echo 'Grouping skipped'"
if ($DoGrouping) {
  $groupCmd = "if [ -f /out/stl_manifest_ancillary.json ]; then python -u /app/group_ancillaries.py /out 0.5; else echo 'No stl_manifest_ancillary.json; skipping ancillary grouping'; fi"
}

# Order: extract -> export -> chirality -> thumbs -> BOM -> grouping -> UI bundle
# (UI bundle last so it can embed ancillary_groups.json if you add that patch)
$cmdParts = @(
  "python -u /app/read_step_xcaf.py $stepIn /out",
  $exportCmd,
  $chiralityCmd,
  "PYVISTA_OFF_SCREEN=true xvfb-run -a -s '-screen 0 1024x768x24' python -u /app/render_thumbnails.py /out $ThumbSize",
  "python -u /app/build_bom_from_xcaf.py /out",
  $groupCmd,
  "python -u /app/build_ui_bundle.py /out"
)

$cmd = ($cmdParts -join " && ")

Write-Host "`n[PIPELINE] Running all steps in one container" -ForegroundColor Green

# Remove stale named container if it exists (avoid PowerShell-native error noise)
# Using cmd.exe avoids the "NativeCommandError" behaviour from docker.exe stderr.
cmd /c "docker rm -f $ContainerName >NUL 2>NUL"

Run-Docker @(
  "run","--rm",
  "--name",$ContainerName,
  "--entrypoint","sh",
  "-v","${inAbs}:/in:ro",
  "-v","${outAbs}:/out",
  "-v","${repoAbs}:/work:ro",
  $ImageName,
  "-lc",
  $cmd
)

# Quick output checks
if (-not (Test-Path -LiteralPath (Join-Path $OutDir "xcaf_instances.json"))) {
  throw "Missing output: $OutDir\xcaf_instances.json"
}
if (-not (Test-Path -LiteralPath (Join-Path $OutDir "stl_manifest.json"))) {
  throw "Missing output: $OutDir\stl_manifest.json"
}

# Count PNGs
$pngDir = Join-Path $OutDir "png"
$pngCount = 0
if (Test-Path -LiteralPath $pngDir) {
  $pngCount = (Get-ChildItem -LiteralPath $pngDir -Filter *.png -File -ErrorAction SilentlyContinue | Measure-Object).Count
}
if ($pngCount -eq 0) {
  throw "No PNG thumbnails produced in: $pngDir"
}

Write-Host "`nDone." -ForegroundColor Cyan
Write-Host "Outputs:" -ForegroundColor Cyan
Write-Host "  $OutDir\xcaf_instances.json"
Write-Host "  $OutDir\stl_manifest.json"
Write-Host "  $OutDir\stl\*.stl"
Write-Host "  $OutDir\png\*.png  (count: $pngCount)"
Write-Host "  $OutDir\bom_from_xcaf_all.csv"
Write-Host "  $OutDir\bom_from_xcaf_leaf.csv"
Write-Host "  $OutDir\ui_bundle.json"

if ($ExportLeafAncillaries) {
  Write-Host "  $OutDir\stl_manifest_ancillary.json / stl_manifest_all.json"
}
if ($DoGrouping) {
  Write-Host "  $OutDir\ancillary_groups.json"
}
