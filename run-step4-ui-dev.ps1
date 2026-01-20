param(
  [string]$ImageName = "step-xcaf-docker",
  [string]$OutDir = "",
  [int]$Port = 8004,
  [string]$Name = "step4-ui"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Ensure-Dir([string]$p) {
  if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
}

function Run-Docker([string[]]$DockerArgs) {
  Write-Host ("docker " + ($DockerArgs -join " ")) -ForegroundColor DarkGray
  & docker @DockerArgs
  if ($LASTEXITCODE -ne 0) { throw ("docker failed with exit code " + $LASTEXITCODE) }
}

$repoRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = Join-Path $repoRoot "out" }
Ensure-Dir $OutDir

$repoAbs = (Resolve-Path -LiteralPath $repoRoot).Path
$outAbs  = (Resolve-Path -LiteralPath $OutDir).Path

# UI must exist
$uiDir = Join-Path $OutDir "review_ui"
if (-not (Test-Path -LiteralPath $uiDir)) {
  throw ("Missing " + $uiDir + " - run .\run-step4-ui-build-dev.ps1 first")
}

# Remove any previous container with same name (ignore if missing)
try {
  & docker rm -f $Name | Out-Null
} catch {
  # ignore "No such container"
}

Write-Host ""
Write-Host ("[Step 4 UI DEV] starting container '" + $Name + "' on http://localhost:" + $Port) -ForegroundColor Green

$scriptArgs = @(
  "run","-d",
  "--name",$Name,
  "--entrypoint","python",
  "-p", ($Port.ToString() + ":" + $Port.ToString()),
  "-v", ($outAbs + ":/out"),
  "-v", ($repoAbs + ":/app"),
  "-w","/app",
  $ImageName,
  "-u","/app/step4_multibody_ui_server.py",
  "--outdir","/out",
  "--port",$Port.ToString(),
  "--bind","0.0.0.0"
)

Run-Docker $scriptArgs

Write-Host ""
Write-Host "Container is running. Useful commands:" -ForegroundColor Cyan
Write-Host ("  docker logs -f " + $Name)
Write-Host ("  docker rm -f " + $Name)
Write-Host ""
Write-Host ("Open: http://localhost:" + $Port + "/") -ForegroundColor Cyan
