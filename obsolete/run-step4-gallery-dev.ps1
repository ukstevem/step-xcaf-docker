Param(
  [Parameter(ValueFromRemainingArguments=$true)]
  [string[]]$Args
)

$ErrorActionPreference = "Stop"

$repoAbs = (Get-Location).Path
$outAbs  = Join-Path $repoAbs "out"
$inAbs   = Join-Path $repoAbs "in"

if (-not (Test-Path $outAbs)) { New-Item -ItemType Directory -Force -Path $outAbs | Out-Null }
if (-not (Test-Path $inAbs))  { New-Item -ItemType Directory -Force -Path $inAbs  | Out-Null }

$envFile  = Join-Path $repoAbs ".env"
$envLocal = Join-Path $repoAbs ".env.local"

$envArgs = @()
if (Test-Path $envFile)  { $envArgs += @("--env-file", $envFile) }
if (Test-Path $envLocal) { $envArgs += @("--env-file", $envLocal) }

docker run --rm `
  @envArgs `
  -v "${repoAbs}:/app" `
  -v "${inAbs}:/in" `
  -v "${outAbs}:/out" `
  -w /app `
  step-xcaf-docker `
  /app/step4_make_gallery.py @Args

if ($LASTEXITCODE -ne 0) { throw "docker failed with exit code $LASTEXITCODE" }
