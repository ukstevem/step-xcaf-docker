param(
  [string]$Name = "step4-ui"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

& docker rm -f $Name *>$null
Write-Host ("Stopped: " + $Name) -ForegroundColor Cyan
