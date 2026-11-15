param([switch]$Deploy, [switch]$Recommended)
$ErrorActionPreference = 'Stop'
$mode = '--check'
if ($Deploy) { $mode = '--deploy' }
if ($Recommended -and -not $Deploy) { throw '-Recommended requires -Deploy. Without it, no changes are made.' }
if ($Recommended) { $mode = '--deploy-recommended' }
$scriptPath = Join-Path $PSScriptRoot 'deploy_vps.sh'
$scriptText = [IO.File]::ReadAllText($scriptPath).Replace("`r`n", "`n")
$scriptText | ssh -o BatchMode=yes -o ConnectTimeout=15 sihan@91.98.132.183 "tr -d '\r' | bash -s -- $mode"
if ($LASTEXITCODE -ne 0) {
    throw "VPS deployment/check failed (exit $LASTEXITCODE). See output above."
}
