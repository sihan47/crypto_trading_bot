param([switch]$Deploy)
$ErrorActionPreference = 'Stop'
$mode = '--check'
if ($Deploy) { $mode = '--deploy' }
$scriptPath = Join-Path $PSScriptRoot 'deploy_vps.sh'
$scriptText = [IO.File]::ReadAllText($scriptPath).Replace("`r`n", "`n")
$scriptText | ssh -o BatchMode=yes -o ConnectTimeout=15 sihan@91.98.132.183 "bash -s -- $mode"
if ($LASTEXITCODE -ne 0) {
    throw "VPS deployment/check failed (exit $LASTEXITCODE). See output above."
}
