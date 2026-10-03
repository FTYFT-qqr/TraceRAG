$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$profileDir = Join-Path $env:TEMP 'tracerag-streamlit-profile'
$uiScript = Join-Path $projectRoot 'app\ui.py'

if (-not (Test-Path -LiteralPath $uiScript)) {
    throw "Streamlit app not found: $uiScript"
}

New-Item -ItemType Directory -Path $profileDir -Force | Out-Null

$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = 'python'
}

$previousUserProfile = $env:USERPROFILE
$exitCode = 0
try {
    $env:USERPROFILE = $profileDir
    Push-Location $projectRoot
    try {
        & $pythonPath -m streamlit run $uiScript --server.address 127.0.0.1 --server.headless=true --server.showEmailPrompt=false --browser.gatherUsageStats=false
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($null -eq $previousUserProfile) {
        Remove-Item Env:USERPROFILE -ErrorAction SilentlyContinue
    }
    else {
        $env:USERPROFILE = $previousUserProfile
    }
}

exit $exitCode
