param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:TORCHINDUCTOR_CACHE_DIR = 'C:/Brazil-RV/.cache/ti'
$env:TRITON_CACHE_DIR = 'C:/Brazil-RV/.cache/tr'
$decisionRoot = (Get-Content docs/v2_decision_run.json -Raw | ConvertFrom-Json).root
function Invoke-Research([string[]]$Arguments) {
    & uv run --project research python -u @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Research command failed: $Arguments" }
}
if ($Smoke) {
    Invoke-Research @('-m', 'brazil_rv.v2.objective_program', 'run-local', '--root', $decisionRoot, '--smoke')
    Invoke-Research @('-m', 'brazil_rv.v2.objective_program', 'accept-smoke', '--root', $decisionRoot)
    exit
}
Invoke-Research @('-m', 'brazil_rv.v2.objective_program', 'run-local', '--root', $decisionRoot)
foreach ($fold in @('F2', 'F6', 'F10', 'F14')) {
    Invoke-Research @('-m', 'brazil_rv.v2.objective_readouts', 'evaluate', '--root', $decisionRoot, '--fold', $fold)
}
Invoke-Research @('-m', 'brazil_rv.v2.objective_readouts', 'summarize', '--root', $decisionRoot)
$screen = Get-Content (Join-Path $decisionRoot 'phase3/screen_summary.json') -Raw | ConvertFrom-Json
if ($screen.survivors.Count -gt 0) {
    Invoke-Research @('-m', 'brazil_rv.v2.objective_readouts', 'mappings', '--root', $decisionRoot, '--confirmation')
    Invoke-Research @('-m', 'brazil_rv.v2.objective_program', 'run-local', '--root', $decisionRoot, '--confirmation')
    foreach ($fold in @('F1', 'F3', 'F4', 'F5', 'F7', 'F8', 'F9', 'F11', 'F12', 'F13')) {
        Invoke-Research @('-m', 'brazil_rv.v2.objective_readouts', 'evaluate', '--root', $decisionRoot, '--fold', $fold, '--confirmation')
    }
    Invoke-Research @('-m', 'brazil_rv.v2.objective_readouts', 'summarize', '--root', $decisionRoot, '--confirmation')
    foreach ($arm in $screen.survivors) {
        Invoke-Research @('-m', 'brazil_rv.v2.objective_readouts', 'continuous', '--root', $decisionRoot, '--arm', $arm)
    }
}
