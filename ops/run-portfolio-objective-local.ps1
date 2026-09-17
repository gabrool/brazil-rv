param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'research/src'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:TORCHINDUCTOR_CACHE_DIR = 'C:/Brazil-RV/.cache/ti'
$env:TRITON_CACHE_DIR = 'C:/Brazil-RV/.cache/tr'
function Invoke-Research([string[]]$Arguments) {
    & uv run --project C:/quant/b3-quant/research --no-sync python -u @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Objective command failed: $Arguments" }
}
Invoke-Research @('-m','brazil_rv.v2.portfolio_objective_training','run','--root',$Root)
foreach ($arm in @('TE_all','C6')) {
    foreach ($fold in @('F2','F6','F10','F14')) {
        Invoke-Research @('-m','brazil_rv.v2.portfolio_objective_readouts','evaluate','--root',$Root,'--arm',$arm,'--fold',$fold)
    }
}
Invoke-Research @('-m','brazil_rv.v2.portfolio_objective_readouts','summarize','--root',$Root)
$screen = Get-Content (Join-Path $Root 'screen_summary.json') -Raw | ConvertFrom-Json
if (@($screen.survivors.PSObject.Properties).Count -gt 0) {
    Invoke-Research @('-m','brazil_rv.v2.portfolio_objective_training','run','--root',$Root,'--confirmation')
    foreach ($arm in $screen.survivors.PSObject.Properties.Name) {
        foreach ($fold in @('F1','F3','F4','F5','F7','F8','F9','F11','F12','F13')) {
            Invoke-Research @('-m','brazil_rv.v2.portfolio_objective_readouts','evaluate','--root',$Root,'--arm',$arm,'--fold',$fold,'--confirmation')
        }
    }
    Invoke-Research @('-m','brazil_rv.v2.portfolio_objective_readouts','summarize','--root',$Root,'--confirmation')
    Invoke-Research @('-m','brazil_rv.v2.portfolio_objective_readouts','continuous','--root',$Root)
}
@{ status='financial_readouts_complete'; completed_at=(Get-Date).ToUniversalTime().ToString('o'); review_and_recovery_remaining=$true } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root 'worker_complete.json') -Encoding utf8
