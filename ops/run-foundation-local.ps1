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
    if ($LASTEXITCODE -ne 0) { throw "Foundation command failed: $Arguments" }
}
Invoke-Research @('-m','brazil_rv.v2.foundation_program','inventory','--root',$Root)
Invoke-Research @('-m','brazil_rv.v2.foundation_program','run','--root',$Root,'--smoke')
Invoke-Research @('-m','brazil_rv.v2.foundation_program','run','--root',$Root)
@{ status='input_wave_training_complete'; completed_at=(Get-Date).ToUniversalTime().ToString('o');
   remaining='Readouts and wave review, variance/structure, component/scaling, residual probe, confirmation and recovery' } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root 'worker_complete.json') -Encoding utf8
