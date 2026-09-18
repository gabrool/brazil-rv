param([Parameter(Mandatory=$true)][string]$Root)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'research/src'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& uv run --project C:/quant/b3-quant/research --no-sync python -u -m brazil_rv.v2.foundation_averaging --root $Root
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint average preparation failed' }
& uv run --project C:/quant/b3-quant/research --no-sync python -u -m brazil_rv.v2.foundation_readouts ensemble --root $Root
if ($LASTEXITCODE -ne 0) { throw 'Ensemble diagnostics failed' }
