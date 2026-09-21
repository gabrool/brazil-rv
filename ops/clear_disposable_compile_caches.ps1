$ErrorActionPreference = 'Stop'
$auditRoot = 'D:/quant-data/b3/processed/model_runs/v2_economic_data_scaling_20260919/scaling_investigation/storage'
New-Item -ItemType Directory -Path $auditRoot -Force | Out-Null
$workers = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv|compact' })
if ($workers.Count) { throw 'A research or compression worker is active' }
$roots = @('C:/quant-data/b3/interim/ti', 'C:/Brazil-RV/.cache/tr')
$before = @(Get-PSDrive -Name C,D | Select-Object Name,Free)
$inventory = @()
foreach ($cacheRoot in $roots) {
    $resolved = [IO.Path]::GetFullPath($cacheRoot).TrimEnd('\')
    $item = Get-Item -LiteralPath $resolved
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Cache root is a reparse point' }
    $entries = @(Get-ChildItem -LiteralPath $resolved -Recurse -Force)
    if (@($entries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Nested reparse point' }
    $inventory += @($entries | Where-Object { -not $_.PSIsContainer } | Select-Object FullName,Length,LastWriteTimeUtc)
}
$plan = [ordered]@{ scope='User-authorized removal of reproducible compiler caches only. Original sources, stores, checkpoints, forecasts, books and archived compiler-failure evidence are preserved.'; roots=$roots; files=$inventory; before=$before }
[IO.File]::WriteAllText((Join-Path $auditRoot 'cache_cleanup_plan.json'), ($plan | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
foreach ($cacheRoot in $roots) {
    $resolved = [IO.Path]::GetFullPath($cacheRoot).TrimEnd('\')
    foreach ($child in Get-ChildItem -LiteralPath $resolved -Force) {
        $target = [IO.Path]::GetFullPath($child.FullName)
        if (-not $target.StartsWith($resolved+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Deletion outside explicit cache root' }
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}
$report = [ordered]@{ status='removed_disposable_caches'; files=$inventory.Count; logical_bytes=($inventory | Measure-Object Length -Sum).Sum; before=$before; after=@(Get-PSDrive -Name C,D | Select-Object Name,Free); preserved_root_compression=$true; source_or_result_deletion=$false }
[IO.File]::WriteAllText((Join-Path $auditRoot 'cache_cleanup_report.json'), ($report | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
$report | ConvertTo-Json -Depth 5
