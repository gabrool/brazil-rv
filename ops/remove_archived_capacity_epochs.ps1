param([Parameter(Mandatory=$true)][ValidateSet('depth','width')][string]$Wave)
$ErrorActionPreference = 'Stop'
$project = Split-Path $PSScriptRoot -Parent
$run = Get-Content -LiteralPath (Join-Path $project 'docs/v2_economic_data_scaling_run.json') -Raw | ConvertFrom-Json
$recoveryRef = $run.("stage_d_${Wave}_recovery")
$recovery = Get-Content -LiteralPath $recoveryRef.path -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $recovery.archive.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $recovery.archive.sha256) { throw 'Archive hash differs' }
$logical = Split-Path $run.("stage_d_${Wave}_plan").path -Parent
if ($Wave -eq 'depth') {
    $relocation = Get-Content -LiteralPath (Join-Path $run.root 'capacity_lstm_storage/report.json') -Raw | ConvertFrom-Json
    if ($logical -ne $relocation.source) { throw 'Unexpected completed depth root' }
    $physical = [IO.Path]::GetFullPath($relocation.target).TrimEnd('\')
    if ([IO.Path]::GetFullPath((Get-Item -LiteralPath $logical).Target).TrimEnd('\') -ne $physical) { throw 'Depth junction differs' }
} else {
    $physical = [IO.Path]::GetFullPath($logical).TrimEnd('\')
    if (-not $physical.StartsWith([IO.Path]::GetFullPath($run.root).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase) -or ((Get-Item -LiteralPath $physical).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unexpected completed width root' }
}
$fitsRoot = [IO.Path]::GetFullPath((Join-Path $physical 'fits')).TrimEnd('\')
$auditRoot = Join-Path $run.root 'scaling_investigation/storage'
$output = Join-Path $auditRoot "retired_${Wave}_epochs.json"
if (Test-Path -LiteralPath $output) { throw 'Cleanup already recorded' }
$files = @(foreach ($arm in (Get-ChildItem -LiteralPath $fitsRoot -Directory)) {
    if (-not ($arm.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        Get-ChildItem -LiteralPath $arm.FullName -Recurse -File -Filter '*.pt' | Where-Object { $_.Directory.Name -eq 'epochs' }
    }
})
$zip = [IO.Compression.ZipFile]::OpenRead($recovery.archive.path)
function Read-ZipJson($archive, $name) {
    $reader = [IO.StreamReader]::new($archive.GetEntry($name).Open())
    try { return ($reader.ReadToEnd() | ConvertFrom-Json -AsHashtable) } finally { $reader.Dispose() }
}
$members = Read-ZipJson $zip 'members.json'
$aliases = Read-ZipJson $zip 'aliases.json'
$records = @()
$before = @(Get-PSDrive -Name C,D | Select-Object Name,Free)
try {
    foreach ($file in $files) {
        $target = [IO.Path]::GetFullPath($file.FullName)
        if (-not $target.StartsWith($fitsRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or $file.Directory.Name -ne 'epochs' -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Deletion outside intermediate depth epochs' }
        $relative = [IO.Path]::GetRelativePath($physical,$target).Replace('\','/')
        $member = "evidence/capacity_${Wave}/"+$relative
        $record = $members[$member]
        if (-not $record) { throw "Not directly recoverable from depth archive: $member" }
        $digest = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -ne $record.sha256 -or $file.Length -ne $record.bytes) { throw 'Live intermediate differs from archive receipt' }
        $archivedName = if ($aliases.ContainsKey($member)) { $aliases[$member] } else { $member }
        $entry = $zip.GetEntry($archivedName)
        if ($entry.Length -ne $file.Length) { throw 'Archived size differs' }
        $stream = $entry.Open()
        $hasher = [Security.Cryptography.SHA256]::Create()
        try { $archiveDigest = [Convert]::ToHexString($hasher.ComputeHash($stream)).ToLowerInvariant() } finally { $stream.Dispose(); $hasher.Dispose() }
        if ($archiveDigest -ne $digest) { throw 'Archived decoded bytes differ' }
        $records += [pscustomobject]@{path=$target; member=$member; archived_member=$archivedName; bytes=$file.Length; sha256=$digest}
    }
    $plan = [ordered]@{archive=$recovery.archive; files=$records; scope="User-authorized removal of redundant intermediate epoch snapshots from native (non-junction) fit directories of the completed, rejected $Wave wave only. Every live file and decompressed archived member hash matches before deletion. Selected checkpoints, EMA checkpoints, fits/history/diagnostics, forecasts/books, original foundation fits, stores and sources remain present."}
    [IO.File]::WriteAllText((Join-Path $auditRoot "retired_${Wave}_epochs_plan.json"),($plan|ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
    foreach ($record in $records) {
        $target = [IO.Path]::GetFullPath($record.path)
        if (-not $target.StartsWith($fitsRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or [IO.Path]::GetFileName([IO.Path]::GetDirectoryName($target)) -ne 'epochs') { throw 'Deletion target changed scope' }
        Remove-Item -LiteralPath $target -Force
    }
    $report = [ordered]@{status='redundant_epoch_copies_removed'; files=$records.Count; logical_bytes=($records|Measure-Object bytes -Sum).Sum; archive=$recovery.archive; plan="retired_${Wave}_epochs_plan.json"; before=$before; after=@(Get-PSDrive -Name C,D | Select-Object Name,Free); restore='Restore each archived_member into the corresponding path and verify sha256 before requesting an intermediate checkpoint. Selected checkpoints remain directly available.'}
    [IO.File]::WriteAllText($output,($report|ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
    $report|ConvertTo-Json -Depth 7
} finally { $zip.Dispose() }
