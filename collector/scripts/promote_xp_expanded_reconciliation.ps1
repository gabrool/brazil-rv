param(
    [string]$ProjectRoot = "C:\quant\b3-quant\collector",
    [string]$ReconciliationBase = "C:\quant-data\b3\interim\m1_reconciliation",
    [string]$ExpandedReconciliationDir = ""
)

$ErrorActionPreference = "Stop"

function Normalize-Text([object]$Value) {
    if ($null -eq $Value) {
        return ""
    }
    return ([string]$Value).Trim()
}

if (-not [string]::IsNullOrWhiteSpace($ExpandedReconciliationDir)) {
    $ResolvedExpanded = Resolve-Path -LiteralPath $ExpandedReconciliationDir -ErrorAction Stop
    $V2ExpandedOut = $ResolvedExpanded.Path
    Write-Host "Using explicitly supplied expanded reconciliation:"
    Write-Host $V2ExpandedOut
}
else {
    Write-Host "Locating the newest expanded XP/COTAHIST v2 reconciliation ..."
    $ExpandedCandidate = Get-ChildItem `
        -LiteralPath $ReconciliationBase `
        -Directory `
        -Filter "xp_cotahist_v2_expanded_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($null -eq $ExpandedCandidate) {
        throw "No xp_cotahist_v2_expanded_* directory was found under $ReconciliationBase"
    }

    $V2ExpandedOut = $ExpandedCandidate.FullName
    Write-Host $V2ExpandedOut
}

$SummaryPath = Join-Path $V2ExpandedOut "summary.json"
$RecoveriesPath = Join-Path $V2ExpandedOut "missing_high_confidence_recoveries.csv"
$ReviewPath = Join-Path $V2ExpandedOut "missing_review.csv"
$UnresolvedPath = Join-Path $V2ExpandedOut "missing_unresolved.csv"
$NoResearchPath = Join-Path $V2ExpandedOut "missing_no_research_days.csv"
$ProvisionalPath = Join-Path $V2ExpandedOut "provisional_source_assignments.csv"

foreach ($RequiredPath in @(
    $SummaryPath,
    $RecoveriesPath,
    $ReviewPath,
    $UnresolvedPath,
    $NoResearchPath,
    $ProvisionalPath
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required reconciliation output is missing: $RequiredPath"
    }
}

$Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json

$Expected = [ordered]@{
    direct_isin_confirmed_count = 143
    missing_recovered_high_count = 15
    missing_review_count = 1
    missing_unresolved_count = 37
    missing_no_research_days_count = 2
    provisional_assignment_count = 158
}

foreach ($Key in $Expected.Keys) {
    $Actual = [int]$Summary.$Key
    if ($Actual -ne [int]$Expected[$Key]) {
        throw "Unexpected $Key. Expected $($Expected[$Key]); received $Actual."
    }
}

# @() is deliberate: Import-Csv returns a scalar PSCustomObject for a one-row CSV.
$Recoveries = @(Import-Csv -LiteralPath $RecoveriesPath)
$Review = @(Import-Csv -LiteralPath $ReviewPath)
$Unresolved = @(Import-Csv -LiteralPath $UnresolvedPath)
$NoResearch = @(Import-Csv -LiteralPath $NoResearchPath)
$Provisional = @(Import-Csv -LiteralPath $ProvisionalPath)

if ($Recoveries.Count -ne 15) {
    throw "missing_high_confidence_recoveries.csv contains $($Recoveries.Count) rows rather than 15."
}
if ($Provisional.Count -ne 158) {
    throw "provisional_source_assignments.csv contains $($Provisional.Count) rows rather than 158."
}
if ($Unresolved.Count -ne 37) {
    throw "missing_unresolved.csv contains $($Unresolved.Count) rows rather than 37."
}
if ($NoResearch.Count -ne 2) {
    throw "missing_no_research_days.csv contains $($NoResearch.Count) rows rather than 2."
}

$RequiredNewRecoveries = [ordered]@{
    "GUAR3" = "RIAA3"
    "JBSS3" = "JBSS32"
    "WIZS3" = "WIZC3"
}

foreach ($HistoricalTicker in $RequiredNewRecoveries.Keys) {
    $ExpectedSource = $RequiredNewRecoveries[$HistoricalTicker]
    $Row = $Recoveries |
        Where-Object {
            (Normalize-Text $_.latest_ticker).ToUpperInvariant() -eq $HistoricalTicker -and
            (Normalize-Text $_.xp_symbol).ToUpperInvariant() -eq $ExpectedSource
        } |
        Select-Object -First 1

    if ($null -eq $Row) {
        throw "Required recovery was not found: $HistoricalTicker -> $ExpectedSource"
    }

    $Checks = @(
        ([double]$Row.overlap_coverage -ge 0.80),
        ([double]$Row.shape_match_5ticks -ge 0.90),
        ([double]$Row.ratio_stable_50bp -ge 0.90),
        ([double]$Row.invariant_match_score -ge 0.80),
        ([double]$Row.score_margin -ge 0.08)
    )

    if ($Checks -contains $false) {
        throw "Recovery failed the acceptance thresholds: $HistoricalTicker -> $ExpectedSource"
    }
}

if ($Review.Count -ne 1) {
    throw "Expected exactly one review row; found $($Review.Count)."
}

$ReviewTicker = (Normalize-Text $Review[0].latest_ticker).ToUpperInvariant()
$ReviewSource = (Normalize-Text $Review[0].xp_symbol).ToUpperInvariant()
$ReviewStatus = (Normalize-Text $Review[0].recovery_status).ToUpperInvariant()

Write-Host "Review case found: $ReviewTicker -> $ReviewSource [$ReviewStatus]"

if ($ReviewTicker -ne "CPLE6" -or $ReviewSource -ne "CPLE3") {
    throw "The sole review case was expected to be CPLE6 -> CPLE3; found $ReviewTicker -> $ReviewSource."
}

$Stamp = Get-Date -Format "yyyy-MM-dd_HHmmssfff"
$DecisionBase = Join-Path $ReconciliationBase "accepted"
$DecisionOut = Join-Path $DecisionBase "xp_v1_158_$Stamp"
New-Item -ItemType Directory -Force -Path $DecisionOut | Out-Null

$ReviewedAtUtc = [DateTime]::UtcNow.ToString("o")

$AcceptedRows = foreach ($Row in $Provisional) {
    $Ticker = (Normalize-Text $Row.latest_ticker).ToUpperInvariant()
    $XpSymbol = (Normalize-Text $Row.xp_symbol).ToUpperInvariant()
    $AssignmentType = Normalize-Text $Row.source_assignment_type
    $DecisionNote = ""

    if ($AssignmentType -eq "DIRECT_ISIN") {
        $DecisionNote = "Accepted by exact XP-catalogue/COTAHIST ISIN equality. Use the XP file only on COTAHIST dates belonging to this security_id."
    }
    elseif ($Ticker -eq "GUAR3" -and $XpSymbol -eq "RIAA3") {
        $DecisionNote = "Accepted historical predecessor segment. Use RIAA3 source only on GUAR3 COTAHIST dates through 2026-02-04."
    }
    elseif ($Ticker -eq "JBSS3" -and $XpSymbol -eq "JBSS32") {
        $DecisionNote = "Accepted historical predecessor segment. Use JBSS32 source only on JBSS3 COTAHIST dates through 2025-06-06; do not assign later BDR dates to JBSS3."
    }
    elseif ($Ticker -eq "WIZS3" -and $XpSymbol -eq "WIZC3") {
        $DecisionNote = "Accepted historical ticker-predecessor segment. Use WIZC3 source only on WIZS3 COTAHIST dates through 2023-02-08."
    }
    else {
        $DecisionNote = "Accepted high-confidence relabelled-history recovery. Use the source only on COTAHIST dates belonging to this security_id."
    }

    $Row | Select-Object `
        *,
        @{Name = "manual_decision"; Expression = { "ACCEPTED" }},
        @{Name = "manual_decision_version"; Expression = { "XP_V1_158" }},
        @{Name = "reviewed_at_utc"; Expression = { $ReviewedAtUtc }},
        @{Name = "normalization_rule"; Expression = { "FILTER_TO_COTAHIST_SECURITY_DATES" }},
        @{Name = "manual_decision_note"; Expression = { $DecisionNote }}
}

$AcceptedCsv = Join-Path $DecisionOut "xp_accepted_source_assignments_v1.csv"
$AcceptedParquet = Join-Path $DecisionOut "xp_accepted_source_assignments_v1.parquet"

$AcceptedRows |
    Export-Csv `
        -LiteralPath $AcceptedCsv `
        -NoTypeInformation `
        -Encoding utf8

$PythonCode = @'
import sys
import pandas as pd

csv_path, parquet_path = sys.argv[1], sys.argv[2]
frame = pd.read_csv(csv_path, low_memory=False)
frame.to_parquet(parquet_path, index=False, compression="zstd")
print(f"Wrote {len(frame):,} accepted assignments to {parquet_path}")
'@

Set-Location $ProjectRoot
$PythonCode | uv run python - "$AcceptedCsv" "$AcceptedParquet"

Copy-Item -LiteralPath $SummaryPath -Destination (Join-Path $DecisionOut "expanded_summary.json")
Copy-Item -LiteralPath $RecoveriesPath -Destination (Join-Path $DecisionOut "accepted_recovery_evidence.csv")
Copy-Item -LiteralPath $ReviewPath -Destination (Join-Path $DecisionOut "pending_review.csv")
Copy-Item -LiteralPath $UnresolvedPath -Destination (Join-Path $DecisionOut "pending_unresolved.csv")
Copy-Item -LiteralPath $NoResearchPath -Destination (Join-Path $DecisionOut "no_research_days.csv")

$DecisionManifest = [ordered]@{
    decision_version = "XP_V1_158"
    created_at_utc = $ReviewedAtUtc
    expanded_reconciliation_dir = $V2ExpandedOut
    accepted_assignment_count = 158
    direct_isin_count = 143
    recovered_relabelled_count = 15
    pending_review_count = 1
    pending_unresolved_count = 37
    no_research_days_count = 2
    explicitly_accepted_new_recoveries = @(
        "GUAR3 -> RIAA3",
        "JBSS3 -> JBSS32",
        "WIZS3 -> WIZC3"
    )
    explicitly_rejected_review_mapping = "CPLE6 -> CPLE3"
    normalization_rule = "An XP source may be used only on dates present in COTAHIST for the assigned permanent security_id."
    accepted_csv = $AcceptedCsv
    accepted_parquet = $AcceptedParquet
}

$DecisionManifest |
    ConvertTo-Json -Depth 6 |
    Set-Content `
        -LiteralPath (Join-Path $DecisionOut "decision_manifest.json") `
        -Encoding utf8

$CanonicalPointer = Join-Path $ReconciliationBase "xp_cotahist_v2_canonical_path.txt"
if (Test-Path -LiteralPath $CanonicalPointer) {
    Copy-Item `
        -LiteralPath $CanonicalPointer `
        -Destination (Join-Path $DecisionOut "previous_canonical_pointer.txt")
}

$V2ExpandedOut |
    Set-Content `
        -LiteralPath $CanonicalPointer `
        -Encoding ascii

$DecisionPointer = Join-Path $DecisionBase "xp_accepted_assignments_v1_path.txt"
$DecisionOut |
    Set-Content `
        -LiteralPath $DecisionPointer `
        -Encoding ascii

Write-Host ""
Write-Host "Expanded v2 reconciliation promoted to canonical:"
Write-Host $V2ExpandedOut
Write-Host ""
Write-Host "Accepted XP assignment package:"
Write-Host $DecisionOut
Write-Host ""
Write-Host "Accepted assignments: 158"
Write-Host "Pending review: CPLE6 -> CPLE3 (not accepted)"
Write-Host "Pending unresolved: 37"
