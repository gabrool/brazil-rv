[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$watcherPath = Join-Path $PSScriptRoot 'lambda-gh200.ps1'
. $watcherPath

$script:Passed = 0
$script:Failed = 0

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not $Condition) {
        throw $Message
    }
}

function Assert-Equal {
    param(
        [AllowNull()][object]$Actual,
        [AllowNull()][object]$Expected,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if ($Actual -cne $Expected) {
        throw "$Message Expected '$Expected', got '$Actual'."
    }
}

function Assert-Throws {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [Parameter(Mandatory = $true)][string]$Message
    )
    $threw = $false
    try {
        & $Action
    }
    catch {
        $threw = $true
    }
    if (-not $threw) {
        throw $Message
    }
}

function Invoke-Test {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Body
    )
    try {
        & $Body
        $script:Passed++
        Write-Host "PASS $Name"
    }
    catch {
        $script:Failed++
        Write-Host "FAIL $Name - $($_.Exception.Message)" -ForegroundColor Red
    }
}

function New-TestInstance {
    param(
        [string]$Id = 'instance-1',
        [string]$Status = 'booting'
    )
    return [pscustomobject]@{
        id = $Id
        name = 'brazil-rv-gh200'
        status = $Status
        ip = '198.51.100.2'
        region = [pscustomobject]@{ name = 'us-east-3' }
        instance_type = [pscustomobject]@{ name = 'gpu_1x_gh200' }
        tags = @(
            [pscustomobject]@{ key = 'project'; value = 'brazil-rv' },
            [pscustomobject]@{ key = 'purpose'; value = 'research-training' },
            [pscustomobject]@{ key = 'managed-by'; value = 'gh200-watcher' }
        )
        ssh_key_names = @('brazil-rv')
        file_system_mounts = @(
            [pscustomobject]@{
                file_system_id = 'fs-east3'
                mount_point = '/lambda/nfs/brazil-rv-east3'
            }
        )
    }
}

function New-TestMarker {
    param(
        [string]$InstanceId = 'instance-1',
        [string]$GitSha = ('a' * 40),
        [string]$BundleSha = ('b' * 64),
        [string]$BootstrapSha = ('c' * 64),
        [bool]$Passed = $true
    )
    return [pscustomobject][ordered]@{
        passed = $Passed
        instance_id = $InstanceId
        git_sha = $GitSha
        bundle_sha256 = $BundleSha
        bootstrap_sha256 = $BootstrapSha
        completed_at_utc = '2026-07-23T12:00:00Z'
        sanity_report_path = '/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/run/sanity_report.json'
    }
}

function New-ProcessResult {
    param(
        [int]$ExitCode,
        [bool]$TimedOut = $false
    )
    return [pscustomobject]@{
        ExitCode = $ExitCode
        TimedOut = $TimedOut
        StdOut = ''
        StdErr = ''
    }
}

Invoke-Test 'GH200 discovery requires one exact one-GPU candidate' {
    $types = [ordered]@{
        gpu_1x_gh200 = [pscustomobject]@{
            instance_type = [pscustomobject]@{
                name = 'gpu_1x_gh200'
                description = 'NVIDIA GH200 Grace Hopper'
                gpu_description = 'GH200'
                price_cents_per_hour = 199
                specs = [pscustomobject]@{ gpus = 1 }
            }
            regions_with_capacity_available = @(
                [pscustomobject]@{ name = 'us-east-3' }
            )
        }
        gpu_8x_gh200 = [pscustomobject]@{
            instance_type = [pscustomobject]@{
                name = 'gpu_8x_gh200'
                description = 'Eight NVIDIA GH200 GPUs'
                gpu_description = 'GH200'
                price_cents_per_hour = 1592
                specs = [pscustomobject]@{ gpus = 8 }
            }
            regions_with_capacity_available = @(
                [pscustomobject]@{ name = 'us-east-3' }
            )
        }
    }
    $found = Find-Gh200InstanceType $types
    Assert-Equal $found.Name 'gpu_1x_gh200' 'Wrong GH200 type.'
    Assert-True $found.Available 'GH200 should be available.'
}

Invoke-Test 'GH200 discovery fails closed on ambiguity' {
    $types = [ordered]@{}
    foreach ($name in @('gpu_1x_gh200', 'gpu_1x_gh200_alt')) {
        $types[$name] = [pscustomobject]@{
            instance_type = [pscustomobject]@{
                name = $name
                description = 'GH200'
                gpu_description = 'GH200'
                price_cents_per_hour = 200
                specs = [pscustomobject]@{ gpus = 1 }
            }
            regions_with_capacity_available = @()
        }
    }
    Assert-Throws { Find-Gh200InstanceType $types } 'Ambiguous discovery did not fail.'
}

Invoke-Test 'Availability transitions are edge-triggered' {
    Assert-Equal (Get-AvailabilityTransition $false $true) 'became_available' 'Missing up edge.'
    Assert-Equal (Get-AvailabilityTransition $true $false) 'became_unavailable' 'Missing down edge.'
    Assert-Equal (Get-AvailabilityTransition $true $true) 'none' 'Stable state should not notify.'
}

Invoke-Test 'Launch payload remains the exact fixed single-instance payload' {
    $fileSystem = [pscustomobject]@{ id = 'fs-east3'; name = 'brazil-rv-east3' }
    $payload = New-LaunchPayload -InstanceTypeName 'gpu_1x_gh200' -FileSystem $fileSystem
    $names = @($payload.Keys | Sort-Object)
    $expected = @(
        'file_system_mounts', 'hostname', 'instance_type_name',
        'name', 'region_name', 'ssh_key_names', 'tags'
    )
    Assert-Equal ($names -join ',') ($expected -join ',') 'Payload keys changed.'
    Assert-Equal $payload.region_name 'us-east-3' 'Region changed.'
    Assert-Equal $payload.name 'brazil-rv-gh200' 'Instance name changed.'
    Assert-Equal $payload.hostname 'brazil-rv-gh200' 'Hostname changed.'
    $tagMap = @{}
    foreach ($tag in $payload.tags) {
        $tagMap[$tag.key] = $tag.value
    }
    Assert-Equal $tagMap.Count 3 'Tag count changed.'
    Assert-Equal $tagMap.project 'brazil-rv' 'Project tag changed.'
    Assert-Equal $tagMap.purpose 'research-training' 'Purpose tag changed.'
    Assert-Equal $tagMap.'managed-by' 'gh200-watcher' 'Managed-by tag changed.'
    Assert-Equal $payload.ssh_key_names.Count 1 'SSH key count changed.'
    Assert-Equal $payload.ssh_key_names[0] 'brazil-rv' 'SSH key changed.'
    Assert-Equal $payload.file_system_mounts[0].file_system_id 'fs-east3' 'Filesystem ID changed.'
    Assert-Equal $payload.file_system_mounts[0].mount_point '/lambda/nfs/brazil-rv-east3' 'Mount changed.'
    foreach ($omitted in @(
        'quantity', 'image', 'user_data', 'file_system_names', 'firewall_rulesets'
    )) {
        Assert-True (-not $payload.Contains($omitted)) "Payload unexpectedly contains $omitted."
    }
}

Invoke-Test 'Lambda errors are classified without retrying fatal launch errors' {
    Assert-Equal (
        Classify-LambdaFailure 'instance-operations/launch/insufficient-capacity' 400 $true
    ) 'retry_capacity' 'Capacity should be retryable.'
    Assert-Equal (Classify-LambdaFailure 'global/quota-exceeded' 400 $true) 'fatal' 'Quota should be fatal.'
    Assert-Equal (Classify-LambdaFailure $null 0 $true) 'reconcile' 'Launch timeout should reconcile.'
    Assert-Equal (Classify-LambdaFailure $null 503 $false) 'transient' 'GET 503 should retry.'
}

Invoke-Test 'Central launch limiter enforces 12.5 seconds and five per minute' {
    Assert-Equal (Get-LaunchWaitSeconds @(0.0) 5.0) 7.5 'Minimum interval failed.'
    Assert-Equal (
        Get-LaunchWaitSeconds @(0.0, 12.5, 25.0, 37.5, 50.0) 50.0
    ) 12.5 'Rolling launch window failed.'
    Assert-True (-not (Test-LaunchAttemptAllowed @(0.0) 12.0)) 'Early launch was allowed.'
}

Invoke-Test 'Atomic state JSON leaves no temporary file' {
    $directory = Join-Path ([IO.Path]::GetTempPath()) ('gh200-state-' + [Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $path = Join-Path $directory 'state.json'
    try {
        Write-AtomicJson -Path $path -Value ([pscustomobject]@{ value = 7 })
        Assert-Equal ((Read-JsonFile $path).value) 7 'Atomic JSON was not readable.'
        Assert-Equal (@(Get-ChildItem -LiteralPath $directory -Filter '*.tmp').Count) 0 'Temporary state remained.'
    }
    finally {
        [IO.File]::Delete($path)
        [IO.Directory]::Delete($directory)
    }
}

Invoke-Test 'Matching instance discovery refuses duplicates' {
    $matches = Find-MatchingInstances @(
        (New-TestInstance -Id 'instance-1'),
        (New-TestInstance -Id 'instance-2')
    ) 'gpu_1x_gh200'
    Assert-Equal $matches.Count 2 'Duplicate fixture was not detected.'
}

Invoke-Test 'Structured logs redact credential-shaped values and registered secrets' {
    $script:SecretValues.Clear()
    Register-SecretValue 'super-secret-value'
    $entry = Format-LogEntry -Level INFO -Event test -Fields @{
        api_key = 'visible'
        note = 'prefix super-secret-value suffix'
    } -Timestamp '2026-07-23T00:00:00Z'
    Assert-True (-not $entry.Contains('visible')) 'API key leaked.'
    Assert-True (-not $entry.Contains('super-secret-value')) 'Registered secret leaked.'
    Assert-True $entry.Contains('[REDACTED]') 'Redaction marker missing.'
}

Invoke-Test 'Instance IDs are treated as opaque safe API identifiers' {
    $path = Get-BootstrapMarkerPath 'opaque.id_7-abc'
    Assert-True $path.EndsWith('bootstrap_gh200_opaque.id_7-abc_success.json') 'Opaque ID changed.'
}

Invoke-Test 'Mode validation keeps exactly Notify and Launch semantics' {
    Assert-Throws {
        Assert-OperationalParameters -SelectedMode Notify -BillingAcknowledged $true `
            -RunSelfTest $false -ForgetCredential $false
    } 'Notify accepted billing acknowledgement.'
    Assert-Throws {
        Assert-OperationalParameters -SelectedMode Launch -BillingAcknowledged $false `
            -RunSelfTest $false -ForgetCredential $false
    } 'Launch did not require billing acknowledgement.'
}

Invoke-Test 'SSH public key comments are ignored but key bodies are exact' {
    $identity = Assert-PublicKeyIdentityMatch `
        'ssh-ed25519 AAAAC3Nzacloud cloud-comment' `
        'ssh-ed25519 AAAAC3Nzacloud local-comment'
    Assert-Equal $identity 'ssh-ed25519 AAAAC3Nzacloud' 'Normalized key identity is wrong.'
    Assert-Throws {
        Assert-PublicKeyIdentityMatch `
            'ssh-ed25519 AAAAC3Nzacloud cloud-comment' `
            'ssh-ed25519 AAAAC3Nzalocal local-comment'
    } 'Mismatched SSH key bodies were accepted.'
}

Invoke-Test 'ssh-agent membership compares normalized identities ordinally' {
    Assert-True (
        Test-SshAgentContainsIdentity `
            "ssh-ed25519 AAAAone comment`nssh-ed25519 AAAAtwo other" `
            'ssh-ed25519 AAAAtwo'
    ) 'Exact agent identity was not found.'
    Assert-True (-not (
        Test-SshAgentContainsIdentity 'ssh-ed25519 AAAAtwo-extra comment' 'ssh-ed25519 AAAAtwo'
    )) 'Agent identity used a substring match.'
}

Invoke-Test 'ssh-agent exit two starts the Windows agent and retries once' {
    $state = [pscustomobject]@{
        Index = 0
        Starts = 0
        Queue = @(
            (New-ProcessResult 2),
            (New-ProcessResult 0)
        )
    }
    $listed = Get-SshAgentListing -SshAddPath 'ssh-add.exe' -ProcessInvoker ({
        param($Path)
        $result = $state.Queue[$state.Index]
        $state.Index++
        return $result
    }.GetNewClosure()) -AgentStarter ({
        $state.Starts++
    }.GetNewClosure())
    Assert-Equal $listed.ExitCode 0 'Recovered agent listing was not returned.'
    Assert-Equal $state.Index 2 'Agent listing was not retried exactly once.'
    Assert-Equal $state.Starts 1 'Windows agent was not started exactly once.'
}

Invoke-Test 'ssh-agent exit one means no identities and does not start the service' {
    $script:SshAgentStarts = 0
    $listed = Get-SshAgentListing -SshAddPath 'ssh-add.exe' -ProcessInvoker {
        param($Path)
        return New-ProcessResult 1
    } -AgentStarter {
        $script:SshAgentStarts++
    }
    Assert-Equal $listed.ExitCode 1 'Empty agent listing status changed.'
    Assert-Equal $script:SshAgentStarts 0 'Empty agent incorrectly restarted the service.'
}

Invoke-Test 'ssh-agent persistent exit two fails after one recovery attempt' {
    $script:SshAgentStarts = 0
    Assert-Throws {
        Get-SshAgentListing -SshAddPath 'ssh-add.exe' -ProcessInvoker {
            param($Path)
            return New-ProcessResult 2
        } -AgentStarter {
            $script:SshAgentStarts++
        }
    } 'Persistent unavailable agent was accepted.'
    Assert-Equal $script:SshAgentStarts 1 'Agent recovery was attempted more than once.'
}

Invoke-Test 'Frozen launch artifacts expose both hashes and detect tampering and CRLF' {
    $directory = Join-Path ([IO.Path]::GetTempPath()) ('gh200-artifacts-' + [Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $bundle = Join-Path $directory ('brazil-rv_{0}.bundle' -f ('a' * 40))
    $bootstrap = Join-Path $directory 'lambda-gh200-bootstrap.sh'
    $crlf = Join-Path $directory 'crlf.sh'
    try {
        [IO.File]::WriteAllBytes($bundle, [byte[]](1, 2, 3, 4))
        [IO.File]::WriteAllText($bootstrap, "#!/bin/bash`necho ok`n", (New-Object Text.UTF8Encoding($false)))
        [IO.File]::WriteAllText($crlf, "#!/bin/bash`r`n", (New-Object Text.UTF8Encoding($false)))
        $artifacts = New-FrozenLaunchArtifacts -GitSha ('a' * 40) `
            -BundlePath $bundle `
            -BundleSha256 (Get-FileHash $bundle -Algorithm SHA256).Hash.ToLowerInvariant() `
            -BootstrapPath $bootstrap `
            -BootstrapSha256 (Get-FileHash $bootstrap -Algorithm SHA256).Hash.ToLowerInvariant()
        Assert-Equal $artifacts.GitSha ('a' * 40) 'Frozen Git SHA missing.'
        Assert-Equal $artifacts.Count 5 'Artifact object fields changed.'
        Assert-True $artifacts.BundlePath.EndsWith(
            ('brazil-rv_{0}.bundle' -f ('a' * 40)),
            [StringComparison]::Ordinal
        ) 'Frozen bundle filename changed.'
        Assert-True (-not [string]::IsNullOrWhiteSpace($artifacts.BundleSha256)) 'Bundle hash missing.'
        Assert-True (-not [string]::IsNullOrWhiteSpace($artifacts.BootstrapSha256)) 'Bootstrap hash missing.'
        Assert-FrozenLaunchArtifacts $artifacts
        Assert-True (-not (Test-Utf8LfFile $crlf $null)) 'CRLF bootstrap was accepted.'
        Assert-Throws { $artifacts.GitSha = 'changed' } 'Artifact object was mutable.'
        [IO.File]::AppendAllText($bundle, 'tamper')
        Assert-Throws { Assert-FrozenLaunchArtifacts $artifacts } 'Tampered bundle was accepted.'
        [IO.File]::WriteAllBytes($bundle, [byte[]](1, 2, 3, 4))
        [IO.File]::AppendAllText($bootstrap, 'tamper')
        Assert-Throws { Assert-FrozenLaunchArtifacts $artifacts } 'Tampered bootstrap was accepted.'
    }
    finally {
        foreach ($path in @($bundle, $bootstrap, $crlf)) {
            [IO.File]::Delete($path)
        }
        [IO.Directory]::Delete($directory)
    }
}

Invoke-Test 'State uses launch identity fields and removes local_git_sha' {
    $state = New-WatcherState Launch
    Assert-True ($null -ne $state.PSObject.Properties['launch_git_sha']) 'launch_git_sha missing.'
    Assert-True ($null -ne $state.PSObject.Properties['launch_bundle_sha256']) 'bundle hash missing.'
    Assert-True ($null -ne $state.PSObject.Properties['launch_bootstrap_sha256']) 'bootstrap hash missing.'
    Assert-True ($null -ne $state.PSObject.Properties['sanity_report_path']) 'sanity path missing.'
    Assert-True ($null -eq $state.PSObject.Properties['local_git_sha']) 'Legacy local_git_sha remains.'
}

Invoke-Test 'Same-instance adoption preserves launch identity; different adoption clears it' {
    $script:PersistWatcherState = $false
    $script:State = New-WatcherState Launch
    $script:State.instance_id = 'instance-1'
    $script:State.launch_git_sha = 'a' * 40
    $script:State.launch_bundle_sha256 = 'b' * 64
    $script:State.launch_bootstrap_sha256 = 'c' * 64
    Set-AdoptedInstanceState (New-TestInstance -Id 'instance-1') 'gpu_1x_gh200'
    Assert-Equal $script:State.launch_git_sha ('a' * 40) 'Same adoption overwrote Git SHA.'
    Set-AdoptedInstanceState (New-TestInstance -Id 'instance-2') 'gpu_1x_gh200'
    Assert-Equal $script:State.launch_git_sha $null 'Different adoption retained Git SHA.'
    Assert-Equal $script:State.launch_bundle_sha256 $null 'Different adoption retained bundle hash.'
}

Invoke-Test 'Reset clears instance, launch identity, bootstrap, and sanity report path' {
    $script:State = New-WatcherState Launch
    foreach ($name in @(
        'instance_id', 'launch_git_sha', 'launch_bundle_sha256',
        'launch_bootstrap_sha256', 'bootstrap_status', 'sanity_report_path'
    )) {
        $script:State.$name = 'set'
    }
    Reset-ActiveState
    foreach ($name in @(
        'instance_id', 'launch_git_sha', 'launch_bundle_sha256',
        'launch_bootstrap_sha256', 'bootstrap_status', 'sanity_report_path'
    )) {
        Assert-Equal $script:State.$name $null "Reset retained $name."
    }
}

Invoke-Test 'Valid marker recovers artifact identity when local state lacks it' {
    $state = New-WatcherState Launch
    $identity = Assert-BootstrapMarker -Marker (New-TestMarker) `
        -SanityReport ([pscustomobject]@{ passed = $true }) `
        -InstanceId 'instance-1' -State $state
    Assert-Equal $identity.GitSha ('a' * 40) 'Marker Git SHA recovery failed.'
    Assert-Equal $identity.BundleSha256 ('b' * 64) 'Marker bundle hash recovery failed.'
    Assert-Equal $identity.BootstrapSha256 ('c' * 64) 'Marker bootstrap hash recovery failed.'
}

Invoke-Test 'Marker validation rejects wrong ID and wrong stored hashes' {
    $empty = New-WatcherState Launch
    Assert-Throws {
        Assert-BootstrapMarker -Marker (New-TestMarker -InstanceId 'wrong') `
            -SanityReport ([pscustomobject]@{ passed = $true }) `
            -InstanceId 'instance-1' -State $empty
    } 'Wrong marker instance ID was accepted.'
    $gitState = New-WatcherState Launch
    $gitState.launch_git_sha = 'd' * 40
    $gitState.launch_bundle_sha256 = 'b' * 64
    $gitState.launch_bootstrap_sha256 = 'c' * 64
    Assert-Throws {
        Assert-BootstrapMarker -Marker (New-TestMarker) `
            -SanityReport ([pscustomobject]@{ passed = $true }) `
            -InstanceId 'instance-1' -State $gitState
    } 'Wrong marker Git SHA was accepted.'
    $state = New-WatcherState Launch
    $state.launch_git_sha = 'a' * 40
    $state.launch_bundle_sha256 = 'd' * 64
    $state.launch_bootstrap_sha256 = 'c' * 64
    Assert-Throws {
        Assert-BootstrapMarker -Marker (New-TestMarker) `
            -SanityReport ([pscustomobject]@{ passed = $true }) `
            -InstanceId 'instance-1' -State $state
    } 'Wrong marker hash was accepted.'
}

Invoke-Test 'Marker validation rejects missing or failed sanity reports' {
    $state = New-WatcherState Launch
    Assert-Throws {
        Assert-BootstrapMarker -Marker (New-TestMarker) -SanityReport $null `
            -InstanceId 'instance-1' -State $state
    } 'Missing sanity report was accepted.'
    Assert-Throws {
        Assert-BootstrapMarker -Marker (New-TestMarker) `
            -SanityReport ([pscustomobject]@{ passed = $false }) `
            -InstanceId 'instance-1' -State $state
    } 'Failed sanity report was accepted.'
}

Invoke-Test 'Recovery requires a marker or original frozen artifacts regardless of local success state' {
    $state = New-WatcherState Launch
    $state.bootstrap_status = 'succeeded'
    Assert-Equal (Get-BootstrapRecoveryAction $null $null) 'refuse' 'Local state authorized recovery.'
    Assert-Equal (
        Get-BootstrapRecoveryAction ([pscustomobject]@{ Marker = 'valid' }) $null
    ) 'marker' 'Marker did not take precedence.'
    Assert-Equal (
        Get-BootstrapRecoveryAction $null ([pscustomobject]@{ GitSha = 'original' })
    ) 'bootstrap' 'Original artifacts did not authorize idempotent bootstrap.'
}

Invoke-Test 'State reconstructs original artifact paths without consulting current HEAD' {
    $oldDirectory = $script:ArtifactsDirectory
    $script:ArtifactsDirectory = 'C:\frozen-artifacts'
    try {
        $script:State = New-WatcherState Launch
        $script:State.launch_git_sha = 'a' * 40
        $script:State.launch_bundle_sha256 = 'b' * 64
        $script:State.launch_bootstrap_sha256 = 'c' * 64
        $artifacts = Get-StateLaunchArtifacts
        Assert-True $artifacts.BundlePath.Contains(('a' * 40)) 'Original SHA not used in bundle path.'
        Assert-Equal $artifacts.GitSha ('a' * 40) 'Stored SHA was replaced.'
    }
    finally {
        $script:ArtifactsDirectory = $oldDirectory
    }
}

Invoke-Test 'Lambda request parameters always include a 30-second timeout' {
    $script:ApiHeaders = @{ Authorization = 'Bearer redacted' }
    $get = New-LambdaRequestParameters -Method GET -Path '/instances' -Body $null
    $post = New-LambdaRequestParameters -Method POST -Path '/instance-operations/launch' `
        -Body ([ordered]@{ region_name = 'us-east-3' })
    Assert-Equal $get.TimeoutSec 30 'GET timeout changed.'
    Assert-Equal $post.TimeoutSec 30 'POST timeout changed.'
    Assert-True $post.ContainsKey('Body') 'POST body missing.'
    $script:ApiHeaders = $null
}

Invoke-Test 'Ambiguous reconciliation repeats empty reads then adopts exactly one instance' {
    $script:PersistWatcherState = $false
    $script:State = New-WatcherState Launch
    $script:TestNow = 0.0
    $holder = [pscustomobject]@{
        Calls = 0
        Instance = New-TestInstance -Id 'reconciled-1'
    }
    $api = {
        $holder.Calls++
        if ($holder.Calls -lt 3) { return @() }
        return @($holder.Instance)
    }.GetNewClosure()
    $now = { $script:TestNow }
    $sleep = { param([double]$Seconds) $script:TestNow += $Seconds }
    $resolved = Resolve-AmbiguousLaunch 'gpu_1x_gh200' $api $now $sleep
    Assert-Equal $resolved.id 'reconciled-1' 'Reconciliation did not adopt one instance.'
    Assert-Equal $holder.Calls 3 'Reconciliation did not repeat GETs.'
}

Invoke-Test 'Ambiguous reconciliation fails on multiple matching instances' {
    $script:State = New-WatcherState Launch
    $api = {
        @(
            (New-TestInstance -Id 'duplicate-1'),
            (New-TestInstance -Id 'duplicate-2')
        )
    }
    Assert-Throws {
        Resolve-AmbiguousLaunch 'gpu_1x_gh200' $api { 0.0 } { param($Seconds) }
    } 'Multiple ambiguous matches were accepted.'
}

Invoke-Test 'Ambiguous reconciliation returns null only after the full 60-second window' {
    $script:TestNow = 0.0
    $script:TestCalls = 0
    $api = { $script:TestCalls++; @() }
    $now = { $script:TestNow }
    $sleep = { param([double]$Seconds) $script:TestNow += $Seconds }
    $resolved = Resolve-AmbiguousLaunch 'gpu_1x_gh200' $api $now $sleep
    Assert-Equal $resolved $null 'Empty reconciliation did not return null.'
    Assert-Equal $script:TestNow 60.0 'Reconciliation did not cover 60 seconds.'
    Assert-True ($script:TestCalls -gt 1) 'Reconciliation made only one GET.'
}

Invoke-Test 'Firewall accepts exact/range/all SSH rules and rejects unrelated or absent rules' {
    Assert-True (Test-FirewallRuleAllowsSsh ([pscustomobject]@{
        protocol = 'tcp'; port_range = @(22, 22)
    })) 'Exact port 22 was rejected.'
    Assert-True (Test-FirewallRuleAllowsSsh ([pscustomobject]@{
        protocol = 'tcp'; port_range = @(20, 30)
    })) 'Range containing 22 was rejected.'
    Assert-True (Test-FirewallRuleAllowsSsh ([pscustomobject]@{
        protocol = 'all'
    })) 'all protocol was rejected.'
    Assert-True (-not (Test-FirewallRuleAllowsSsh ([pscustomobject]@{
        protocol = 'udp'; port_range = @(22, 22)
    }))) 'UDP port 22 was accepted.'
    Assert-True (-not (Test-FirewallRuleAllowsSsh ([pscustomobject]@{
        protocol = 'tcp'; port_range = @(80, 443)
    }))) 'Unrelated TCP ports were accepted.'
    Assert-Throws {
        Assert-GlobalFirewallAllowsSsh ([pscustomobject]@{
            rules = @([pscustomobject]@{ protocol = 'tcp'; port_range = @(80, 443) })
        })
    } 'Unrelated firewall rules were accepted.'
    Assert-Throws {
        Assert-GlobalFirewallAllowsSsh ([pscustomobject]@{ rules = @() })
    } 'Absent firewall rules were accepted.'
}

Invoke-Test 'Authenticated SSH ignores TCP-only and failed SSH before a later zero exit' {
    $script:TestNow = 0.0
    $script:SshAttempts = 0
    $preflight = [pscustomobject]@{
        PrivateKeyPath = 'key'
        SshPath = 'ssh'
    }
    $tcp = { param($HostName) $true }
    $attempt = {
        param($Arguments)
        $script:SshAttempts++
        if ($script:SshAttempts -eq 1) { return New-ProcessResult 255 }
        return New-ProcessResult 0
    }
    Wait-ForAuthenticatedSsh '198.51.100.2' 'instance-1' $preflight 'known-hosts' `
        $tcp $attempt { $script:TestNow } `
        { param([double]$Seconds) $script:TestNow += $Seconds }
    Assert-Equal $script:SshAttempts 2 'Failed authentication was treated as ready.'
}

Invoke-Test 'Authenticated SSH times out when every attempt times out' {
    $oldTimeout = $script:SshTimeoutSeconds
    $script:SshTimeoutSeconds = 4
    try {
        $script:TestNow = 0.0
        $preflight = [pscustomobject]@{
            PrivateKeyPath = 'key'
            SshPath = 'ssh'
        }
        Assert-Throws {
            Wait-ForAuthenticatedSsh '198.51.100.2' 'instance-1' $preflight 'known-hosts' `
                { param($HostName) $true } `
                { param($Arguments) New-ProcessResult -1 $true } `
                { $script:TestNow } `
                { param([double]$Seconds) $script:TestNow += $Seconds }
        } 'Authenticated SSH timeout was accepted.'
    }
    finally {
        $script:SshTimeoutSeconds = $oldTimeout
    }
}

Invoke-Test 'SCP retries failures and succeeds on the third attempt' {
    $script:ScpIndex = 0
    $script:ScpQueue = @(
        (New-ProcessResult 1),
        (New-ProcessResult 2),
        (New-ProcessResult 0)
    )
    $result = Invoke-ScpWithRetry 'scp' @('source', 'destination') `
        { param($File, $Arguments) $item = $script:ScpQueue[$script:ScpIndex]; $script:ScpIndex++; $item } `
        { param($Seconds) }
    Assert-True $result.Succeeded 'Third SCP success was not accepted.'
    Assert-Equal $result.Attempts 3 'SCP attempt count is wrong.'
}

Invoke-Test 'SCP reports final actual exit and maps only a final timeout to minus one' {
    $script:ScpIndex = 0
    $script:ScpQueue = @(
        (New-ProcessResult 3),
        (New-ProcessResult 4),
        (New-ProcessResult 9)
    )
    $failed = Invoke-ScpWithRetry 'scp' @('argument') `
        { param($File, $Arguments) $item = $script:ScpQueue[$script:ScpIndex]; $script:ScpIndex++; $item } `
        { param($Seconds) }
    Assert-Equal $failed.ExitCode 9 'SCP did not preserve the final exit code.'
    $timed = Invoke-ScpWithRetry 'scp' @('argument') `
        { param($File, $Arguments) New-ProcessResult -1 $true } `
        { param($Seconds) }
    Assert-Equal $timed.ExitCode -1 'Timed-out SCP did not report minus one.'
}

Invoke-Test 'All-failed SCP path records failed bootstrap state' {
    $script:PersistWatcherState = $false
    $script:State = New-WatcherState Launch
    Assert-Throws {
        Stop-RemoteBootstrapFailure 'instance-1' '198.51.100.2' 9 `
            '/persistent/log' 'ssh manual' 'SCP failed'
    } 'Failure handler did not throw.'
    Assert-Equal $script:State.bootstrap_status 'failed' 'Failure state was not recorded.'
}

Invoke-Test 'Bootstrap SSH invocation passes four identity values in order' {
    $arguments = New-BootstrapSshArguments -PrivateKeyPath 'C:\key path\id' `
        -KnownHostsPath 'C:\known hosts\file' `
        -IpAddress '198.51.100.2' `
        -GitSha ('a' * 40) `
        -BundleSha256 ('b' * 64) `
        -BootstrapSha256 ('c' * 64) `
        -InstanceId 'instance-1'
    $bashIndex = [Array]::IndexOf($arguments, 'bash')
    Assert-True ($bashIndex -ge 0) 'Remote bash command missing.'
    Assert-True ($arguments -ccontains 'C:\key path\id') 'Private-key path with spaces changed.'
    Assert-True ($arguments -ccontains 'UserKnownHostsFile=C:\known hosts\file') 'Known-host path with spaces changed.'
    Assert-Equal $arguments[$bashIndex + 1] '/home/ubuntu/lambda-gh200-bootstrap.sh' 'Bootstrap path changed.'
    Assert-Equal $arguments[$bashIndex + 2] ('a' * 40) 'Git SHA argument changed.'
    Assert-Equal $arguments[$bashIndex + 3] ('b' * 64) 'Bundle hash argument changed.'
    Assert-Equal $arguments[$bashIndex + 4] ('c' * 64) 'Bootstrap hash argument changed.'
    Assert-Equal $arguments[$bashIndex + 5] 'instance-1' 'Instance ID argument changed.'
}

Invoke-Test 'Bootstrap source enforces four args, idempotent marker, and no training' {
    $bootstrap = [IO.File]::ReadAllText(
        (Join-Path $PSScriptRoot 'lambda-gh200-bootstrap.sh'),
        [Text.Encoding]::UTF8
    )
    Assert-True $bootstrap.Contains('[[ $# -ne 4 ]]') 'Bootstrap does not require four args.'
    Assert-True $bootstrap.Contains('EXPECTED_BOOTSTRAP_SHA256="$3"') 'Self hash arg missing.'
    Assert-True $bootstrap.Contains('BRAZIL_RV_BOOTSTRAP_ALREADY_COMPLETE=') 'Idempotent exit missing.'
    Assert-True $bootstrap.Contains('"passed": True') 'Expanded marker passed field missing.'
    Assert-True $bootstrap.Contains('validate_repository') 'Existing repository validation missing.'
    Assert-True (-not $bootstrap.Contains('brazil_rv.modeling.train')) 'Training command was added.'
    Assert-True (-not $bootstrap.Contains('torchrun')) 'Distributed training was added.'
}

Invoke-Test 'Remote marker probe emits exact shell-safe file test command' {
    $marker = '/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/_ops/marker.json'
    Assert-Equal (New-RemoteFileExistsCommand $marker) `
        "[ -f '$marker' ]" 'Marker probe command changed.'
    $metacharacterPath = '/tmp/marker path;$HOME;*?[x]'
    Assert-Equal (New-RemoteFileExistsCommand $metacharacterPath) `
        '[ -f ''/tmp/marker path;$HOME;*?[x]'' ]' 'Metacharacter path was not quoted.'
    $quotedPath = "/tmp/it's here"
    $singleQuote = [string][char]39
    $doubleQuote = [string][char]34
    $escaped = $singleQuote + $doubleQuote + $singleQuote + $doubleQuote + $singleQuote
    $expected = '[ -f ' + $singleQuote + '/tmp/it' + $escaped + 's here' + $singleQuote + ' ]'
    Assert-Equal (New-RemoteFileExistsCommand $quotedPath) $expected 'Single quote was not escaped.'
    Assert-True (-not (New-RemoteFileExistsCommand $marker).Contains('test -f --')) `
        'Invalid test syntax remains in the generated command.'
}

Invoke-Test 'Remote marker probe classifies present absent and failures' {
    Assert-True (Resolve-RemoteFileExistsProbe (New-ProcessResult 0)) 'Exit 0 was not present.'
    Assert-True (-not (Resolve-RemoteFileExistsProbe (New-ProcessResult 1))) 'Exit 1 was not absent.'
    Assert-Throws {
        Resolve-RemoteFileExistsProbe (New-ProcessResult 2)
    } 'Exit 2 was not a failure.'
    Assert-Throws {
        Resolve-RemoteFileExistsProbe (New-ProcessResult -1 $true)
    } 'Timeout was not a failure.'
}

Invoke-Test 'API request order records the post-preflight actual start' {
    $script:LastApiRequestStart = 0.0
    $script:LaunchAttemptStarts = @()
    $clock = [pscustomobject]@{ Now = 0.0 }
    $events = New-Object 'System.Collections.Generic.List[string]'
    $now = { $clock.Now }.GetNewClosure()
    $waitUntil = {
        param([double]$EligibleAt)
        if ($clock.Now -lt $EligibleAt) { $clock.Now = $EligibleAt }
    }.GetNewClosure()
    $wait = {
        param([bool]$IsLaunch)
        $events.Add('wait')
        Wait-ApiRequestSlot -Launch:$IsLaunch -Now $now -WaitUntil $waitUntil
    }.GetNewClosure()
    $before = {
        $events.Add('before')
        $clock.Now += 0.90
    }.GetNewClosure()
    $register = {
        param([bool]$IsLaunch)
        $events.Add('register')
        Register-ApiRequestStart -Launch:$IsLaunch -Now $now
    }.GetNewClosure()
    $http = {
        param([hashtable]$Parameters)
        $events.Add('http')
        [pscustomobject]@{ Content = '{"data":{"ok":true}}'; StatusCode = 200 }
    }.GetNewClosure()
    $result = Invoke-LambdaApi -Method GET -Path '/instances' -BeforeRequest $before `
        -WaitForSlot $wait -RequestStartRegistrar $register -WebRequestInvoker $http
    Assert-Equal $result.RequestStartedSeconds 2.0 'Preflight time was not excluded from gating.'
    Assert-Equal ($events -join ',') 'wait,before,register,http' 'Request event order changed.'
    Wait-ApiRequestSlot -Now $now -WaitUntil $waitUntil
    Assert-Equal $clock.Now 3.1 'Next request was not gated from the actual start.'
}

Invoke-Test 'Variable preflight durations preserve the 1.10-second actual-start interval' {
    $script:LastApiRequestStart = -1.0
    $script:LaunchAttemptStarts = @()
    $clock = [pscustomobject]@{ Now = 0.0 }
    $now = { $clock.Now }.GetNewClosure()
    $waitUntil = {
        param([double]$EligibleAt)
        if ($clock.Now -lt $EligibleAt) { $clock.Now = $EligibleAt }
    }.GetNewClosure()
    $starts = @()
    foreach ($duration in @(0.20, 2.00, 0.05)) {
        Wait-ApiRequestSlot -Now $now -WaitUntil $waitUntil
        $clock.Now += $duration
        $starts += Register-ApiRequestStart -Now $now
    }
    for ($index = 1; $index -lt $starts.Count; $index++) {
        Assert-True (($starts[$index] - $starts[$index - 1]) -ge 1.10) `
            'Actual API starts were less than 1.10 seconds apart.'
    }
}

Invoke-Test 'Launch spacing is measured from actual starts after slow and fast preflight' {
    $script:PersistWatcherState = $false
    $script:State = $null
    $script:LastApiRequestStart = -1.0
    $script:LaunchAttemptStarts = @()
    $clock = [pscustomobject]@{ Now = 0.0 }
    $now = { $clock.Now }.GetNewClosure()
    $waitUntil = {
        param([double]$EligibleAt)
        if ($clock.Now -lt $EligibleAt) { $clock.Now = $EligibleAt }
    }.GetNewClosure()
    Wait-ApiRequestSlot -Launch -Now $now -WaitUntil $waitUntil
    $clock.Now += 5.0
    $first = Register-ApiRequestStart -Launch -Now $now
    Wait-ApiRequestSlot -Launch -Now $now -WaitUntil $waitUntil
    $clock.Now += 0.1
    $second = Register-ApiRequestStart -Launch -Now $now
    Assert-True (($second - $first) -ge 12.50) 'Launch actual starts were too close.'
}

Invoke-Test 'Launch rolling window stores at most five actual starts' {
    $script:State = $null
    $script:LastApiRequestStart = -1.0
    $script:LaunchAttemptStarts = @()
    $clock = [pscustomobject]@{ Now = 0.0 }
    $now = { $clock.Now }.GetNewClosure()
    $waitUntil = {
        param([double]$EligibleAt)
        if ($clock.Now -lt $EligibleAt) { $clock.Now = $EligibleAt }
    }.GetNewClosure()
    $actual = @()
    $maximumStored = 0
    foreach ($attempt in 1..6) {
        Wait-ApiRequestSlot -Launch -Now $now -WaitUntil $waitUntil
        $actual += Register-ApiRequestStart -Launch -Now $now
        $maximumStored = [Math]::Max($maximumStored, $script:LaunchAttemptStarts.Count)
    }
    Assert-Equal $maximumStored 5 'More than five launch starts were retained.'
    Assert-Equal $actual[5] 62.5 'Sixth actual start ignored the rolling window.'
    Assert-Equal ($script:LaunchAttemptStarts -join ',') `
        ($actual[1..5] -join ',') 'Limiter stored reservation rather than actual starts.'
}

Invoke-Test 'BeforeRequest failure records no request or launch attempt' {
    $script:PersistWatcherState = $false
    $script:State = New-WatcherState Launch
    $script:State.launch_attempt_count = 7
    $script:State.last_launch_attempt_utc = '2026-07-23T00:00:00Z'
    $script:LastApiRequestStart = 100.0
    $script:LaunchAttemptStarts = @(87.5, 100.0)
    $clock = [pscustomobject]@{ Now = 112.5 }
    $now = { $clock.Now }.GetNewClosure()
    $waitUntil = {
        param([double]$EligibleAt)
        if ($clock.Now -lt $EligibleAt) { $clock.Now = $EligibleAt }
    }.GetNewClosure()
    $httpCalls = 0
    $script:TestSaveCalls = 0
    $wait = {
        param([bool]$IsLaunch)
        Wait-ApiRequestSlot -Launch:$IsLaunch -Now $now -WaitUntil $waitUntil
    }.GetNewClosure()
    $register = {
        param([bool]$IsLaunch)
        Register-ApiRequestStart -Launch:$IsLaunch -Now $now `
            -UtcNow { 'changed' } -StateSaver { $script:TestSaveCalls++ }
    }.GetNewClosure()
    $http = {
        param([hashtable]$Parameters)
        $httpCalls++
        [pscustomobject]@{ Content = '{"data":{}}'; StatusCode = 200 }
    }.GetNewClosure()
    Assert-Throws {
        Invoke-LambdaApi -Method POST -Path '/instance-operations/launch' -Body @{} -Launch `
            -BeforeRequest { throw 'preflight failed' } -WaitForSlot $wait `
            -RequestStartRegistrar $register -WebRequestInvoker $http
    } 'BeforeRequest failure did not propagate.'
    Assert-Equal $httpCalls 0 'HTTP ran after BeforeRequest failure.'
    Assert-Equal $script:LastApiRequestStart 100.0 'Last API start changed.'
    Assert-Equal ($script:LaunchAttemptStarts -join ',') '87.5,100' 'Launch starts changed.'
    Assert-Equal $script:State.launch_attempt_count 7 'Launch count changed.'
    Assert-Equal $script:State.last_launch_attempt_utc '2026-07-23T00:00:00Z' 'Launch UTC changed.'
    Assert-Equal $script:TestSaveCalls 0 'State was persisted after BeforeRequest failure.'
}

Invoke-Test 'Successful launch registration persists one actual start' {
    $script:PersistWatcherState = $false
    $script:State = New-WatcherState Launch
    $script:LastApiRequestStart = 0.0
    $script:LaunchAttemptStarts = @()
    $clock = [pscustomobject]@{ Now = 0.0 }
    $now = { $clock.Now }.GetNewClosure()
    $waitUntil = {
        param([double]$EligibleAt)
        if ($clock.Now -lt $EligibleAt) { $clock.Now = $EligibleAt }
    }.GetNewClosure()
    Wait-ApiRequestSlot -Launch -Now $now -WaitUntil $waitUntil
    $clock.Now += 0.90
    $script:TestSaveCalls = 0
    $started = Register-ApiRequestStart -Launch -Now $now `
        -UtcNow { '2026-07-23T12:34:56Z' } -StateSaver { $script:TestSaveCalls++ }
    Assert-Equal $started 2.0 'Launch registration used reservation time.'
    Assert-Equal ($script:LaunchAttemptStarts -join ',') '2' 'Actual launch start was not stored.'
    Assert-Equal $script:State.launch_attempt_count 1 'Launch count did not increment exactly once.'
    Assert-Equal $script:State.last_launch_attempt_utc '2026-07-23T12:34:56Z' 'Launch UTC was not stored.'
    Assert-Equal $script:TestSaveCalls 1 'Launch state was not persisted exactly once.'
}
$total = $script:Passed + $script:Failed
Write-Host ""
Write-Host "Lambda GH200 watcher tests: $($script:Passed)/$total passed."
if ($script:Failed -ne 0) {
    exit 1
}





