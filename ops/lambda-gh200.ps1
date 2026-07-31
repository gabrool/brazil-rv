<#
.SYNOPSIS
Watches Lambda Cloud for a single-GPU GH200 in us-east-3.

.DESCRIPTION
This Windows watcher has exactly two operational modes. Notify polls the official
Lambda Cloud API and alerts when capacity changes. Launch performs the same polling,
launches one matching instance when capacity appears, mounts brazil-rv-east3, uses
the brazil-rv SSH key, waits for SSH, uploads a verified Git bundle, bootstraps the
workspace, runs the repository checks, and runs the real gh200 model sanity check.

The API key is never accepted on the command line or through the clipboard. On first
use it is read as a SecureString and may be stored for the current Windows user with
DPAPI at:
  %LOCALAPPDATA%\BrazilRV\lambda-gh200-watcher\credential\lambda-api-key.dpapi

State and logs are written under:
  %LOCALAPPDATA%\BrazilRV\lambda-gh200-watcher\state\launch-state.json
  %LOCALAPPDATA%\BrazilRV\lambda-gh200-watcher\logs\

Availability polls start no more often than every 1.25 seconds. One central limiter
keeps all API requests at least 1.10 seconds apart. Launch attempts are also kept at
least 12.50 seconds apart and to no more than five attempts in any rolling minute.
HTTP 429 Retry-After values, bounded local rate-limit backoff, and bounded GET
backoff are respected.

Launch mode may create a billable instance. Launch mode does not automatically
terminate it. Launch mode does not automatically start production training. It runs
only repository validation and the contracted GH200 sanity check. The PC must remain
powered on, awake, and online while waiting. Stop either watcher with Ctrl+C.

Use -ForgetStoredApiKey to remove only the DPAPI credential. Inspect the paths above
for logs and state. After launch, the script prints a complete manual SSH command.

.PARAMETER Mode
The operational mode: Notify or Launch.

.PARAMETER IUnderstandBilling
Required for Launch and rejected for Notify.

.PARAMETER SelfTest
Runs the local no-network watcher tests.

.PARAMETER ForgetStoredApiKey
Deletes the current user's DPAPI-encrypted Lambda API key and exits.

.EXAMPLE
.\ops\lambda-gh200.ps1 -Mode Notify

.EXAMPLE
.\ops\lambda-gh200.ps1 -Mode Launch -IUnderstandBilling

.EXAMPLE
.\ops\lambda-gh200.ps1 -ForgetStoredApiKey
#>
[CmdletBinding()]
param(
    [ValidateSet('Notify', 'Launch')]
    [string]$Mode,

    [switch]$IUnderstandBilling,
    [switch]$SelfTest,
    [switch]$ForgetStoredApiKey
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:ScriptVersion = '1.1.2'
$script:ApiBaseUri = 'https://cloud.lambda.ai/api/v1'
$script:TargetRegion = 'us-east-3'
$script:FileSystemName = 'brazil-rv-east3'
$script:FileSystemMount = '/lambda/nfs/brazil-rv-east3'
$script:SshKeyName = 'brazil-rv'
$script:InstanceName = 'brazil-rv-gh200'
$script:InstanceHostname = 'brazil-rv-gh200'
$script:RepositoryRoot = 'C:\Brazil-RV\quant\b3-quant'
$script:RemoteWorkspace = '/home/ubuntu/Brazil-RV'
$script:RemoteRepository = '/home/ubuntu/Brazil-RV/quant/b3-quant'
$script:AvailabilityPollSeconds = 1.25
$script:ApiMinimumSeconds = 1.10
$script:LaunchMinimumSeconds = 12.50
$script:LaunchWindowSeconds = 60.0
$script:LaunchWindowMaximum = 5
$script:InstancePollSeconds = 1.25
$script:SshPollSeconds = 5.0
$script:ActiveTimeoutSeconds = 20 * 60
$script:SshTimeoutSeconds = 45 * 60
$script:SshAttemptTimeoutSeconds = 15
$script:SshProgressSeconds = 5 * 60
$script:ScpTimeoutSeconds = 5 * 60
$script:ScpAttempts = 3
$script:ScpRetrySeconds = 5
$script:AmbiguousLaunchSeconds = 60.0
$script:AmbiguousLaunchPollSeconds = 1.25
$script:HttpTimeoutSeconds = 30
$script:BootstrapTimeoutSeconds = 90 * 60
$script:TransientGetBackoffSeconds = @(2, 4, 8, 16, 32, 60)
$script:RateLimitBackoffSeconds = @(2, 4, 8, 16, 32, 60)
$script:RateLimitRecoverySuccesses = 3
$script:StateFieldNames = @(
    'script_version',
    'mode',
    'watcher_started_at_utc',
    'last_availability',
    'last_availability_change_utc',
    'last_successful_poll_utc',
    'launch_attempt_count',
    'last_launch_attempt_utc',
    'instance_id',
    'instance_name',
    'instance_type_name',
    'region',
    'status',
    'ip',
    'filesystem_id',
    'launch_git_sha',
    'launch_bundle_sha256',
    'launch_bootstrap_sha256',
    'bootstrap_status',
    'bootstrap_started_at_utc',
    'bootstrap_completed_at_utc',
    'sanity_report_path'
)

$script:WatcherScriptPath = $PSCommandPath
$script:OpsDirectory = Split-Path -Parent $PSCommandPath
$script:RuntimeRoot = $null
$script:CredentialPath = $null
$script:StatePath = $null
$script:LockPath = $null
$script:LogPath = $null
$script:KnownHostsDirectory = $null
$script:ArtifactsDirectory = $null
$script:ApiHeaders = $null
$script:ApiClock = $null
$script:LastApiRequestStart = -1.0
$script:LaunchAttemptStarts = @()
$script:TransientGetBackoffIndex = 0
$script:RateLimitBackoffIndex = 0
$script:RateLimitHealthySuccessCount = 0
$script:RateLimitLastLoggedDelay = $null
$script:LockStream = $null
$script:NotifyIcon = $null
$script:State = $null
$script:PersistWatcherState = $false
$script:SecretValues = New-Object 'System.Collections.Generic.List[string]'

function Get-PropertyValue {
    param(
        [AllowNull()]
        [object]$InputObject,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if ($null -eq $InputObject) {
        return $null
    }
    if ($InputObject -is [System.Collections.IDictionary]) {
        if ($InputObject.Contains($Name)) {
            return $InputObject[$Name]
        }
        return $null
    }
    $property = $InputObject.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Get-UtcTimestamp {
    return [DateTime]::UtcNow.ToString('o')
}

function Register-SecretValue {
    param([AllowEmptyString()][string]$Value)

    if (-not [string]::IsNullOrEmpty($Value) -and -not $script:SecretValues.Contains($Value)) {
        $script:SecretValues.Add($Value)
    }
}

function ConvertTo-SafeLogValue {
    param(
        [string]$Name,
        [AllowNull()]
        [object]$Value
    )

    if ($Name -match '(?i)(api.?key|authorization|jupyter|token|private.?key|secret)') {
        return '[REDACTED]'
    }
    if ($null -eq $Value) {
        return 'null'
    }
    if ($Value -is [bool]) {
        return $Value.ToString().ToLowerInvariant()
    }
    $text = [string]$Value
    foreach ($secret in $script:SecretValues) {
        if (-not [string]::IsNullOrEmpty($secret)) {
            $text = $text.Replace($secret, '[REDACTED]')
        }
    }
    $text = $text.Replace("`r", ' ').Replace("`n", ' ').Replace("`t", ' ')
    $text = $text.Replace('\', '\\').Replace('"', '\"')
    return '"{0}"' -f $text
}

function Format-LogEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Level,
        [Parameter(Mandatory = $true)]
        [string]$Event,
        [hashtable]$Fields = @{},
        [string]$Timestamp = (Get-UtcTimestamp)
    )

    $parts = @()
    $parts += ('timestamp={0}' -f (ConvertTo-SafeLogValue -Name 'timestamp' -Value $Timestamp))
    $parts += ('level={0}' -f (ConvertTo-SafeLogValue -Name 'level' -Value $Level.ToUpperInvariant()))
    $parts += ('event={0}' -f (ConvertTo-SafeLogValue -Name 'event' -Value $Event))
    foreach ($key in @($Fields.Keys | Sort-Object)) {
        $parts += ('{0}={1}' -f $key, (ConvertTo-SafeLogValue -Name $key -Value $Fields[$key]))
    }
    return $parts -join ' '
}

function Write-Log {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('DEBUG', 'INFO', 'WARN', 'ERROR')]
        [string]$Level,
        [Parameter(Mandatory = $true)]
        [string]$Event,
        [hashtable]$Fields = @{}
    )

    $entry = Format-LogEntry -Level $Level -Event $Event -Fields $Fields
    if (-not [string]::IsNullOrEmpty($script:LogPath)) {
        [IO.File]::AppendAllText(
            $script:LogPath,
            $entry + [Environment]::NewLine,
            (New-Object Text.UTF8Encoding($false))
        )
    }
}

function New-WatcherState {
    param([Parameter(Mandatory = $true)][string]$WatcherMode)

    return [pscustomobject][ordered]@{
        script_version = $script:ScriptVersion
        mode = $WatcherMode
        watcher_started_at_utc = Get-UtcTimestamp
        last_availability = $null
        last_availability_change_utc = $null
        last_successful_poll_utc = $null
        launch_attempt_count = 0
        last_launch_attempt_utc = $null
        instance_id = $null
        instance_name = $null
        instance_type_name = $null
        region = $script:TargetRegion
        status = $null
        ip = $null
        filesystem_id = $null
        launch_git_sha = $null
        launch_bundle_sha256 = $null
        launch_bootstrap_sha256 = $null
        bootstrap_status = $null
        bootstrap_started_at_utc = $null
        bootstrap_completed_at_utc = $null
        sanity_report_path = $null
    }
}

function Copy-SafeState {
    param([Parameter(Mandatory = $true)][object]$InputState)

    $safe = [ordered]@{}
    foreach ($name in $script:StateFieldNames) {
        $safe[$name] = Get-PropertyValue -InputObject $InputState -Name $name
    }
    return [pscustomobject]$safe
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Value
    )

    $directory = Split-Path -Parent $Path
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $temporary = Join-Path $directory ('.{0}.{1}.{2}.tmp' -f (
        [IO.Path]::GetFileName($Path), $PID, [Guid]::NewGuid().ToString('N')
    ))
    try {
        $json = $Value | ConvertTo-Json -Depth 20
        [IO.File]::WriteAllText($temporary, $json, (New-Object Text.UTF8Encoding($false)))
        if ([IO.File]::Exists($Path)) {
            $backup = $temporary + '.bak'
            [IO.File]::Replace($temporary, $Path, $backup)
            [IO.File]::Delete($backup)
        }
        else {
            [IO.File]::Move($temporary, $Path)
        }
    }
    finally {
        if ([IO.File]::Exists($temporary)) {
            [IO.File]::Delete($temporary)
        }
        if ($null -ne (Get-Variable -Name backup -ErrorAction SilentlyContinue) -and
            [IO.File]::Exists($backup)) {
            [IO.File]::Delete($backup)
        }
    }
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not [IO.File]::Exists($Path)) {
        return $null
    }
    return [IO.File]::ReadAllText($Path, [Text.Encoding]::UTF8) | ConvertFrom-Json
}

function Save-WatcherState {
    if ($script:PersistWatcherState -and $null -ne $script:State -and
        -not [string]::IsNullOrEmpty($script:StatePath)) {
        Write-AtomicJson -Path $script:StatePath -Value (Copy-SafeState $script:State)
    }
}

function Set-StateValue {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [AllowNull()][object]$Value
    )

    if ($script:StateFieldNames -notcontains $Name) {
        throw "State field is not permitted: $Name"
    }
    $script:State.$Name = $Value
}

function Reset-ActiveState {
    foreach ($name in @(
        'instance_id', 'instance_name', 'instance_type_name', 'status', 'ip',
        'launch_git_sha', 'launch_bundle_sha256', 'launch_bootstrap_sha256',
        'bootstrap_status', 'bootstrap_started_at_utc', 'bootstrap_completed_at_utc',
        'sanity_report_path'
    )) {
        Set-StateValue -Name $name -Value $null
    }
}

function Set-AdoptedInstanceState {
    param(
        [Parameter(Mandatory = $true)][object]$Instance,
        [Parameter(Mandatory = $true)][string]$InstanceTypeName
    )

    $instanceId = [string](Get-PropertyValue $Instance 'id')
    if ([string]::IsNullOrWhiteSpace($instanceId)) {
        throw 'A matching instance has no ID.'
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$script:State.instance_id) -and
        -not [string]::Equals(
            [string]$script:State.instance_id,
            $instanceId,
            [StringComparison]::Ordinal
        )) {
        Reset-ActiveState
    }
    Set-StateValue 'instance_id' $instanceId
    Set-StateValue 'instance_name' (Get-PropertyValue $Instance 'name')
    Set-StateValue 'instance_type_name' $InstanceTypeName
    Set-StateValue 'region' $script:TargetRegion
    Set-StateValue 'status' (Get-PropertyValue $Instance 'status')
    Set-StateValue 'ip' (Get-PropertyValue $Instance 'ip')
    Save-WatcherState
}

function Get-AvailabilityTransition {
    param(
        [bool]$PreviousAvailable,
        [bool]$CurrentAvailable
    )

    if (-not $PreviousAvailable -and $CurrentAvailable) {
        return 'became_available'
    }
    if ($PreviousAvailable -and -not $CurrentAvailable) {
        return 'became_unavailable'
    }
    return 'none'
}

function Test-RegionAvailability {
    param(
        [Parameter(Mandatory = $true)][object]$InstanceTypesItem,
        [string]$Region = $script:TargetRegion
    )

    foreach ($candidate in @(Get-PropertyValue $InstanceTypesItem 'regions_with_capacity_available')) {
        if ((Get-PropertyValue $candidate 'name') -eq $Region) {
            return $true
        }
    }
    return $false
}

function Find-Gh200InstanceType {
    param([Parameter(Mandatory = $true)][object]$InstanceTypes)

    $items = @()
    if ($InstanceTypes -is [System.Collections.IDictionary]) {
        foreach ($value in $InstanceTypes.Values) {
            $items += ,$value
        }
    }
    else {
        foreach ($property in $InstanceTypes.PSObject.Properties) {
            $items += ,$property.Value
        }
    }

    $gh200Matches = @()
    foreach ($item in $items) {
        $instanceType = Get-PropertyValue $item 'instance_type'
        $specs = Get-PropertyValue $instanceType 'specs'
        $gpuCount = Get-PropertyValue $specs 'gpus'
        $gpuDescription = [string](Get-PropertyValue $instanceType 'gpu_description')
        $description = [string](Get-PropertyValue $instanceType 'description')
        if ($gpuCount -eq 1 -and ($gpuDescription -match '(?i)GH200' -or $description -match '(?i)GH200')) {
            $match = [pscustomobject]@{
                Item = $item
                InstanceType = $instanceType
                Name = [string](Get-PropertyValue $instanceType 'name')
                Description = $description
                PriceCentsPerHour = [int](Get-PropertyValue $instanceType 'price_cents_per_hour')
                Available = (Test-RegionAvailability -InstanceTypesItem $item)
            }
            $gh200Matches = @($gh200Matches) + @($match)
        }
    }
    if ($gh200Matches.Count -ne 1) {
        throw "Expected exactly one single-GPU GH200 instance type; found $($gh200Matches.Count)."
    }
    if ([string]::IsNullOrWhiteSpace($gh200Matches[0].Name)) {
        throw 'The discovered GH200 instance type has no name.'
    }
    return $gh200Matches[0]
}

function New-LaunchPayload {
    param(
        [Parameter(Mandatory = $true)][string]$InstanceTypeName,
        [Parameter(Mandatory = $true)][object]$FileSystem
    )

    $fileSystemId = [string](Get-PropertyValue $FileSystem 'id')
    if ([string]::IsNullOrWhiteSpace($fileSystemId)) {
        throw 'The Lambda Cloud API filesystem ID is required.'
    }
    if ($fileSystemId -match '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$') {
        throw 'A 36-character S3 Adapter bucket UUID is not a Cloud API filesystem ID.'
    }
    return [ordered]@{
        region_name = $script:TargetRegion
        instance_type_name = $InstanceTypeName
        ssh_key_names = @($script:SshKeyName)
        file_system_mounts = @(
            [ordered]@{
                file_system_id = $fileSystemId
                mount_point = $script:FileSystemMount
            }
        )
        hostname = $script:InstanceHostname
        name = $script:InstanceName
        tags = @(
            [ordered]@{ key = 'project'; value = 'brazil-rv' },
            [ordered]@{ key = 'purpose'; value = 'research-training' },
            [ordered]@{ key = 'managed-by'; value = 'gh200-watcher' }
        )
    }
}

function Classify-LambdaFailure {
    param(
        [AllowNull()][string]$Code,
        [int]$HttpStatus,
        [bool]$IsLaunch
    )

    if ($Code -eq 'instance-operations/launch/insufficient-capacity') {
        return 'retry_capacity'
    }
    $fatalCodes = @(
        'instance-operations/launch/file-system-in-wrong-region',
        'global/invalid-parameters',
        'global/object-does-not-exist',
        'global/quota-exceeded',
        'global/invalid-api-key',
        'global/account-inactive',
        'global/invalid-address',
        'global/forbidden',
        'global/not-found'
    )
    if ($fatalCodes -contains $Code) {
        return 'fatal'
    }
    if ($HttpStatus -eq 429) {
        return 'rate_limited'
    }
    if ($HttpStatus -eq 0 -or $HttpStatus -eq 408 -or $HttpStatus -ge 500) {
        if ($IsLaunch) {
            return 'reconcile'
        }
        return 'transient'
    }
    return 'fatal'
}

function Get-LaunchWaitSeconds {
    param(
        [double[]]$AttemptStarts,
        [double]$NowSeconds
    )

    $recent = @($AttemptStarts | Where-Object {
        $_ -le $NowSeconds -and ($NowSeconds - $_) -lt $script:LaunchWindowSeconds
    } | Sort-Object)
    $eligibleAt = $NowSeconds
    if ($recent.Count -gt 0) {
        $eligibleAt = [Math]::Max($eligibleAt, $recent[-1] + $script:LaunchMinimumSeconds)
    }
    if ($recent.Count -ge $script:LaunchWindowMaximum) {
        $eligibleAt = [Math]::Max(
            $eligibleAt,
            $recent[$recent.Count - $script:LaunchWindowMaximum] + $script:LaunchWindowSeconds
        )
    }
    return [Math]::Max(0.0, $eligibleAt - $NowSeconds)
}

function Test-LaunchAttemptAllowed {
    param(
        [double[]]$AttemptStarts,
        [double]$NowSeconds
    )

    return (Get-LaunchWaitSeconds -AttemptStarts $AttemptStarts -NowSeconds $NowSeconds) -le 0.0
}

function ConvertTo-ProcessArgument {
    param([AllowEmptyString()][string]$Argument)

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }
    $builder = New-Object Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ([int]$character -eq 92) {
            $backslashes++
            continue
        }
        if ([int]$character -eq 34) {
            [void]$builder.Append(('\' * (2 * $backslashes + 1)))
            [void]$builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append(('\' * (2 * $backslashes)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function New-ProcessStartInfo {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [AllowNull()][string]$WorkingDirectory,
        [bool]$CaptureOutput = $true
    )

    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $FilePath
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $CaptureOutput
    $info.RedirectStandardError = $CaptureOutput
    if (-not [string]::IsNullOrEmpty($WorkingDirectory)) {
        $info.WorkingDirectory = $WorkingDirectory
    }
    if ($null -ne $info.PSObject.Properties['ArgumentList']) {
        foreach ($argument in $ArgumentList) {
            $info.ArgumentList.Add($argument)
        }
    }
    else {
        $info.Arguments = (@($ArgumentList | ForEach-Object { ConvertTo-ProcessArgument $_ })) -join ' '
    }
    foreach ($name in @('LAMBDA_API_KEY', 'LAMBDA_CLOUD_API_KEY')) {
        if ($info.EnvironmentVariables.ContainsKey($name)) {
            $info.EnvironmentVariables.Remove($name)
        }
    }
    return $info
}

function Invoke-CheckedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [AllowNull()][string]$WorkingDirectory,
        [ValidateRange(1, 2147483)][int]$TimeoutSeconds = 300,
        [int[]]$AllowedExitCodes = @(0),
        [switch]$NoCapture,
        [switch]$AllowTimeout
    )

    $info = New-ProcessStartInfo -FilePath $FilePath -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory -CaptureOutput (-not $NoCapture)
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    try {
        if (-not $process.Start()) {
            throw "Failed to start external process: $([IO.Path]::GetFileName($FilePath))"
        }
        $stdoutTask = $null
        $stderrTask = $null
        if (-not $NoCapture) {
            $stdoutTask = $process.StandardOutput.ReadToEndAsync()
            $stderrTask = $process.StandardError.ReadToEndAsync()
        }
        $timedOut = -not $process.WaitForExit($TimeoutSeconds * 1000)
        if ($timedOut) {
            if (-not $process.HasExited) {
                $process.Kill()
            }
            if (-not $process.WaitForExit(5000)) {
                throw "Timed-out process $([IO.Path]::GetFileName($FilePath)) could not be reaped."
            }
        }
        $process.WaitForExit()
        $stdout = if ($null -ne $stdoutTask) {
            $stdoutTask.GetAwaiter().GetResult()
        }
        else { '' }
        $stderr = if ($null -ne $stderrTask) {
            $stderrTask.GetAwaiter().GetResult()
        }
        else { '' }
        if ($timedOut) {
            $timeoutResult = [pscustomobject]@{
                ExitCode = -1
                TimedOut = $true
                StdOut = $stdout
                StdErr = $stderr
            }
            if (-not $AllowTimeout) {
                throw "External process $([IO.Path]::GetFileName($FilePath)) timed out."
            }
            return $timeoutResult
        }
        $result = [pscustomobject]@{
            ExitCode = $process.ExitCode
            TimedOut = $false
            StdOut = $stdout
            StdErr = $stderr
        }
        if ($AllowedExitCodes -notcontains $result.ExitCode) {
            throw "External process $([IO.Path]::GetFileName($FilePath)) exited with code $($result.ExitCode)."
        }
        return $result
    }
    finally {
        $process.Dispose()
    }
}

function Get-SshOptions {
    param(
        [Parameter(Mandatory = $true)][string]$PrivateKeyPath,
        [Parameter(Mandatory = $true)][string]$KnownHostsPath
    )

    return @(
        '-i', $PrivateKeyPath,
        '-o', 'IdentitiesOnly=yes',
        '-o', 'BatchMode=yes',
        '-o', 'PreferredAuthentications=publickey',
        '-o', 'PasswordAuthentication=no',
        '-o', 'KbdInteractiveAuthentication=no',
        '-o', 'ConnectionAttempts=1',
        '-o', 'StrictHostKeyChecking=accept-new',
        '-o', "UserKnownHostsFile=$KnownHostsPath",
        '-o', 'ConnectTimeout=10',
        '-o', 'ServerAliveInterval=30',
        '-o', 'ServerAliveCountMax=3'
    )
}

function New-ScpArguments {
    param(
        [Parameter(Mandatory = $true)][string]$PrivateKeyPath,
        [Parameter(Mandatory = $true)][string]$KnownHostsPath,
        [Parameter(Mandatory = $true)][string[]]$LocalPaths,
        [Parameter(Mandatory = $true)][string]$IpAddress
    )

    return @(
        @(Get-SshOptions -PrivateKeyPath $PrivateKeyPath -KnownHostsPath $KnownHostsPath) +
        @($LocalPaths) +
        @("ubuntu@${IpAddress}:/home/ubuntu/")
    )
}

function New-BootstrapSshArguments {
    param(
        [Parameter(Mandatory = $true)][string]$PrivateKeyPath,
        [Parameter(Mandatory = $true)][string]$KnownHostsPath,
        [Parameter(Mandatory = $true)][string]$IpAddress,
        [Parameter(Mandatory = $true)][string]$GitSha,
        [Parameter(Mandatory = $true)][string]$BundleSha256,
        [Parameter(Mandatory = $true)][string]$BootstrapSha256,
        [Parameter(Mandatory = $true)][string]$InstanceId
    )

    return @(
        @(Get-SshOptions -PrivateKeyPath $PrivateKeyPath -KnownHostsPath $KnownHostsPath) +
        @(
            "ubuntu@$IpAddress",
            'bash',
            '/home/ubuntu/lambda-gh200-bootstrap.sh',
            $GitSha,
            $BundleSha256,
            $BootstrapSha256,
            $InstanceId
        )
    )
}

function Get-TagMap {
    param([AllowNull()][object]$Tags)

    $map = @{}
    foreach ($tag in @($Tags)) {
        $key = [string](Get-PropertyValue $tag 'key')
        if (-not [string]::IsNullOrEmpty($key)) {
            $map[$key] = [string](Get-PropertyValue $tag 'value')
        }
    }
    return $map
}

function Test-ManagedInstanceMatch {
    param(
        [Parameter(Mandatory = $true)][object]$Instance,
        [Parameter(Mandatory = $true)][string]$InstanceTypeName
    )

    if ((Get-PropertyValue $Instance 'name') -ne $script:InstanceName) {
        return $false
    }
    $region = Get-PropertyValue (Get-PropertyValue $Instance 'region') 'name'
    if ($region -ne $script:TargetRegion) {
        return $false
    }
    $typeName = Get-PropertyValue (Get-PropertyValue $Instance 'instance_type') 'name'
    if ($typeName -ne $InstanceTypeName) {
        return $false
    }
    $tags = Get-TagMap (Get-PropertyValue $Instance 'tags')
    if ($tags.ContainsKey('project') -and $tags['project'] -ne 'brazil-rv') {
        return $false
    }
    if ($tags.ContainsKey('managed-by') -and $tags['managed-by'] -ne 'gh200-watcher') {
        return $false
    }
    return $true
}

function Test-NonterminalInstance {
    param([Parameter(Mandatory = $true)][object]$Instance)

    $status = [string](Get-PropertyValue $Instance 'status')
    return @('terminated', 'preempted') -notcontains $status
}

function Find-MatchingInstances {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Instances,
        [Parameter(Mandatory = $true)][string]$InstanceTypeName
    )

    $matches = @($Instances | Where-Object {
        (Test-NonterminalInstance $_) -and
        (Test-ManagedInstanceMatch -Instance $_ -InstanceTypeName $InstanceTypeName)
    })
    Write-Output -NoEnumerate $matches
}

function Test-ExpectedFileSystemMount {
    param(
        [Parameter(Mandatory = $true)][object]$Instance,
        [Parameter(Mandatory = $true)][string]$FileSystemId
    )

    foreach ($mount in @(Get-PropertyValue $Instance 'file_system_mounts')) {
        if ((Get-PropertyValue $mount 'file_system_id') -eq $FileSystemId -and
            (Get-PropertyValue $mount 'mount_point') -eq $script:FileSystemMount) {
            return $true
        }
    }
    return $false
}

function Assert-OperationalParameters {
    param(
        [AllowNull()][string]$SelectedMode,
        [bool]$BillingAcknowledged,
        [bool]$RunSelfTest,
        [bool]$ForgetCredential
    )

    if ($RunSelfTest -and $ForgetCredential) {
        throw '-SelfTest and -ForgetStoredApiKey cannot be combined.'
    }
    if (($RunSelfTest -or $ForgetCredential) -and
        (-not [string]::IsNullOrEmpty($SelectedMode) -or $BillingAcknowledged)) {
        throw 'Administrative switches cannot be combined with an operational mode or billing acknowledgement.'
    }
    if ($RunSelfTest -or $ForgetCredential) {
        return
    }
    if ([string]::IsNullOrEmpty($SelectedMode)) {
        throw '-Mode Notify or -Mode Launch is required.'
    }
    if ($SelectedMode -eq 'Launch' -and -not $BillingAcknowledged) {
        throw 'Launch mode requires -IUnderstandBilling.'
    }
    if ($SelectedMode -eq 'Notify' -and $BillingAcknowledged) {
        throw '-IUnderstandBilling is rejected for Notify mode.'
    }
}

function Initialize-RuntimePaths {
    $localAppData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
    if ([string]::IsNullOrWhiteSpace($localAppData)) {
        throw 'LOCALAPPDATA could not be resolved for the current Windows user.'
    }
    $script:RuntimeRoot = Join-Path $localAppData 'BrazilRV\lambda-gh200-watcher'
    $credentialDirectory = Join-Path $script:RuntimeRoot 'credential'
    $stateDirectory = Join-Path $script:RuntimeRoot 'state'
    $logsDirectory = Join-Path $script:RuntimeRoot 'logs'
    $script:KnownHostsDirectory = Join-Path $script:RuntimeRoot 'known-hosts'
    $script:ArtifactsDirectory = Join-Path $script:RuntimeRoot 'artifacts'
    foreach ($directory in @(
        $credentialDirectory, $stateDirectory, $logsDirectory,
        $script:KnownHostsDirectory, $script:ArtifactsDirectory
    )) {
        [IO.Directory]::CreateDirectory($directory) | Out-Null
    }
    $script:CredentialPath = Join-Path $credentialDirectory 'lambda-api-key.dpapi'
    $script:StatePath = Join-Path $stateDirectory 'launch-state.json'
    $script:LockPath = Join-Path $stateDirectory 'watcher.lock'
    $script:LogPath = Join-Path $logsDirectory (
        'lambda-gh200_{0}_{1}.log' -f [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'), $PID
    )
    [IO.File]::WriteAllText($script:LogPath, '', (New-Object Text.UTF8Encoding($false)))
}

function Protect-CurrentUserAcl {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$Directory
    )

    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    if ($Directory) {
        $security = New-Object Security.AccessControl.DirectorySecurity
        $inheritance = [Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit'
    }
    else {
        $security = New-Object Security.AccessControl.FileSecurity
        $inheritance = [Security.AccessControl.InheritanceFlags]::None
    }
    $security.SetOwner($sid)
    $security.SetAccessRuleProtection($true, $false)
    $rule = New-Object Security.AccessControl.FileSystemAccessRule(
        $sid,
        [Security.AccessControl.FileSystemRights]::FullControl,
        $inheritance,
        [Security.AccessControl.PropagationFlags]::None,
        [Security.AccessControl.AccessControlType]::Allow
    )
    $security.AddAccessRule($rule)
    Set-Acl -LiteralPath $Path -AclObject $security
}

function Acquire-WatcherLock {
    try {
        $script:LockStream = New-Object IO.FileStream(
            $script:LockPath,
            [IO.FileMode]::OpenOrCreate,
            [IO.FileAccess]::ReadWrite,
            [IO.FileShare]::None
        )
    }
    catch {
        throw 'Another Lambda GH200 watcher is already running for this Windows user.'
    }
}

function Enable-SleepPrevention {
    if ($null -eq ('BrazilRV.PowerState' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
namespace BrazilRV {
    public static class PowerState {
        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern uint SetThreadExecutionState(uint executionState);
    }
}
'@
    }
    $result = [BrazilRV.PowerState]::SetThreadExecutionState([uint32]2147483649)
    if ($result -eq 0) {
        throw 'Windows sleep prevention could not be enabled.'
    }
}

function Disable-SleepPrevention {
    if ($null -ne ('BrazilRV.PowerState' -as [type])) {
        [void][BrazilRV.PowerState]::SetThreadExecutionState([uint32]2147483648)
    }
}

function Initialize-Notifier {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        Add-Type -AssemblyName System.Drawing
        $script:NotifyIcon = New-Object Windows.Forms.NotifyIcon
        $script:NotifyIcon.Icon = [Drawing.SystemIcons]::Information
        $script:NotifyIcon.Text = 'Brazil-RV Lambda GH200 watcher'
        $script:NotifyIcon.Visible = $true
    }
    catch {
        $script:NotifyIcon = $null
        Write-Log -Level WARN -Event 'notification_unavailable'
    }
}

function Send-WatcherNotification {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Message,
        [ValidateSet('Info', 'Warning', 'Error')][string]$Kind = 'Info',
        [switch]$Sound
    )

    try {
        if ($null -ne $script:NotifyIcon) {
            $icon = [Windows.Forms.ToolTipIcon]$Kind
            $script:NotifyIcon.ShowBalloonTip(8000, $Title, $Message, $icon)
        }
        if ($Sound) {
            if ($Kind -eq 'Error') {
                [Media.SystemSounds]::Hand.Play()
            }
            elseif ($Kind -eq 'Warning') {
                [Media.SystemSounds]::Exclamation.Play()
            }
            else {
                [Media.SystemSounds]::Asterisk.Play()
            }
        }
    }
    catch {
        Write-Log -Level WARN -Event 'notification_failed' -Fields @{ kind = $Kind }
    }
}

function Wait-ForMonotonicTime {
    param([double]$EligibleAtSeconds)

    while ($script:ApiClock.Elapsed.TotalSeconds -lt $EligibleAtSeconds) {
        $remaining = $EligibleAtSeconds - $script:ApiClock.Elapsed.TotalSeconds
        Start-Sleep -Milliseconds ([int][Math]::Max(1, [Math]::Ceiling($remaining * 1000)))
    }
}

function Wait-ApiRequestSlot {
    param(
        [switch]$Launch,
        [AllowNull()][scriptblock]$Now,
        [AllowNull()][scriptblock]$WaitUntil
    )

    if ($null -eq $Now) {
        $Now = { $script:ApiClock.Elapsed.TotalSeconds }
    }
    if ($null -eq $WaitUntil) {
        $WaitUntil = { param([double]$EligibleAtSeconds)
            Wait-ForMonotonicTime $EligibleAtSeconds
        }
    }

    if ($script:LastApiRequestStart -ge 0) {
        & $WaitUntil ($script:LastApiRequestStart + $script:ApiMinimumSeconds)
    }
    if ($Launch) {
        while ($true) {
            $nowSeconds = [double](& $Now)
            $wait = Get-LaunchWaitSeconds -AttemptStarts $script:LaunchAttemptStarts `
                -NowSeconds $nowSeconds
            if ($wait -le 0) {
                break
            }
            & $WaitUntil ($nowSeconds + $wait)
        }
    }
}

function Register-ApiRequestStart {
    param(
        [switch]$Launch,
        [AllowNull()][scriptblock]$Now,
        [AllowNull()][scriptblock]$UtcNow,
        [AllowNull()][scriptblock]$StateSaver
    )

    if ($null -eq $Now) {
        $Now = { $script:ApiClock.Elapsed.TotalSeconds }
    }
    if ($null -eq $UtcNow) {
        $UtcNow = { Get-UtcTimestamp }
    }
    if ($null -eq $StateSaver) {
        $StateSaver = { Save-WatcherState }
    }

    $started = [double](& $Now)
    $script:LastApiRequestStart = $started
    if ($Launch) {
        $script:LaunchAttemptStarts = @($script:LaunchAttemptStarts | Where-Object {
            $_ -le $started -and ($started - $_) -lt $script:LaunchWindowSeconds
        }) + @($started)
        if ($null -ne $script:State) {
            Set-StateValue 'launch_attempt_count' ([int]$script:State.launch_attempt_count + 1)
            Set-StateValue 'last_launch_attempt_utc' (& $UtcNow)
            & $StateSaver
        }
    }
    return $started
}

function Get-RetryAfterSeconds {
    param([AllowNull()][string]$Value)

    $seconds = 0.0
    if ([double]::TryParse(
        $Value,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$seconds
    )) {
        return [Math]::Max(0.0, $seconds)
    }
    $date = [DateTimeOffset]::MinValue
    if ([DateTimeOffset]::TryParse($Value, [ref]$date)) {
        return [Math]::Max(0.0, ($date - [DateTimeOffset]::UtcNow).TotalSeconds)
    }
    return 2.0
}

function Get-HttpFailureDetails {
    param([Parameter(Mandatory = $true)][Management.Automation.ErrorRecord]$ErrorRecord)

    $status = 0
    $retryAfter = $null
    $body = $null
    $response = $ErrorRecord.Exception.Response
    if ($null -ne $response) {
        try { $status = [int]$response.StatusCode } catch { }
        try { $retryAfter = [string]$response.Headers['Retry-After'] } catch { }
    }
    if ($null -ne $ErrorRecord.ErrorDetails -and
        -not [string]::IsNullOrWhiteSpace($ErrorRecord.ErrorDetails.Message)) {
        $body = $ErrorRecord.ErrorDetails.Message
    }
    if ([string]::IsNullOrWhiteSpace($body) -and $null -ne $response) {
        try {
            if ($null -ne $response.Content) {
                $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            }
        }
        catch { }
        if ([string]::IsNullOrWhiteSpace($body)) {
            try {
                $stream = $response.GetResponseStream()
                $reader = New-Object IO.StreamReader($stream)
                try { $body = $reader.ReadToEnd() } finally { $reader.Dispose() }
            }
            catch { }
        }
    }
    $code = $null
    if (-not [string]::IsNullOrWhiteSpace($body)) {
        try {
            $parsed = $body | ConvertFrom-Json
            $code = [string](Get-PropertyValue (Get-PropertyValue $parsed 'error') 'code')
        }
        catch { }
    }
    if ($status -eq 401 -and [string]::IsNullOrEmpty($code)) {
        $code = 'global/invalid-api-key'
    }
    return [pscustomobject]@{
        HttpStatus = $status
        Code = $code
        RetryAfter = $retryAfter
    }
}

function Throw-SafeLambdaFailure {
    param(
        [AllowNull()][string]$Code,
        [int]$HttpStatus
    )

    $safeCode = if ([string]::IsNullOrEmpty($Code)) { 'unclassified' } else { $Code }
    $exception = New-Object InvalidOperationException(
        "Lambda API request failed with code '$safeCode' (HTTP $HttpStatus)."
    )
    $exception.Data['LambdaCode'] = $safeCode
    $exception.Data['HttpStatus'] = $HttpStatus
    throw $exception
}

function New-LambdaRequestParameters {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('GET', 'POST')][string]$Method,
        [Parameter(Mandatory = $true)][string]$Path,
        [AllowNull()][object]$Body
    )

    $parameters = @{
        Uri = $script:ApiBaseUri + $Path
        Method = $Method
        Headers = $script:ApiHeaders
        UseBasicParsing = $true
        ErrorAction = 'Stop'
        TimeoutSec = $script:HttpTimeoutSeconds
    }
    if ($null -ne $Body) {
        $parameters['ContentType'] = 'application/json'
        $parameters['Body'] = $Body | ConvertTo-Json -Depth 10 -Compress
    }
    return $parameters
}

function Invoke-LambdaApi {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('GET', 'POST')][string]$Method,
        [Parameter(Mandatory = $true)][string]$Path,
        [AllowNull()][object]$Body,
        [switch]$Launch,
        [AllowNull()][scriptblock]$BeforeRequest,
        [AllowNull()][scriptblock]$WaitForSlot,
        [AllowNull()][scriptblock]$RequestStartRegistrar,
        [AllowNull()][scriptblock]$WebRequestInvoker,
        [AllowNull()][scriptblock]$Sleep,
        [AllowNull()][scriptblock]$Logger
    )

    if ($null -eq $WaitForSlot) {
        $WaitForSlot = { param([bool]$IsLaunch)
            Wait-ApiRequestSlot -Launch:$IsLaunch
        }
    }
    if ($null -eq $RequestStartRegistrar) {
        $RequestStartRegistrar = { param([bool]$IsLaunch)
            Register-ApiRequestStart -Launch:$IsLaunch
        }
    }
    if ($null -eq $WebRequestInvoker) {
        $WebRequestInvoker = { param([hashtable]$RequestParameters)
            Invoke-WebRequest @RequestParameters
        }
    }
    if ($null -eq $Sleep) {
        $Sleep = { param([double]$Seconds)
            Start-Sleep -Milliseconds ([int][Math]::Ceiling($Seconds * 1000))
        }
    }
    if ($null -eq $Logger) {
        $Logger = { param([string]$Level, [string]$Event, [hashtable]$Fields)
            Write-Log -Level $Level -Event $Event -Fields $Fields
        }
    }

    while ($true) {
        $parameters = New-LambdaRequestParameters -Method $Method -Path $Path -Body $Body
        & $WaitForSlot ([bool]$Launch)
        if ($null -ne $BeforeRequest) {
            & $BeforeRequest
        }
        $requestStarted = [double](& $RequestStartRegistrar ([bool]$Launch))
        try {
            $response = & $WebRequestInvoker $parameters
            $parsed = $response.Content | ConvertFrom-Json
            if ($null -eq $parsed.PSObject.Properties['data']) {
                Throw-SafeLambdaFailure -Code 'client/invalid-response' -HttpStatus ([int]$response.StatusCode)
            }
            $script:TransientGetBackoffIndex = 0
            if ($script:RateLimitBackoffIndex -gt 0) {
                $script:RateLimitHealthySuccessCount++
                if ($script:RateLimitHealthySuccessCount -ge $script:RateLimitRecoverySuccesses) {
                    $script:RateLimitBackoffIndex = [Math]::Max(
                        0,
                        $script:RateLimitBackoffIndex - 1
                    )
                    $script:RateLimitHealthySuccessCount = 0
                    if ($script:RateLimitBackoffIndex -eq 0) {
                        $script:RateLimitLastLoggedDelay = $null
                    }
                }
            }
            else {
                $script:RateLimitHealthySuccessCount = 0
            }
            return [pscustomobject]@{
                Succeeded = $true
                Data = $parsed.data
                RequestStartedSeconds = $requestStarted
                Classification = 'success'
                ErrorCode = $null
                HttpStatus = [int]$response.StatusCode
            }
        }
        catch {
            if ($null -ne $_.Exception.Data['LambdaCode']) {
                throw
            }
            $failure = Get-HttpFailureDetails $_
            $classification = Classify-LambdaFailure -Code $failure.Code `
                -HttpStatus $failure.HttpStatus -IsLaunch ([bool]$Launch)
            if ($classification -eq 'rate_limited') {
                $index = [Math]::Min(
                    $script:RateLimitBackoffIndex,
                    $script:RateLimitBackoffSeconds.Count - 1
                )
                $localDelay = [double]$script:RateLimitBackoffSeconds[$index]
                $script:RateLimitBackoffIndex = [Math]::Min(
                    $script:RateLimitBackoffIndex + 1,
                    $script:RateLimitBackoffSeconds.Count
                )
                $script:RateLimitHealthySuccessCount = 0
                $serverDelay = Get-RetryAfterSeconds $failure.RetryAfter
                $delay = [Math]::Max($serverDelay, $localDelay)
                if ($null -eq $script:RateLimitLastLoggedDelay -or
                    $delay -ne $script:RateLimitLastLoggedDelay) {
                    & $Logger 'WARN' 'api_rate_limited' @{
                        http_status = $failure.HttpStatus
                        server_retry_after_seconds = $serverDelay
                        local_backoff_seconds = $localDelay
                        effective_delay_seconds = $delay
                        rate_limit_level = $script:RateLimitBackoffIndex
                    }
                    $script:RateLimitLastLoggedDelay = $delay
                }
                & $Sleep $delay
                continue
            }
            if ($Launch) {
                return [pscustomobject]@{
                    Succeeded = $false
                    Data = $null
                    RequestStartedSeconds = $requestStarted
                    Classification = $classification
                    ErrorCode = $failure.Code
                    HttpStatus = $failure.HttpStatus
                }
            }
            if ($classification -eq 'transient') {
                $index = [Math]::Min(
                    $script:TransientGetBackoffIndex,
                    $script:TransientGetBackoffSeconds.Count - 1
                )
                $delay = $script:TransientGetBackoffSeconds[$index]
                $script:TransientGetBackoffIndex++
                & $Logger 'WARN' 'api_get_transient_failure' @{
                    http_status = $failure.HttpStatus
                    backoff_seconds = $delay
                }
                & $Sleep $delay
                continue
            }
            Throw-SafeLambdaFailure -Code $failure.Code -HttpStatus $failure.HttpStatus
        }
    }
}

function Convert-SecureStringToApiCredential {
    param([Parameter(Mandatory = $true)][Security.SecureString]$SecureValue)

    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureValue)
    $plainText = $null
    try {
        $plainText = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
        $plainText = $plainText.Trim()
        if ([string]::IsNullOrEmpty($plainText) -or $plainText -match '\s') {
            throw 'The Lambda API key must be nonempty and contain no whitespace.'
        }
        $trimmedSecure = New-Object Security.SecureString
        foreach ($character in $plainText.ToCharArray()) {
            $trimmedSecure.AppendChar($character)
        }
        $trimmedSecure.MakeReadOnly()
        Register-SecretValue $plainText
        $script:ApiHeaders = @{
            Accept = 'application/json'
            Authorization = "Bearer $plainText"
        }
        return $trimmedSecure
    }
    finally {
        if ($pointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        }
        $plainText = $null
    }
}

function Save-DpapiCredential {
    param([Parameter(Mandatory = $true)][Security.SecureString]$SecureValue)

    $directory = Split-Path -Parent $script:CredentialPath
    Protect-CurrentUserAcl -Path $directory -Directory
    $encrypted = ConvertFrom-SecureString $SecureValue
    try {
        [IO.File]::WriteAllText(
            $script:CredentialPath,
            $encrypted,
            (New-Object Text.UTF8Encoding($false))
        )
        Protect-CurrentUserAcl -Path $script:CredentialPath
    }
    catch {
        if ([IO.File]::Exists($script:CredentialPath)) {
            [IO.File]::Delete($script:CredentialPath)
        }
        throw
    }
    finally {
        $encrypted = $null
    }
}

function Initialize-ApiCredential {
    if ([IO.File]::Exists($script:CredentialPath)) {
        try {
            Protect-CurrentUserAcl -Path (Split-Path -Parent $script:CredentialPath) -Directory
            Protect-CurrentUserAcl -Path $script:CredentialPath
            $encrypted = [IO.File]::ReadAllText($script:CredentialPath, [Text.Encoding]::UTF8)
            $secure = ConvertTo-SecureString $encrypted
            [void](Convert-SecureStringToApiCredential $secure)
            $encrypted = $null
            return
        }
        catch {
            throw 'The stored DPAPI Lambda API key could not be secured and decrypted for the current Windows user.'
        }
    }

    $entered = Read-Host 'Enter the Lambda Cloud API key' -AsSecureString
    $trimmedSecure = Convert-SecureStringToApiCredential $entered
    [void](Invoke-LambdaApi -Method GET -Path '/instances')
    $answer = Read-Host 'Type SAVE to store this API key encrypted for the current Windows user.'
    if ($answer -ceq 'SAVE') {
        Save-DpapiCredential $trimmedSecure
        Write-Host "Stored DPAPI credential: $script:CredentialPath"
    }
}

function Clear-ApiCredential {
    if ($null -ne $script:ApiHeaders) {
        $script:ApiHeaders.Clear()
    }
    $script:ApiHeaders = $null
    $script:SecretValues.Clear()
}

function Restore-LaunchLimiterFromState {
    if ($null -eq $script:State -or $null -eq $script:State.last_launch_attempt_utc) {
        return
    }
    try {
        $last = [DateTimeOffset]::Parse([string]$script:State.last_launch_attempt_utc)
        $age = ([DateTimeOffset]::UtcNow - $last).TotalSeconds
        if ($age -ge 0 -and $age -lt $script:LaunchWindowSeconds) {
            $count = [Math]::Min([int]$script:State.launch_attempt_count, $script:LaunchWindowMaximum)
            $synthetic = $script:ApiClock.Elapsed.TotalSeconds - $age
            if ($count -gt 0) {
                $script:LaunchAttemptStarts = @(1..$count | ForEach-Object { $synthetic })
            }
        }
    }
    catch {
        throw 'Stored launch-attempt time is invalid.'
    }
}

function Initialize-WatcherState {
    param([Parameter(Mandatory = $true)][string]$WatcherMode)

    $script:PersistWatcherState = $WatcherMode -eq 'Launch'
    $loaded = if ($WatcherMode -eq 'Launch') { Read-JsonFile $script:StatePath } else { $null }
    if ($null -eq $loaded) {
        $script:State = New-WatcherState $WatcherMode
    }
    else {
        $state = New-WatcherState $WatcherMode
        foreach ($name in $script:StateFieldNames) {
            $value = Get-PropertyValue $loaded $name
            if ($null -ne $value) {
                $state.$name = $value
            }
        }
        $state.script_version = $script:ScriptVersion
        $state.mode = $WatcherMode
        $state.watcher_started_at_utc = Get-UtcTimestamp
        $script:State = $state
    }
    Save-WatcherState
}

function Get-AvailabilitySnapshot {
    $result = Invoke-LambdaApi -Method GET -Path '/instance-types'
    $discovery = Find-Gh200InstanceType $result.Data
    return [pscustomobject]@{
        Discovery = $discovery
        RequestStartedSeconds = $result.RequestStartedSeconds
    }
}

function Update-AvailabilityObservation {
    param([Parameter(Mandatory = $true)][object]$Discovery)

    $previous = $false
    if ($null -ne $script:State.last_availability) {
        $previous = [bool]$script:State.last_availability
    }
    $current = [bool]$Discovery.Available
    $transition = Get-AvailabilityTransition -PreviousAvailable $previous -CurrentAvailable $current
    Set-StateValue 'last_availability' $current
    Set-StateValue 'last_successful_poll_utc' (Get-UtcTimestamp)
    if ($transition -ne 'none') {
        Set-StateValue 'last_availability_change_utc' (Get-UtcTimestamp)
        if ($transition -eq 'became_available') {
            Write-Log -Level INFO -Event 'gh200_available' -Fields @{
                region = $script:TargetRegion
                instance_type = $Discovery.Name
                hourly_price = ([decimal]$Discovery.PriceCentsPerHour / 100)
            }
            Send-WatcherNotification -Title 'Lambda GH200 available' `
                -Message "A single-GPU GH200 is available in $script:TargetRegion." -Sound
        }
        else {
            Write-Log -Level INFO -Event 'gh200_unavailable' -Fields @{
                region = $script:TargetRegion
                instance_type = $Discovery.Name
            }
            Send-WatcherNotification -Title 'Lambda GH200 unavailable' `
                -Message "GH200 capacity in $script:TargetRegion is no longer available."
        }
        Save-WatcherState
    }
    return $transition
}

function Show-DiscoveredTarget {
    param([Parameter(Mandatory = $true)][object]$Discovery)

    $price = [decimal]$Discovery.PriceCentsPerHour / 100
    Write-Host "Target region:             $script:TargetRegion"
    Write-Host "Instance type:             $($Discovery.Name)"
    Write-Host "Description:               $($Discovery.Description)"
    Write-Host ('Current hourly price:      ${0:N2}' -f $price)
    Write-Host "Availability poll interval: $($script:AvailabilityPollSeconds) seconds"
    Write-Log -Level INFO -Event 'target_discovered' -Fields @{
        region = $script:TargetRegion
        instance_type = $Discovery.Name
        hourly_price = $price
    }
}

function Start-NotifyWatcher {
    $snapshot = Get-AvailabilitySnapshot
    Show-DiscoveredTarget $snapshot.Discovery
    [void](Update-AvailabilityObservation $snapshot.Discovery)
    $lastPollStart = $snapshot.RequestStartedSeconds
    while ($true) {
        Wait-ForMonotonicTime ($lastPollStart + $script:AvailabilityPollSeconds)
        $snapshot = Get-AvailabilitySnapshot
        $lastPollStart = $snapshot.RequestStartedSeconds
        [void](Update-AvailabilityObservation $snapshot.Discovery)
    }
}

function Get-ExactlyOneNamedResource {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][object[]]$Resources,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Kind
    )

    $matches = @($Resources | Where-Object { (Get-PropertyValue $_ 'name') -eq $Name })
    if ($matches.Count -ne 1) {
        throw "Expected exactly one $Kind named '$Name'; found $($matches.Count)."
    }
    return $matches[0]
}

function Get-GitExecutable {
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($null -eq $git) {
        $git = Get-Command git -ErrorAction Stop
    }
    return $git.Source
}

function Get-CleanMainGitSha {
    param([Parameter(Mandatory = $true)][string]$GitPath)

    if (-not (Test-Path -LiteralPath (Join-Path $script:RepositoryRoot '.git') -PathType Container)) {
        throw "Repository is not a Git worktree: $script:RepositoryRoot"
    }
    $status = Invoke-CheckedProcess -FilePath $GitPath -ArgumentList @('status', '--porcelain') `
        -WorkingDirectory $script:RepositoryRoot
    if (-not [string]::IsNullOrEmpty($status.StdOut)) {
        throw 'Launch requires a clean Git worktree.'
    }
    $head = (Invoke-CheckedProcess -FilePath $GitPath -ArgumentList @('rev-parse', 'HEAD') `
        -WorkingDirectory $script:RepositoryRoot).StdOut.Trim()
    if ($head -notmatch '^[0-9a-fA-F]{40,64}$') {
        throw 'git rev-parse HEAD did not return a valid commit SHA.'
    }
    [void](Invoke-CheckedProcess -FilePath $GitPath `
        -ArgumentList @('show-ref', '--verify', '--quiet', 'refs/heads/main') `
        -WorkingDirectory $script:RepositoryRoot)
    $main = (Invoke-CheckedProcess -FilePath $GitPath -ArgumentList @('rev-parse', 'main') `
        -WorkingDirectory $script:RepositoryRoot).StdOut.Trim()
    if (-not [string]::Equals($main, $head, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'HEAD must equal the local main branch before freezing launch artifacts.'
    }
    return $head.ToLowerInvariant()
}

function Test-Utf8LfFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [AllowNull()][object]$ExpectedText
    )

    $bytes = [IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and
        $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        return $false
    }
    if ($bytes -contains 13) {
        return $false
    }
    try {
        $utf8 = New-Object Text.UTF8Encoding -ArgumentList $false, $true
        $text = $utf8.GetString($bytes)
    }
    catch {
        return $false
    }
    if ($null -ne $ExpectedText -and
        -not [string]::Equals($text, $ExpectedText, [StringComparison]::Ordinal)) {
        return $false
    }
    return $true
}

function New-FrozenLaunchArtifacts {
    param(
        [Parameter(Mandatory = $true)][string]$GitSha,
        [Parameter(Mandatory = $true)][string]$BundlePath,
        [Parameter(Mandatory = $true)][string]$BundleSha256,
        [Parameter(Mandatory = $true)][string]$BootstrapPath,
        [Parameter(Mandatory = $true)][string]$BootstrapSha256
    )

    $values = New-Object 'System.Collections.Generic.Dictionary[string,object]'
    $values.Add('GitSha', $GitSha)
    $values.Add('BundlePath', $BundlePath)
    $values.Add('BundleSha256', $BundleSha256)
    $values.Add('BootstrapPath', $BootstrapPath)
    $values.Add('BootstrapSha256', $BootstrapSha256)
    return New-Object 'System.Collections.ObjectModel.ReadOnlyDictionary[string,object]' `
        -ArgumentList (, $values)
}

function Get-FrozenLaunchArtifacts {
    $gitPath = Get-GitExecutable
    $gitSha = Get-CleanMainGitSha -GitPath $gitPath
    $artifactDirectory = Join-Path $script:ArtifactsDirectory $gitSha
    [IO.Directory]::CreateDirectory($artifactDirectory) | Out-Null
    $bundlePath = Join-Path $artifactDirectory ("brazil-rv_{0}.bundle" -f $gitSha)
    $bootstrapPath = Join-Path $artifactDirectory 'lambda-gh200-bootstrap.sh'

    $objectName = '{0}:ops/lambda-gh200-bootstrap.sh' -f $gitSha
    $committed = (Invoke-CheckedProcess -FilePath $gitPath `
        -ArgumentList @('show', $objectName) -WorkingDirectory $script:RepositoryRoot).StdOut
    $committed = $committed.Replace("`r`n", "`n").Replace("`r", "`n")
    if (-not [IO.File]::Exists($bootstrapPath)) {
        [IO.File]::WriteAllText(
            $bootstrapPath,
            $committed,
            (New-Object Text.UTF8Encoding($false))
        )
    }
    if (-not (Test-Utf8LfFile -Path $bootstrapPath -ExpectedText $committed)) {
        throw 'Frozen bootstrap is not the exact committed UTF-8 LF Git object.'
    }

    if (-not [IO.File]::Exists($bundlePath)) {
        $temporaryBundle = Join-Path $artifactDirectory (
            '.brazil-rv.{0}.{1}.tmp' -f $PID, [Guid]::NewGuid().ToString('N')
        )
        try {
            [void](Invoke-CheckedProcess -FilePath $gitPath `
                -ArgumentList @('bundle', 'create', $temporaryBundle, 'refs/heads/main') `
                -WorkingDirectory $script:RepositoryRoot -TimeoutSeconds 900)
            [void](Invoke-CheckedProcess -FilePath $gitPath `
                -ArgumentList @('bundle', 'verify', $temporaryBundle) `
                -WorkingDirectory $script:RepositoryRoot -TimeoutSeconds 900)
            $heads = (Invoke-CheckedProcess -FilePath $gitPath `
                -ArgumentList @('bundle', 'list-heads', $temporaryBundle) `
                -WorkingDirectory $script:RepositoryRoot).StdOut
            if ($heads -notmatch "(?m)^$gitSha\s+refs/heads/main\s*$") {
                throw 'The new Git bundle does not contain the expected main commit.'
            }
            if ([IO.File]::Exists($bundlePath)) {
                throw 'The frozen bundle path appeared concurrently; it was not replaced.'
            }
            [IO.File]::Move($temporaryBundle, $bundlePath)
        }
        finally {
            if ([IO.File]::Exists($temporaryBundle)) {
                [IO.File]::Delete($temporaryBundle)
            }
        }
    }
    [void](Invoke-CheckedProcess -FilePath $gitPath `
        -ArgumentList @('bundle', 'verify', $bundlePath) `
        -WorkingDirectory $script:RepositoryRoot -TimeoutSeconds 900)
    $bundleHeads = (Invoke-CheckedProcess -FilePath $gitPath `
        -ArgumentList @('bundle', 'list-heads', $bundlePath) `
        -WorkingDirectory $script:RepositoryRoot).StdOut
    if ($bundleHeads -notmatch "(?m)^$gitSha\s+refs/heads/main\s*$") {
        throw 'The frozen Git bundle does not contain the expected main commit.'
    }

    return New-FrozenLaunchArtifacts -GitSha $gitSha -BundlePath $bundlePath `
        -BundleSha256 (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash.ToLowerInvariant() `
        -BootstrapPath $bootstrapPath `
        -BootstrapSha256 (Get-FileHash -LiteralPath $bootstrapPath -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-StateLaunchArtifacts {
    $identity = @(
        [string]$script:State.launch_git_sha,
        [string]$script:State.launch_bundle_sha256,
        [string]$script:State.launch_bootstrap_sha256
    )
    $present = @($identity | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($present.Count -eq 0) {
        return $null
    }
    if ($present.Count -ne 3) {
        throw 'Stored launch artifact identity is incomplete.'
    }
    $gitSha = $identity[0].ToLowerInvariant()
    if ($gitSha -notmatch '^[0-9a-f]{40,64}$' -or
        $identity[1] -notmatch '^[0-9a-fA-F]{64}$' -or
        $identity[2] -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'Stored launch artifact identity is invalid.'
    }
    $artifactDirectory = Join-Path $script:ArtifactsDirectory $gitSha
    return New-FrozenLaunchArtifacts -GitSha $gitSha `
        -BundlePath (Join-Path $artifactDirectory ("brazil-rv_{0}.bundle" -f $gitSha)) `
        -BundleSha256 $identity[1].ToLowerInvariant() `
        -BootstrapPath (Join-Path $artifactDirectory 'lambda-gh200-bootstrap.sh') `
        -BootstrapSha256 $identity[2].ToLowerInvariant()
}

function Assert-FrozenLaunchArtifacts {
    param([Parameter(Mandatory = $true)][object]$Artifacts)

    foreach ($path in @($Artifacts.BundlePath, $Artifacts.BootstrapPath)) {
        if (-not [IO.File]::Exists($path)) {
            throw "Frozen launch artifact is missing: $path"
        }
    }
    $bundleHash = (Get-FileHash -LiteralPath $Artifacts.BundlePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $bootstrapHash = (Get-FileHash -LiteralPath $Artifacts.BootstrapPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [string]::Equals(
        $bundleHash,
        [string]$Artifacts.BundleSha256,
        [StringComparison]::Ordinal
    )) {
        throw 'Frozen Git bundle hash changed after preflight.'
    }
    if (-not [string]::Equals(
        $bootstrapHash,
        [string]$Artifacts.BootstrapSha256,
        [StringComparison]::Ordinal
    )) {
        throw 'Frozen bootstrap hash changed after preflight.'
    }
    if (-not (Test-Utf8LfFile -Path $Artifacts.BootstrapPath -ExpectedText $null)) {
        throw 'Frozen bootstrap is not UTF-8 without BOM using LF line endings.'
    }
}

function Get-PublicKeyIdentity {
    param([Parameter(Mandatory = $true)][string]$Text)

    $parts = @($Text.Trim() -split '\s+')
    if ($parts.Count -lt 2 -or [string]::IsNullOrWhiteSpace($parts[0]) -or
        [string]::IsNullOrWhiteSpace($parts[1])) {
        throw 'SSH public key text is invalid.'
    }
    return "$($parts[0]) $($parts[1])"
}

function Assert-PublicKeyIdentityMatch {
    param(
        [Parameter(Mandatory = $true)][string]$CloudPublicKey,
        [Parameter(Mandatory = $true)][string]$LocalPublicKey
    )

    $cloudIdentity = Get-PublicKeyIdentity $CloudPublicKey
    $localIdentity = Get-PublicKeyIdentity $LocalPublicKey
    if (-not [string]::Equals(
        $cloudIdentity,
        $localIdentity,
        [StringComparison]::Ordinal
    )) {
        throw 'The Lambda brazil-rv SSH public key does not match the local key.'
    }
    return $localIdentity
}

function Test-SshAgentContainsIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$AgentOutput,
        [Parameter(Mandatory = $true)][string]$PublicKeyIdentity
    )

    foreach ($line in @($AgentOutput -split "`r?`n")) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $listedIdentity = Get-PublicKeyIdentity $line
            if ([string]::Equals(
                $listedIdentity,
                $PublicKeyIdentity,
                [StringComparison]::Ordinal
            )) {
                return $true
            }
        }
        catch { }
    }
    return $false
}

function Start-WindowsSshAgent {
    try {
        $service = Get-Service -Name 'ssh-agent' -ErrorAction Stop
        if ([string]$service.Status -ne 'Running') {
            Start-Service -Name 'ssh-agent' -ErrorAction Stop
            $service = Get-Service -Name 'ssh-agent' -ErrorAction Stop
        }
        if ([string]$service.Status -ne 'Running') {
            throw 'The service did not enter the Running state.'
        }
    }
    catch {
        throw (
            'Windows ssh-agent is unavailable. Open PowerShell as Administrator once and run: ' +
            'Set-Service -Name ssh-agent -StartupType Manual; Start-Service ssh-agent'
        )
    }
}

function Get-SshAgentListing {
    param(
        [Parameter(Mandatory = $true)][string]$SshAddPath,
        [AllowNull()][scriptblock]$ProcessInvoker,
        [AllowNull()][scriptblock]$AgentStarter
    )

    if ($null -eq $ProcessInvoker) {
        $ProcessInvoker = {
            param([string]$Executable)
            Invoke-CheckedProcess -FilePath $Executable -ArgumentList @('-L') `
                -AllowedExitCodes @(0, 1, 2)
        }
    }
    if ($null -eq $AgentStarter) {
        $AgentStarter = { Start-WindowsSshAgent }
    }

    $listed = & $ProcessInvoker $SshAddPath
    if ($listed.ExitCode -eq 2) {
        & $AgentStarter
        $listed = & $ProcessInvoker $SshAddPath
    }
    if ($listed.ExitCode -eq 2) {
        throw 'ssh-add still cannot communicate with Windows ssh-agent after service recovery.'
    }
    return $listed
}

function Get-SshPreflight {
    $sshDirectory = Join-Path $env:USERPROFILE '.ssh'
    $privateKey = Join-Path $sshDirectory 'lambda_brazil_rv_ed25519'
    $publicKey = $privateKey + '.pub'
    foreach ($path in @($privateKey, $publicKey)) {
        if (-not [IO.File]::Exists($path)) {
            throw "Required SSH key file is missing: $path"
        }
    }
    $ssh = Get-Command ssh.exe -ErrorAction Stop
    $scp = Get-Command scp.exe -ErrorAction Stop
    $sshAdd = Get-Command ssh-add.exe -ErrorAction Stop
    $identity = Get-PublicKeyIdentity (
        [IO.File]::ReadAllText($publicKey, [Text.Encoding]::UTF8)
    )
    $listed = Get-SshAgentListing -SshAddPath $sshAdd.Source
    if ($listed.ExitCode -ne 0 -or
        -not (Test-SshAgentContainsIdentity $listed.StdOut $identity)) {
        Write-Host 'Loading the Brazil-RV SSH key into ssh-agent. Enter its passphrase if prompted.'
        [void](Invoke-CheckedProcess -FilePath $sshAdd.Source -ArgumentList @($privateKey) `
            -TimeoutSeconds 300 -NoCapture)
        $listed = Get-SshAgentListing -SshAddPath $sshAdd.Source
    }
    if (-not (Test-SshAgentContainsIdentity $listed.StdOut $identity)) {
        throw 'The brazil-rv public key is not loaded in ssh-agent.'
    }
    return [pscustomobject]@{
        PrivateKeyPath = $privateKey
        PublicKeyPath = $publicKey
        PublicKeyIdentity = $identity
        SshPath = $ssh.Source
        ScpPath = $scp.Source
        SshAddPath = $sshAdd.Source
    }
}

function Assert-LaunchSshIdentity {
    param([Parameter(Mandatory = $true)][object]$SshPreflight)

    foreach ($path in @($SshPreflight.PrivateKeyPath, $SshPreflight.PublicKeyPath)) {
        if (-not [IO.File]::Exists($path)) {
            throw "Required SSH key file disappeared before launch: $path"
        }
    }
    $currentIdentity = Get-PublicKeyIdentity (
        [IO.File]::ReadAllText($SshPreflight.PublicKeyPath, [Text.Encoding]::UTF8)
    )
    if (-not [string]::Equals(
        $currentIdentity,
        [string]$SshPreflight.PublicKeyIdentity,
        [StringComparison]::Ordinal
    )) {
        throw 'The local SSH public key identity changed after preflight.'
    }
    $listed = Get-SshAgentListing -SshAddPath $SshPreflight.SshAddPath
    if (-not (Test-SshAgentContainsIdentity $listed.StdOut $currentIdentity)) {
        throw 'The preflight SSH identity is no longer loaded in ssh-agent.'
    }
}

function Test-FirewallRuleAllowsSsh {
    param([Parameter(Mandatory = $true)][object]$Rule)

    $protocol = [string](Get-PropertyValue $Rule 'protocol')
    if ([string]::Equals($protocol, 'all', [StringComparison]::Ordinal)) {
        return $true
    }
    if (-not [string]::Equals($protocol, 'tcp', [StringComparison]::Ordinal)) {
        return $false
    }
    $ports = @(Get-PropertyValue $Rule 'port_range')
    if ($ports.Count -ne 2) {
        return $false
    }
    try {
        return [int]$ports[0] -le 22 -and [int]$ports[1] -ge 22
    }
    catch {
        return $false
    }
}

function ConvertTo-SafeDisplayText {
    param([AllowNull()][object]$Value)

    if ($null -eq $Value) {
        return ''
    }
    return ([string]$Value).Replace("`r", ' ').Replace("`n", ' ').Replace("`t", ' ')
}

function Assert-GlobalFirewallAllowsSsh {
    param([Parameter(Mandatory = $true)][object]$Ruleset)

    $matching = @(
        @(Get-PropertyValue $Ruleset 'rules') |
            Where-Object { Test-FirewallRuleAllowsSsh $_ }
    )
    if ($matching.Count -eq 0) {
        throw 'The global Lambda firewall ruleset does not allow inbound SSH on TCP port 22.'
    }
    foreach ($rule in $matching) {
        $protocol = ConvertTo-SafeDisplayText (Get-PropertyValue $rule 'protocol')
        $ports = @(Get-PropertyValue $rule 'port_range')
        $portRange = if ($ports.Count -eq 2) {
            '{0}-{1}' -f [int]$ports[0], [int]$ports[1]
        }
        else {
            ''
        }
        $source = ConvertTo-SafeDisplayText (Get-PropertyValue $rule 'source_network')
        $description = ConvertTo-SafeDisplayText (Get-PropertyValue $rule 'description')
        Write-Host "  Firewall rule: protocol=$protocol ports=$portRange source=$source description=$description"
        Write-Log -Level INFO -Event 'firewall_rule' -Fields @{
            protocol = $protocol
            port_range = $portRange
            source_network = $source
            description = $description
        }
    }
}

function Invoke-LaunchPreflight {
    $instanceTypesResult = Invoke-LambdaApi -Method GET -Path '/instance-types'
    $discovery = Find-Gh200InstanceType $instanceTypesResult.Data
    $instances = @((Invoke-LambdaApi -Method GET -Path '/instances').Data)
    $recordedId = [string]$script:State.instance_id
    if (-not [string]::IsNullOrEmpty($recordedId)) {
        $recordedCurrent = @($instances | Where-Object {
            (Get-PropertyValue $_ 'id') -eq $recordedId -and (Test-NonterminalInstance $_)
        })
        if ($recordedCurrent.Count -gt 1) {
            throw 'The recorded instance ID appeared more than once in the Lambda response.'
        }
        if ($recordedCurrent.Count -eq 1 -and
            -not (Test-ManagedInstanceMatch $recordedCurrent[0] $discovery.Name)) {
            throw 'The recorded nonterminal instance no longer matches the fixed project launch identity.'
        }
    }
    $matching = Find-MatchingInstances -Instances $instances -InstanceTypeName $discovery.Name
    if ($matching.Count -gt 1) {
        throw 'More than one matching nonterminal brazil-rv-gh200 instance exists.'
    }
    $adopted = if ($matching.Count -eq 1) { $matching[0] } else { $null }
    if ($null -eq $adopted -and -not [string]::IsNullOrEmpty($recordedId)) {
        Reset-ActiveState
        Save-WatcherState
    }
    elseif ($null -ne $adopted) {
        Set-AdoptedInstanceState -Instance $adopted -InstanceTypeName $discovery.Name
    }

    $fileSystems = @((Invoke-LambdaApi -Method GET -Path '/file-systems').Data)
    $fileSystem = Get-ExactlyOneNamedResource -Resources $fileSystems `
        -Name $script:FileSystemName -Kind 'filesystem'
    $fileSystemId = [string](Get-PropertyValue $fileSystem 'id')
    $fileSystemRegion = Get-PropertyValue (Get-PropertyValue $fileSystem 'region') 'name'
    if ($fileSystemRegion -ne $script:TargetRegion) {
        throw "Filesystem '$script:FileSystemName' is not in $script:TargetRegion."
    }
    if ([bool](Get-PropertyValue $fileSystem 'is_in_use')) {
        if ($null -eq $adopted -or -not (Test-ExpectedFileSystemMount $adopted $fileSystemId)) {
            throw "Filesystem '$script:FileSystemName' is already in use by another instance."
        }
    }

    $sshKeys = @((Invoke-LambdaApi -Method GET -Path '/ssh-keys').Data)
    $cloudSshKey = Get-ExactlyOneNamedResource -Resources $sshKeys `
        -Name $script:SshKeyName -Kind 'SSH key'
    $ssh = Get-SshPreflight
    [void](Assert-PublicKeyIdentityMatch `
        -CloudPublicKey ([string](Get-PropertyValue $cloudSshKey 'public_key')) `
        -LocalPublicKey ([IO.File]::ReadAllText($ssh.PublicKeyPath, [Text.Encoding]::UTF8)))

    $globalFirewall = (Invoke-LambdaApi -Method GET -Path '/firewall-rulesets/global').Data
    Assert-GlobalFirewallAllowsSsh $globalFirewall

    $artifacts = $null
    if ($null -eq $adopted) {
        $artifacts = Get-FrozenLaunchArtifacts
        Set-StateValue 'launch_git_sha' $artifacts.GitSha
        Set-StateValue 'launch_bundle_sha256' $artifacts.BundleSha256
        Set-StateValue 'launch_bootstrap_sha256' $artifacts.BootstrapSha256
    }
    else {
        $artifacts = Get-StateLaunchArtifacts
    }
    Set-StateValue 'filesystem_id' $fileSystemId
    Set-StateValue 'instance_type_name' $discovery.Name
    Save-WatcherState

    Show-DiscoveredTarget $discovery
    Write-Host 'Launch plan:'
    Write-Host "  One $($discovery.Name) in $script:TargetRegion"
    Write-Host "  Filesystem $script:FileSystemName ($fileSystemId) at $script:FileSystemMount"
    Write-Host "  SSH key $script:SshKeyName (cloud and local identities match)"
    Write-Host "  Instance name and hostname $script:InstanceName"
    if ($null -ne $artifacts) {
        Write-Host "  Frozen Git commit $($artifacts.GitSha)"
    }
    else {
        Write-Host '  Existing instance recovery will require a valid remote marker'
    }
    Write-Host '  Repository tests, Ruff checks, and real gh200 sanity; no production training'
    Write-Warning 'A successful launch may begin billing. This watcher never terminates the instance.'

    return [pscustomobject]@{
        Discovery = $discovery
        DiscoveryRequestStartedSeconds = $instanceTypesResult.RequestStartedSeconds
        Instances = $instances
        AdoptedInstance = $adopted
        FileSystem = $fileSystem
        Artifacts = $artifacts
        Ssh = $ssh
    }
}

function Resolve-AmbiguousLaunch {
    param(
        [Parameter(Mandatory = $true)][string]$InstanceTypeName,
        [AllowNull()][scriptblock]$ApiInvoker,
        [AllowNull()][scriptblock]$Now,
        [AllowNull()][scriptblock]$Sleep
    )

    $clock = $null
    if ($null -eq $Now) {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $Now = { $clock.Elapsed.TotalSeconds }.GetNewClosure()
    }
    if ($null -eq $Sleep) {
        $Sleep = { param([double]$Seconds)
            Start-Sleep -Milliseconds ([int][Math]::Ceiling($Seconds * 1000))
        }
    }
    if ($null -eq $ApiInvoker) {
        $ApiInvoker = { (Invoke-LambdaApi -Method GET -Path '/instances').Data }
    }

    $started = [double](& $Now)
    while ($true) {
        $instances = @(& $ApiInvoker)
        $matching = Find-MatchingInstances -Instances $instances -InstanceTypeName $InstanceTypeName
        if ($matching.Count -gt 1) {
            throw 'Ambiguous launch reconciliation found multiple matching instances.'
        }
        if ($matching.Count -eq 1) {
            $resolved = $matching[0]
            Set-AdoptedInstanceState -Instance $resolved -InstanceTypeName $InstanceTypeName
            $instanceId = [string](Get-PropertyValue $resolved 'id')
            Write-Log -Level WARN -Event 'ambiguous_launch_adopted' -Fields @{
                instance_id = $instanceId
            }
            Send-WatcherNotification -Title 'Lambda GH200 launch reconciled' `
                -Message 'A matching paid instance exists and was adopted for monitoring.' -Kind Warning -Sound
            return $resolved
        }
        $elapsed = [double](& $Now) - $started
        if ($elapsed -ge $script:AmbiguousLaunchSeconds) {
            Write-Log -Level WARN -Event 'ambiguous_launch_no_instance_found'
            return $null
        }
        & $Sleep ([Math]::Min(
            $script:AmbiguousLaunchPollSeconds,
            $script:AmbiguousLaunchSeconds - $elapsed
        ))
    }
}

function Invoke-Gh200LaunchAttempt {
    param([Parameter(Mandatory = $true)][object]$Preflight)

    if ($null -eq $Preflight.Artifacts) {
        throw 'A new launch cannot proceed without frozen launch artifacts.'
    }
    $payload = New-LaunchPayload -InstanceTypeName $Preflight.Discovery.Name `
        -FileSystem $Preflight.FileSystem
    $beforeRequest = {
        Assert-FrozenLaunchArtifacts $Preflight.Artifacts
        Assert-LaunchSshIdentity $Preflight.Ssh
        Write-Log -Level INFO -Event 'launch_attempt_sent' -Fields @{
            region = $script:TargetRegion
            instance_type = $Preflight.Discovery.Name
            filesystem_id = Get-PropertyValue $Preflight.FileSystem 'id'
        }
        Send-WatcherNotification -Title 'Lambda GH200 launch attempt' `
            -Message 'The billable GH200 launch request is being sent.' -Kind Warning -Sound
    }
    $result = Invoke-LambdaApi -Method POST -Path '/instance-operations/launch' `
        -Body $payload -Launch -BeforeRequest $beforeRequest
    if (-not $result.Succeeded) {
        if ($result.Classification -eq 'retry_capacity') {
            Write-Log -Level WARN -Event 'launch_insufficient_capacity' -Fields @{
                error_code = $result.ErrorCode
            }
            Send-WatcherNotification -Title 'Lambda GH200 launch lost' `
                -Message 'Capacity disappeared before Lambda accepted the launch; polling continues.' `
                -Kind Warning -Sound
            return $null
        }
        if ($result.Classification -eq 'reconcile') {
            Write-Log -Level WARN -Event 'launch_ambiguous' -Fields @{
                http_status = $result.HttpStatus
            }
            return Resolve-AmbiguousLaunch $Preflight.Discovery.Name
        }
        Throw-SafeLambdaFailure -Code $result.ErrorCode -HttpStatus $result.HttpStatus
    }
    $ids = @(Get-PropertyValue $result.Data 'instance_ids')
    if ($ids.Count -ne 1 -or [string]::IsNullOrWhiteSpace([string]$ids[0])) {
        throw 'Lambda launch response did not contain exactly one instance_id.'
    }
    $instanceId = [string]$ids[0]
    Set-StateValue 'instance_id' $instanceId
    Set-StateValue 'instance_name' $script:InstanceName
    Set-StateValue 'instance_type_name' $Preflight.Discovery.Name
    Set-StateValue 'region' $script:TargetRegion
    Set-StateValue 'status' 'booting'
    Save-WatcherState
    Write-Log -Level INFO -Event 'launch_accepted' -Fields @{
        instance_id = $instanceId
        instance_name = $script:InstanceName
        instance_type = $Preflight.Discovery.Name
        region = $script:TargetRegion
    }
    Send-WatcherNotification -Title 'Lambda GH200 launch accepted' `
        -Message 'Launch accepted. A paid instance may now be billing.' -Kind Warning -Sound
    return [pscustomobject]@{
        id = $instanceId
        name = $script:InstanceName
        status = 'booting'
        region = [pscustomobject]@{ name = $script:TargetRegion }
        instance_type = [pscustomobject]@{ name = $Preflight.Discovery.Name }
        tags = $payload.tags
    }
}

function Assert-ActiveInstanceConfiguration {
    param(
        [Parameter(Mandatory = $true)][object]$Instance,
        [Parameter(Mandatory = $true)][string]$InstanceTypeName,
        [Parameter(Mandatory = $true)][string]$FileSystemId
    )

    $ip = [string](Get-PropertyValue $Instance 'ip')
    if ([string]::IsNullOrWhiteSpace($ip)) {
        throw 'Active instance has no public IP address.'
    }
    if ((Get-PropertyValue (Get-PropertyValue $Instance 'region') 'name') -ne $script:TargetRegion) {
        throw 'Active instance is in the wrong region.'
    }
    if ((Get-PropertyValue (Get-PropertyValue $Instance 'instance_type') 'name') -ne $InstanceTypeName) {
        throw 'Active instance has the wrong instance type.'
    }
    if (-not (Test-ExpectedFileSystemMount $Instance $FileSystemId)) {
        throw 'Active instance does not have the expected filesystem ID and mount point.'
    }
    if (@(Get-PropertyValue $Instance 'ssh_key_names') -notcontains $script:SshKeyName) {
        throw 'Active instance does not include the brazil-rv SSH key.'
    }
}

function Wait-ForActiveInstance {
    param(
        [Parameter(Mandatory = $true)][string]$InstanceId,
        [Parameter(Mandatory = $true)][string]$InstanceTypeName,
        [Parameter(Mandatory = $true)][string]$FileSystemId
    )

    if ($InstanceId -notmatch '^[A-Za-z0-9._-]+$') {
        throw 'Instance ID contains unsafe characters.'
    }
    $started = $script:ApiClock.Elapsed.TotalSeconds
    $lastPollStart = -1.0
    $lastStatus = $null
    while (($script:ApiClock.Elapsed.TotalSeconds - $started) -lt $script:ActiveTimeoutSeconds) {
        if ($lastPollStart -ge 0) {
            Wait-ForMonotonicTime ($lastPollStart + $script:InstancePollSeconds)
        }
        $result = Invoke-LambdaApi -Method GET -Path ("/instances/{0}" -f [Uri]::EscapeDataString($InstanceId))
        $lastPollStart = $result.RequestStartedSeconds
        $instance = $result.Data
        $status = [string](Get-PropertyValue $instance 'status')
        $ip = [string](Get-PropertyValue $instance 'ip')
        Set-StateValue 'status' $status
        Set-StateValue 'ip' $(if ([string]::IsNullOrWhiteSpace($ip)) { $null } else { $ip })
        if ($status -ne $lastStatus) {
            Write-Log -Level INFO -Event 'instance_status' -Fields @{
                instance_id = $InstanceId
                status = $status
                ip = $ip
            }
            Save-WatcherState
            $lastStatus = $status
        }
        if ($status -eq 'active') {
            Assert-ActiveInstanceConfiguration -Instance $instance `
                -InstanceTypeName $InstanceTypeName -FileSystemId $FileSystemId
            Save-WatcherState
            Send-WatcherNotification -Title 'Lambda GH200 active' `
                -Message "Instance $InstanceId is active at $ip." -Sound
            Write-Host "Instance active: $InstanceId at $ip"
            return $instance
        }
        if (@('unhealthy', 'terminated', 'terminating', 'preempted') -contains $status) {
            throw "Instance $InstanceId entered failure status '$status' before bootstrap."
        }
        if ($status -ne 'booting') {
            throw "Instance $InstanceId returned unexpected status '$status'."
        }
    }
    throw "Instance $InstanceId did not become active within 20 minutes."
}

function Test-TcpPort {
    param(
        [Parameter(Mandatory = $true)][string]$HostName,
        [int]$Port = 22,
        [int]$TimeoutMilliseconds = 2000
    )

    $client = New-Object Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync($HostName, $Port)
        if (-not $task.Wait($TimeoutMilliseconds)) {
            return $false
        }
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

function Get-SshFailureClassification {
    param(
        [bool]$TcpReachable,
        [AllowNull()][object]$ProcessResult
    )

    if (-not $TcpReachable) {
        return 'tcp_unavailable'
    }
    if ($null -eq $ProcessResult) {
        return 'other_ssh_failure'
    }
    if ($ProcessResult.TimedOut) {
        return 'ssh_process_timeout'
    }

    $stderr = [string]$ProcessResult.StdErr
    if ($stderr -match 'host key verification failed|remote host identification has changed|offending .* key') {
        return 'host_key_failure'
    }
    if ($stderr -match 'connection refused') {
        return 'connection_refused'
    }
    if ($stderr -match 'connection timed out|operation timed out') {
        return 'connection_timeout'
    }
    if ($stderr -match 'connection (closed|reset)|closed by remote host') {
        return 'connection_closed'
    }
    if ($stderr -match 'permission denied \(publickey') {
        return 'publickey_rejected'
    }
    return 'other_ssh_failure'
}

function Wait-ForAuthenticatedSsh {
    param(
        [Parameter(Mandatory = $true)][string]$IpAddress,
        [Parameter(Mandatory = $true)][string]$InstanceId,
        [Parameter(Mandatory = $true)][object]$SshPreflight,
        [Parameter(Mandatory = $true)][string]$KnownHostsPath,
        [AllowNull()][scriptblock]$TcpProbe,
        [AllowNull()][scriptblock]$SshAttempt,
        [AllowNull()][scriptblock]$Now,
        [AllowNull()][scriptblock]$Sleep,
        [AllowNull()][scriptblock]$Logger
    )

    $clock = $null
    if ($null -eq $Now) {
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $Now = { $clock.Elapsed.TotalSeconds }.GetNewClosure()
    }
    if ($null -eq $Sleep) {
        $Sleep = { param([double]$Seconds)
            Start-Sleep -Milliseconds ([int][Math]::Ceiling($Seconds * 1000))
        }
    }
    if ($null -eq $TcpProbe) {
        $TcpProbe = { param([string]$HostName)
            Test-TcpPort -HostName $HostName -TimeoutMilliseconds 2000
        }
    }
    if ($null -eq $SshAttempt) {
        $sshPath = [string]$SshPreflight.SshPath
        $sshAttemptTimeoutSeconds = [int]$script:SshAttemptTimeoutSeconds
        $SshAttempt = { param([string[]]$Arguments)
            Invoke-CheckedProcess -FilePath $sshPath `
                -ArgumentList $Arguments `
                -TimeoutSeconds $sshAttemptTimeoutSeconds `
                -AllowedExitCodes (0..255) -AllowTimeout
        }.GetNewClosure()
    }
    if ($null -eq $Logger) {
        $Logger = { param([string]$Level, [string]$Event, [hashtable]$Fields)
            Write-Log -Level $Level -Event $Event -Fields $Fields
        }
    }
    $arguments = @(Get-SshOptions -PrivateKeyPath $SshPreflight.PrivateKeyPath `
        -KnownHostsPath $KnownHostsPath) + @("ubuntu@$IpAddress", 'true')
    $started = [double](& $Now)
    $deadline = $started + $script:SshTimeoutSeconds
    $nextProgress = $started + $script:SshProgressSeconds
    $tcpProbeCount = 0
    $tcpReachableCount = 0
    $sshAttemptCount = 0
    $lastExitCode = $null
    $lastAttemptTimedOut = $false
    $lastFailureClassification = $null
    $lastLoggedClassification = $null

    & $Logger 'INFO' 'ssh_readiness_wait_started' @{
        instance_id = $InstanceId
        ip = $IpAddress
        timeout_seconds = $script:SshTimeoutSeconds
        poll_seconds = $script:SshPollSeconds
    }
    Write-Host "SSH wait started: instance $InstanceId, 45-minute deadline."

    while ([double](& $Now) -lt $deadline) {
        $tcpProbeCount++
        $tcpReachable = [bool](& $TcpProbe $IpAddress)
        $result = $null
        if ($tcpReachable) {
            $tcpReachableCount++
            $sshAttemptCount++
            $result = & $SshAttempt $arguments
            $lastExitCode = [int]$result.ExitCode
            $lastAttemptTimedOut = [bool]$result.TimedOut
            if (-not $lastAttemptTimedOut -and $lastExitCode -eq 0) {
                $elapsed = [Math]::Max(0.0, [double](& $Now) - $started)
                & $Logger 'INFO' 'ssh_authenticated' @{
                    instance_id = $InstanceId
                    ip = $IpAddress
                    elapsed_seconds = [Math]::Round($elapsed, 1)
                    tcp_probe_count = $tcpProbeCount
                    tcp_reachable_count = $tcpReachableCount
                    ssh_attempt_count = $sshAttemptCount
                }
                Write-Host "SSH authenticated: instance $InstanceId after $([Math]::Round($elapsed, 1)) seconds."
                Send-WatcherNotification -Title 'Lambda GH200 SSH authenticated' `
                    -Message "Authenticated SSH is ready for instance $InstanceId." -Sound
                return
            }
        }

        $lastFailureClassification = Get-SshFailureClassification `
            -TcpReachable $tcpReachable -ProcessResult $result
        $nowSeconds = [double](& $Now)
        if (
            $null -eq $lastLoggedClassification -or
            $lastFailureClassification -ne $lastLoggedClassification -or
            $nowSeconds -ge $nextProgress
        ) {
            & $Logger 'INFO' 'ssh_readiness_wait_progress' @{
                instance_id = $InstanceId
                elapsed_seconds = [Math]::Round(
                    [Math]::Max(0.0, $nowSeconds - $started),
                    1
                )
                tcp_probe_count = $tcpProbeCount
                tcp_reachable_count = $tcpReachableCount
                ssh_attempt_count = $sshAttemptCount
                last_ssh_exit_code = $lastExitCode
                last_attempt_timed_out = $lastAttemptTimedOut
                last_failure_classification = $lastFailureClassification
            }
            Write-Host (
                "SSH waiting: elapsed=$([Math]::Round([Math]::Max(0.0, $nowSeconds - $started), 1))s " +
                "status=$lastFailureClassification attempts=$sshAttemptCount"
            )
            $lastLoggedClassification = $lastFailureClassification
            if ($nowSeconds -ge $nextProgress) {
                $nextProgress = $nowSeconds + $script:SshProgressSeconds
            }
        }

        $remaining = $deadline - [double](& $Now)
        if ($remaining -gt 0) {
            & $Sleep ([Math]::Min($script:SshPollSeconds, $remaining))
        }
    }

    $elapsed = [Math]::Max(0.0, [double](& $Now) - $started)
    if ($null -eq $lastFailureClassification) {
        $lastFailureClassification = 'tcp_unavailable'
    }
    & $Logger 'ERROR' 'ssh_readiness_wait_timeout' @{
        instance_id = $InstanceId
        elapsed_seconds = [Math]::Round($elapsed, 1)
        tcp_probe_count = $tcpProbeCount
        tcp_reachable_count = $tcpReachableCount
        ssh_attempt_count = $sshAttemptCount
        last_ssh_exit_code = $lastExitCode
        last_attempt_timed_out = $lastAttemptTimedOut
        last_failure_classification = $lastFailureClassification
    }
    $exitCodeText = if ($null -eq $lastExitCode) { 'none' } else { [string]$lastExitCode }
    throw (
        (
            'Authenticated SSH did not succeed within 45 minutes for instance {0}. ' +
            'elapsed_seconds={1}; tcp_probes={2}; tcp_reachable={3}; ssh_attempts={4}; ' +
            'last_ssh_exit_code={5}; last_attempt_timed_out={6}; last_failure={7}.'
        ) -f
        $InstanceId,
        [Math]::Round($elapsed, 1),
        $tcpProbeCount,
        $tcpReachableCount,
        $sshAttemptCount,
        $exitCodeText,
        $lastAttemptTimedOut,
        $lastFailureClassification
    )
}

function Get-ManualSshCommand {
    param(
        [Parameter(Mandatory = $true)][object]$SshPreflight,
        [Parameter(Mandatory = $true)][string]$KnownHostsPath,
        [Parameter(Mandatory = $true)][string]$IpAddress
    )

    $arguments = @(Get-SshOptions -PrivateKeyPath $SshPreflight.PrivateKeyPath `
        -KnownHostsPath $KnownHostsPath) + @("ubuntu@$IpAddress")
    return (ConvertTo-ProcessArgument $SshPreflight.SshPath) + ' ' + `
        ((@($arguments | ForEach-Object { ConvertTo-ProcessArgument $_ })) -join ' ')
}

function Get-BootstrapMarkerPath {
    param([Parameter(Mandatory = $true)][string]$InstanceId)

    if ($InstanceId -notmatch '^[A-Za-z0-9._-]+$') {
        throw 'Instance ID contains unsafe characters.'
    }
    return '{0}/quant-data/b3/processed/model_runs/_ops/bootstrap_gh200_{1}_success.json' -f `
        $script:FileSystemMount, $InstanceId
}

function Test-SafeRemotePath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return $Path -match '^/[A-Za-z0-9_./-]+$' -and $Path -notmatch '(^|/)\.\.(/|$)'
}

function ConvertTo-PosixShellSingleQuotedArgument {
    param([Parameter(Mandatory = $true)][string]$Value)

    $singleQuote = [string][char]39
    $doubleQuote = [string][char]34
    $escapedQuote = $singleQuote + $doubleQuote + $singleQuote + $doubleQuote + $singleQuote
    return $singleQuote + $Value.Replace($singleQuote, $escapedQuote) + $singleQuote
}

function New-RemoteFileExistsCommand {
    param([Parameter(Mandatory = $true)][string]$Path)

    return '[ -f {0} ]' -f (ConvertTo-PosixShellSingleQuotedArgument $Path)
}

function Resolve-RemoteFileExistsProbe {
    param([Parameter(Mandatory = $true)][object]$Probe)

    if ($Probe.TimedOut) {
        throw 'Remote bootstrap marker probe timed out.'
    }
    if ($Probe.ExitCode -eq 0) {
        return $true
    }
    if ($Probe.ExitCode -eq 1) {
        return $false
    }
    throw 'Remote bootstrap marker probe failed.'
}

function Invoke-RemoteCommand {
    param(
        [Parameter(Mandatory = $true)][object]$SshPreflight,
        [Parameter(Mandatory = $true)][string]$KnownHostsPath,
        [Parameter(Mandatory = $true)][string]$IpAddress,
        [Parameter(Mandatory = $true)][string]$Command,
        [int]$TimeoutSeconds = 30
    )

    $arguments = @(Get-SshOptions -PrivateKeyPath $SshPreflight.PrivateKeyPath `
        -KnownHostsPath $KnownHostsPath) + @("ubuntu@$IpAddress", $Command)
    return Invoke-CheckedProcess -FilePath $SshPreflight.SshPath `
        -ArgumentList $arguments -TimeoutSeconds $TimeoutSeconds `
        -AllowedExitCodes (0..255) -AllowTimeout
}

function Assert-BootstrapMarker {
    param(
        [Parameter(Mandatory = $true)][object]$Marker,
        [Parameter(Mandatory = $true)][object]$SanityReport,
        [Parameter(Mandatory = $true)][string]$InstanceId,
        [Parameter(Mandatory = $true)][object]$State
    )

    $expectedNames = @(
        'passed', 'instance_id', 'git_sha', 'bundle_sha256',
        'bootstrap_sha256', 'completed_at_utc', 'sanity_report_path'
    ) | Sort-Object
    $actualNames = @($Marker.PSObject.Properties.Name | Sort-Object)
    if (($actualNames -join "`n") -cne ($expectedNames -join "`n")) {
        throw 'Remote bootstrap marker schema is invalid.'
    }
    if ((Get-PropertyValue $Marker 'passed') -isnot [bool] -or
        -not [bool](Get-PropertyValue $Marker 'passed')) {
        throw 'Remote bootstrap marker did not record success.'
    }
    if (-not [string]::Equals(
        [string](Get-PropertyValue $Marker 'instance_id'),
        $InstanceId,
        [StringComparison]::Ordinal
    )) {
        throw 'Remote bootstrap marker instance ID is wrong.'
    }
    $gitSha = [string](Get-PropertyValue $Marker 'git_sha')
    $bundleSha = [string](Get-PropertyValue $Marker 'bundle_sha256')
    $bootstrapSha = [string](Get-PropertyValue $Marker 'bootstrap_sha256')
    if ($gitSha -notmatch '^[0-9a-fA-F]{40,64}$' -or
        $bundleSha -notmatch '^[0-9a-fA-F]{64}$' -or
        $bootstrapSha -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'Remote bootstrap marker artifact identity is invalid.'
    }
    $completed = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse(
        [string](Get-PropertyValue $Marker 'completed_at_utc'),
        [ref]$completed
    )) {
        throw 'Remote bootstrap marker completion time is invalid.'
    }
    $reportPath = [string](Get-PropertyValue $Marker 'sanity_report_path')
    $reportPrefix = "$script:FileSystemMount/quant-data/b3/processed/model_runs/"
    if (-not (Test-SafeRemotePath $reportPath) -or
        -not $reportPath.StartsWith($reportPrefix, [StringComparison]::Ordinal)) {
        throw 'Remote bootstrap marker sanity report path is invalid.'
    }
    if ((Get-PropertyValue $SanityReport 'passed') -isnot [bool] -or
        -not [bool](Get-PropertyValue $SanityReport 'passed')) {
        throw 'Remote GH200 sanity report did not pass.'
    }

    $stored = @(
        [string](Get-PropertyValue $State 'launch_git_sha'),
        [string](Get-PropertyValue $State 'launch_bundle_sha256'),
        [string](Get-PropertyValue $State 'launch_bootstrap_sha256')
    )
    $storedPresent = @($stored | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($storedPresent.Count -ne 0 -and $storedPresent.Count -ne 3) {
        throw 'Stored launch artifact identity is incomplete.'
    }
    if ($storedPresent.Count -eq 3) {
        foreach ($pair in @(
            @($stored[0], $gitSha),
            @($stored[1], $bundleSha),
            @($stored[2], $bootstrapSha)
        )) {
            if (-not [string]::Equals(
                [string]$pair[0],
                [string]$pair[1],
                [StringComparison]::Ordinal
            )) {
                throw 'Remote bootstrap marker does not match the stored launch artifacts.'
            }
        }
    }
    return [pscustomobject]@{
        GitSha = $gitSha.ToLowerInvariant()
        BundleSha256 = $bundleSha.ToLowerInvariant()
        BootstrapSha256 = $bootstrapSha.ToLowerInvariant()
        CompletedAtUtc = [string](Get-PropertyValue $Marker 'completed_at_utc')
        SanityReportPath = $reportPath
    }
}

function Get-RemoteBootstrapEvidence {
    param(
        [Parameter(Mandatory = $true)][object]$SshPreflight,
        [Parameter(Mandatory = $true)][string]$KnownHostsPath,
        [Parameter(Mandatory = $true)][string]$IpAddress,
        [Parameter(Mandatory = $true)][string]$InstanceId
    )

    $markerPath = Get-BootstrapMarkerPath $InstanceId
    $probe = Invoke-RemoteCommand -SshPreflight $SshPreflight `
        -KnownHostsPath $KnownHostsPath -IpAddress $IpAddress `
        -Command (New-RemoteFileExistsCommand $markerPath)
    if (-not (Resolve-RemoteFileExistsProbe $probe)) {
        return $null
    }
    $markerResult = Invoke-RemoteCommand -SshPreflight $SshPreflight `
        -KnownHostsPath $KnownHostsPath -IpAddress $IpAddress `
        -Command "cat -- $markerPath"
    if ($markerResult.TimedOut -or $markerResult.ExitCode -ne 0) {
        throw 'Remote bootstrap marker could not be read.'
    }
    try {
        $marker = $markerResult.StdOut | ConvertFrom-Json
    }
    catch {
        throw 'Remote bootstrap marker is not valid JSON.'
    }
    $reportPath = [string](Get-PropertyValue $marker 'sanity_report_path')
    if (-not (Test-SafeRemotePath $reportPath)) {
        throw 'Remote bootstrap marker sanity report path is unsafe.'
    }
    $reportResult = Invoke-RemoteCommand -SshPreflight $SshPreflight `
        -KnownHostsPath $KnownHostsPath -IpAddress $IpAddress `
        -Command "cat -- $reportPath"
    if ($reportResult.TimedOut -or $reportResult.ExitCode -ne 0) {
        throw 'Remote GH200 sanity report is missing.'
    }
    try {
        $sanityReport = $reportResult.StdOut | ConvertFrom-Json
    }
    catch {
        throw 'Remote GH200 sanity report is not valid JSON.'
    }
    return [pscustomobject]@{
        Marker = $marker
        SanityReport = $sanityReport
    }
}

function Complete-BootstrapFromMarker {
    param(
        [Parameter(Mandatory = $true)][object]$Evidence,
        [Parameter(Mandatory = $true)][string]$InstanceId,
        [Parameter(Mandatory = $true)][string]$IpAddress,
        [Parameter(Mandatory = $true)][object]$SshPreflight,
        [Parameter(Mandatory = $true)][string]$KnownHostsPath
    )

    $identity = Assert-BootstrapMarker -Marker $Evidence.Marker `
        -SanityReport $Evidence.SanityReport -InstanceId $InstanceId -State $script:State
    Set-StateValue 'launch_git_sha' $identity.GitSha
    Set-StateValue 'launch_bundle_sha256' $identity.BundleSha256
    Set-StateValue 'launch_bootstrap_sha256' $identity.BootstrapSha256
    Set-StateValue 'bootstrap_status' 'succeeded'
    Set-StateValue 'bootstrap_completed_at_utc' $identity.CompletedAtUtc
    Set-StateValue 'sanity_report_path' $identity.SanityReportPath
    Save-WatcherState
    Write-Log -Level INFO -Event 'bootstrap_marker_validated' -Fields @{
        instance_id = $InstanceId
        ip = $IpAddress
        git_sha = $identity.GitSha
        sanity_report_path = $identity.SanityReportPath
    }
    Send-WatcherNotification -Title 'Lambda GH200 bootstrap succeeded' `
        -Message 'Repository tests and the real gh200 sanity check passed.' -Sound
    Write-Host 'Bootstrap result: succeeded; success marker validated.'
    Write-Host "Sanity report: $($identity.SanityReportPath)"
    Write-Host ('Manual SSH: ' + (Get-ManualSshCommand $SshPreflight $KnownHostsPath $IpAddress))
}

function Invoke-ScpWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [AllowNull()][scriptblock]$ProcessInvoker,
        [AllowNull()][scriptblock]$Sleep
    )

    if ($null -eq $ProcessInvoker) {
        $ProcessInvoker = { param([string]$Executable, [string[]]$Arguments)
            Invoke-CheckedProcess -FilePath $Executable -ArgumentList $Arguments `
                -TimeoutSeconds $script:ScpTimeoutSeconds `
                -AllowedExitCodes (0..255) -AllowTimeout
        }
    }
    if ($null -eq $Sleep) {
        $Sleep = { param([double]$Seconds)
            Start-Sleep -Milliseconds ([int][Math]::Ceiling($Seconds * 1000))
        }
    }
    $finalResult = $null
    for ($attempt = 1; $attempt -le $script:ScpAttempts; $attempt++) {
        $finalResult = & $ProcessInvoker $FilePath $ArgumentList
        if (-not $finalResult.TimedOut -and $finalResult.ExitCode -eq 0) {
            return [pscustomobject]@{
                Succeeded = $true
                Attempts = $attempt
                ExitCode = 0
                TimedOut = $false
            }
        }
        if ($attempt -lt $script:ScpAttempts) {
            & $Sleep $script:ScpRetrySeconds
        }
    }
    return [pscustomobject]@{
        Succeeded = $false
        Attempts = $script:ScpAttempts
        ExitCode = if ($finalResult.TimedOut) { -1 } else { [int]$finalResult.ExitCode }
        TimedOut = [bool]$finalResult.TimedOut
    }
}

function Stop-RemoteBootstrapFailure {
    param(
        [Parameter(Mandatory = $true)][string]$InstanceId,
        [Parameter(Mandatory = $true)][string]$IpAddress,
        [Parameter(Mandatory = $true)][int]$ExitCode,
        [Parameter(Mandatory = $true)][string]$PersistentLog,
        [Parameter(Mandatory = $true)][string]$ManualSsh,
        [Parameter(Mandatory = $true)][string]$Detail
    )

    if ($null -eq $script:State.bootstrap_started_at_utc) {
        Set-StateValue 'bootstrap_started_at_utc' (Get-UtcTimestamp)
    }
    Set-StateValue 'bootstrap_status' 'failed'
    Set-StateValue 'bootstrap_completed_at_utc' (Get-UtcTimestamp)
    Save-WatcherState
    Write-Log -Level ERROR -Event 'bootstrap_failed' -Fields @{
        instance_id = $InstanceId
        ip = $IpAddress
        bootstrap_exit_code = $ExitCode
    }
    Send-WatcherNotification -Title 'Lambda GH200 bootstrap FAILED' `
        -Message "Instance $InstanceId remains running for diagnosis." -Kind Error -Sound
    Write-Host "Bootstrap result: failed. $Detail. Instance: $InstanceId IP: $IpAddress" `
        -ForegroundColor Red
    Write-Host "Persistent bootstrap log: $PersistentLog" -ForegroundColor Red
    Write-Host "Manual SSH: $ManualSsh" -ForegroundColor Red
    Write-Host 'The paid instance was deliberately left running.' -ForegroundColor Red
    $exception = New-Object InvalidOperationException(
        'Remote GH200 bootstrap failed; the instance remains running.'
    )
    $exception.Data['AlreadyNotified'] = $true
    throw $exception
}

function Stop-UntrustedRecovery {
    param(
        [Parameter(Mandatory = $true)][string]$InstanceId,
        [Parameter(Mandatory = $true)][string]$IpAddress,
        [Parameter(Mandatory = $true)][string]$ManualSsh
    )

    Set-StateValue 'bootstrap_status' 'failed'
    Set-StateValue 'bootstrap_completed_at_utc' (Get-UtcTimestamp)
    Save-WatcherState
    Write-Log -Level ERROR -Event 'bootstrap_recovery_identity_missing' -Fields @{
        instance_id = $InstanceId
        ip = $IpAddress
    }
    Send-WatcherNotification -Title 'Lambda GH200 recovery stopped safely' `
        -Message "Instance $InstanceId remains running; launch identity is unavailable." `
        -Kind Error -Sound
    Write-Host "Recovery stopped safely. Instance: $InstanceId IP: $IpAddress" -ForegroundColor Red
    Write-Host 'No valid success marker or trustworthy frozen launch identity was available.' `
        -ForegroundColor Red
    Write-Host "Manual SSH: $ManualSsh" -ForegroundColor Red
    Write-Host 'The paid instance was deliberately left running.' -ForegroundColor Red
    $exception = New-Object InvalidOperationException(
        'Existing GH200 recovery lacks a trustworthy launch identity; the instance remains running.'
    )
    $exception.Data['AlreadyNotified'] = $true
    throw $exception
}

function Invoke-RemoteBootstrap {
    param(
        [Parameter(Mandatory = $true)][object]$Preflight,
        [Parameter(Mandatory = $true)][object]$Instance,
        [Parameter(Mandatory = $true)][string]$KnownHostsPath
    )

    $instanceId = [string](Get-PropertyValue $Instance 'id')
    $ip = [string](Get-PropertyValue $Instance 'ip')
    $artifacts = $Preflight.Artifacts
    Assert-FrozenLaunchArtifacts $artifacts
    $manualSsh = Get-ManualSshCommand $Preflight.Ssh $KnownHostsPath $ip
    $expectedLog = '{0}/quant-data/b3/processed/model_runs/_ops/bootstrap_gh200_{1}_<UTC-TIMESTAMP>.log' -f `
        $script:FileSystemMount, $instanceId
    $scpArguments = New-ScpArguments -PrivateKeyPath $Preflight.Ssh.PrivateKeyPath `
        -KnownHostsPath $KnownHostsPath `
        -LocalPaths @($artifacts.BundlePath, $artifacts.BootstrapPath) -IpAddress $ip
    $scpResult = Invoke-ScpWithRetry -FilePath $Preflight.Ssh.ScpPath `
        -ArgumentList $scpArguments
    if (-not $scpResult.Succeeded) {
        Stop-RemoteBootstrapFailure $instanceId $ip $scpResult.ExitCode `
            $expectedLog $manualSsh 'SCP upload failed after three attempts'
    }

    Set-StateValue 'bootstrap_status' 'running'
    Set-StateValue 'bootstrap_started_at_utc' (Get-UtcTimestamp)
    Set-StateValue 'bootstrap_completed_at_utc' $null
    Set-StateValue 'sanity_report_path' $null
    Save-WatcherState
    Write-Host "Bootstrap started: instance $instanceId."
    $sshArguments = New-BootstrapSshArguments `
        -PrivateKeyPath $Preflight.Ssh.PrivateKeyPath `
        -KnownHostsPath $KnownHostsPath `
        -IpAddress $ip `
        -GitSha $artifacts.GitSha `
        -BundleSha256 $artifacts.BundleSha256 `
        -BootstrapSha256 $artifacts.BootstrapSha256 `
        -InstanceId $instanceId
    $result = Invoke-CheckedProcess -FilePath $Preflight.Ssh.SshPath `
        -ArgumentList $sshArguments -TimeoutSeconds $script:BootstrapTimeoutSeconds `
        -AllowedExitCodes (0..255) -AllowTimeout
    if ($result.TimedOut -or $result.ExitCode -ne 0) {
        $exitCode = if ($result.TimedOut) { -1 } else { [int]$result.ExitCode }
        Stop-RemoteBootstrapFailure $instanceId $ip $exitCode `
            $expectedLog $manualSsh 'Bootstrap failed'
    }
    try {
        $evidence = Get-RemoteBootstrapEvidence -SshPreflight $Preflight.Ssh `
            -KnownHostsPath $KnownHostsPath -IpAddress $ip -InstanceId $instanceId
        if ($null -eq $evidence) {
            throw 'Remote bootstrap exited successfully without a success marker.'
        }
        Complete-BootstrapFromMarker -Evidence $evidence -InstanceId $instanceId `
            -IpAddress $ip -SshPreflight $Preflight.Ssh -KnownHostsPath $KnownHostsPath
    }
    catch {
        Stop-RemoteBootstrapFailure $instanceId $ip 0 `
            $expectedLog $manualSsh 'Bootstrap success marker validation failed'
    }
}

function Get-BootstrapRecoveryAction {
    param(
        [AllowNull()][object]$MarkerEvidence,
        [AllowNull()][object]$Artifacts
    )

    if ($null -ne $MarkerEvidence) {
        return 'marker'
    }
    if ($null -ne $Artifacts) {
        return 'bootstrap'
    }
    return 'refuse'
}

function Start-LaunchWatcher {
    $preflight = Invoke-LaunchPreflight
    Restore-LaunchLimiterFromState
    $instance = $preflight.AdoptedInstance
    if ($null -ne $instance) {
        $instanceId = [string](Get-PropertyValue $instance 'id')
        Write-Log -Level INFO -Event 'existing_instance_adopted' -Fields @{
            instance_id = $instanceId
            status = Get-PropertyValue $instance 'status'
        }
    }
    else {
        [void](Update-AvailabilityObservation $preflight.Discovery)
        $lastPollStart = $preflight.DiscoveryRequestStartedSeconds
        while ($null -eq $instance) {
            $launchAllowed = Test-LaunchAttemptAllowed `
                -AttemptStarts $script:LaunchAttemptStarts `
                -NowSeconds $script:ApiClock.Elapsed.TotalSeconds
            if ($preflight.Discovery.Available -and $launchAllowed) {
                $instance = Invoke-Gh200LaunchAttempt $preflight
                if ($null -ne $instance) {
                    break
                }
            }
            Wait-ForMonotonicTime ($lastPollStart + $script:AvailabilityPollSeconds)
            $snapshot = Get-AvailabilitySnapshot
            $lastPollStart = $snapshot.RequestStartedSeconds
            if ($snapshot.Discovery.Name -ne $preflight.Discovery.Name) {
                throw 'The discovered GH200 instance type changed during polling.'
            }
            $preflight.Discovery = $snapshot.Discovery
            [void](Update-AvailabilityObservation $snapshot.Discovery)
        }
    }

    $instanceId = [string](Get-PropertyValue $instance 'id')
    $active = Wait-ForActiveInstance -InstanceId $instanceId `
        -InstanceTypeName $preflight.Discovery.Name `
        -FileSystemId ([string](Get-PropertyValue $preflight.FileSystem 'id'))
    $ip = [string](Get-PropertyValue $active 'ip')
    $knownHostsPath = Join-Path $script:KnownHostsDirectory $instanceId
    if (-not [IO.File]::Exists($knownHostsPath)) {
        [IO.File]::WriteAllText($knownHostsPath, '', (New-Object Text.UTF8Encoding($false)))
    }
    Wait-ForAuthenticatedSsh -IpAddress $ip -InstanceId $instanceId `
        -SshPreflight $preflight.Ssh -KnownHostsPath $knownHostsPath

    $manualSsh = Get-ManualSshCommand $preflight.Ssh $knownHostsPath $ip
    try {
        $evidence = Get-RemoteBootstrapEvidence -SshPreflight $preflight.Ssh `
            -KnownHostsPath $knownHostsPath -IpAddress $ip -InstanceId $instanceId
    }
    catch {
        Stop-UntrustedRecovery -InstanceId $instanceId -IpAddress $ip -ManualSsh $manualSsh
    }

    $artifacts = $preflight.Artifacts
    if ($null -eq $artifacts) {
        try {
            $artifacts = Get-StateLaunchArtifacts
        }
        catch {
            Stop-UntrustedRecovery -InstanceId $instanceId -IpAddress $ip -ManualSsh $manualSsh
        }
    }
    $action = Get-BootstrapRecoveryAction -MarkerEvidence $evidence -Artifacts $artifacts
    if ($action -eq 'marker') {
        Complete-BootstrapFromMarker -Evidence $evidence -InstanceId $instanceId `
            -IpAddress $ip -SshPreflight $preflight.Ssh -KnownHostsPath $knownHostsPath
        return
    }
    if ($action -eq 'refuse') {
        Stop-UntrustedRecovery -InstanceId $instanceId -IpAddress $ip -ManualSsh $manualSsh
    }
    try {
        Assert-FrozenLaunchArtifacts $artifacts
    }
    catch {
        Stop-UntrustedRecovery -InstanceId $instanceId -IpAddress $ip -ManualSsh $manualSsh
    }
    $preflight.Artifacts = $artifacts
    Invoke-RemoteBootstrap -Preflight $preflight -Instance $active `
        -KnownHostsPath $knownHostsPath
}

function Invoke-WatcherEntry {
    Assert-OperationalParameters -SelectedMode $Mode `
        -BillingAcknowledged ([bool]$IUnderstandBilling) `
        -RunSelfTest ([bool]$SelfTest) `
        -ForgetCredential ([bool]$ForgetStoredApiKey)

    if ($SelfTest) {
        & (Join-Path $script:OpsDirectory 'test-lambda-gh200.ps1')
        if ($LASTEXITCODE -ne 0) {
            throw "Lambda GH200 watcher self-test failed with exit code $LASTEXITCODE."
        }
        return
    }

    if ($ForgetStoredApiKey) {
        $localAppData = [Environment]::GetFolderPath(
            [Environment+SpecialFolder]::LocalApplicationData
        )
        if ([string]::IsNullOrWhiteSpace($localAppData)) {
            throw 'LOCALAPPDATA could not be resolved for the current Windows user.'
        }
        $credentialPath = Join-Path $localAppData `
            'BrazilRV\lambda-gh200-watcher\credential\lambda-api-key.dpapi'
        if ([IO.File]::Exists($credentialPath)) {
            [IO.File]::Delete($credentialPath)
            Write-Host "Deleted stored DPAPI credential: $credentialPath"
        }
        else {
            Write-Host 'No stored DPAPI Lambda API key exists.'
        }
        return
    }

    Initialize-RuntimePaths
    if ($PSVersionTable.PSVersion.Major -le 5) {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    }
    $script:ApiClock = [Diagnostics.Stopwatch]::StartNew()
    try {
        Acquire-WatcherLock
        Initialize-WatcherState $Mode
        Enable-SleepPrevention
        Initialize-Notifier
        Write-Log -Level INFO -Event 'watcher_started' -Fields @{ mode = $Mode }
        Send-WatcherNotification -Title 'Lambda GH200 watcher started' `
            -Message "$Mode mode is running. Keep this PC awake and online." -Sound
        Initialize-ApiCredential
        if ($Mode -eq 'Notify') {
            Start-NotifyWatcher
        }
        else {
            Start-LaunchWatcher
        }
    }
    catch {
        $code = [string]$_.Exception.Data['LambdaCode']
        Write-Log -Level ERROR -Event 'fatal_error' -Fields @{
            lambda_error_code = $code
            error_type = $_.Exception.GetType().Name
        }
        if ($code -eq 'global/invalid-api-key') {
            Send-WatcherNotification -Title 'Lambda API authentication failed' `
                -Message 'The Lambda API key was rejected. The watcher stopped.' -Kind Error -Sound
        }
        elseif (-not [bool]$_.Exception.Data['AlreadyNotified']) {
            Send-WatcherNotification -Title 'Lambda GH200 watcher stopped' `
                -Message $_.Exception.Message -Kind Error -Sound
        }
        throw
    }
    finally {
        try { Save-WatcherState } catch { }
        Clear-ApiCredential
        Disable-SleepPrevention
        if ($null -ne $script:NotifyIcon) {
            $script:NotifyIcon.Visible = $false
            $script:NotifyIcon.Dispose()
            $script:NotifyIcon = $null
        }
        if ($null -ne $script:LockStream) {
            $script:LockStream.Dispose()
            $script:LockStream = $null
        }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    Invoke-WatcherEntry
}





























