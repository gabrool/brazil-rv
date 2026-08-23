<#
.SYNOPSIS
Polls Lambda Cloud for a single-GPU GH200 or launches one reviewed Brazil-RV host.

.DESCRIPTION
Notify prints availability changes. Launch requires -IUnderstandBilling, adopts at
most one matching instance or launches one, waits for SSH, and transfers one verified
Git bundle plus the committed bootstrap script. It never starts training and never
terminates an instance.
#>
[CmdletBinding()]
param(
    [ValidateSet('Notify', 'Launch')]
    [string]$Mode,
    [switch]$IUnderstandBilling,
    [switch]$Notify,
    [switch]$SelfTest,
    [switch]$ForgetStoredApiKey
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Security

$script:ApiBaseUri = 'https://cloud.lambda.ai/api/v1'
$script:TargetRegion = 'us-east-3'
$script:FileSystemName = 'brazil-rv-east3'
$script:FileSystemMount = '/lambda/nfs/brazil-rv-east3'
$script:SshKeyName = 'brazil-rv'
$script:InstanceName = 'brazil-rv-gh200'
$script:RepositoryRoot = 'C:\Brazil-RV\quant\b3-quant'
$script:PollSeconds = 2.0
$script:ApiMinimumSeconds = 1.10
$script:ApiHeaders = $null
$script:LastApiRequestUtc = [DateTime]::MinValue
$script:Secrets = New-Object 'System.Collections.Generic.List[string]'
$script:RuntimeRoot = $null
$script:CredentialPath = $null
$script:StatePath = $null
$script:LogPath = $null
$script:KnownHostsDirectory = $null
$script:ArtifactsDirectory = $null
$script:LockStream = $null
$script:ActiveInstance = $null

function Get-Value {
    param([AllowNull()][object]$Object, [Parameter(Mandatory = $true)][string]$Name)
    if ($null -eq $Object) { return $null }
    if ($Object -is [Collections.IDictionary]) {
        return $(if ($Object.Contains($Name)) { $Object[$Name] } else { $null })
    }
    $property = $Object.PSObject.Properties[$Name]
    return $(if ($null -eq $property) { $null } else { $property.Value })
}

function Assert-Invocation {
    param(
        [AllowNull()][string]$SelectedMode,
        [bool]$BillingAcknowledged,
        [bool]$RunSelfTest,
        [bool]$ForgetCredential
    )
    if ($RunSelfTest -or $ForgetCredential) {
        if (-not [string]::IsNullOrEmpty($SelectedMode) -or $BillingAcknowledged) {
            throw 'Self-test and credential deletion cannot be combined with a mode or billing acknowledgement.'
        }
        return
    }
    if ([string]::IsNullOrEmpty($SelectedMode)) { throw '-Mode is required.' }
    if ($SelectedMode -eq 'Launch' -and -not $BillingAcknowledged) {
        throw 'Launch requires -IUnderstandBilling.'
    }
    if ($SelectedMode -eq 'Notify' -and $BillingAcknowledged) {
        throw '-IUnderstandBilling is valid only with Launch.'
    }
}

function Protect-Text {
    param(
        [AllowNull()][object]$Text,
        [object[]]$SecretValues = @($script:Secrets)
    )
    $safe = [string]$Text
    foreach ($secret in $SecretValues) {
        if (-not [string]::IsNullOrEmpty($secret)) { $safe = $safe.Replace($secret, '<redacted>') }
    }
    return $safe.Replace("`r", ' ').Replace("`n", ' ')
}

function Write-Log {
    param([Parameter(Mandatory = $true)][string]$Message)
    $line = '{0} {1}' -f [DateTime]::UtcNow.ToString('o'), (Protect-Text $Message)
    Write-Host $line
    if (-not [string]::IsNullOrEmpty($script:LogPath)) {
        [IO.File]::AppendAllText($script:LogPath, $line + [Environment]::NewLine)
    }
}

function Initialize-Paths {
    $local = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
    if ([string]::IsNullOrWhiteSpace($local)) { throw 'LOCALAPPDATA is unavailable.' }
    $script:RuntimeRoot = Join-Path $local 'BrazilRV\lambda-gh200'
    $credential = Join-Path $script:RuntimeRoot 'credential'
    $logs = Join-Path $script:RuntimeRoot 'logs'
    $script:KnownHostsDirectory = Join-Path $script:RuntimeRoot 'known-hosts'
    $script:ArtifactsDirectory = Join-Path $script:RuntimeRoot 'artifacts'
    foreach ($directory in @($script:RuntimeRoot, $credential, $logs, $script:KnownHostsDirectory, $script:ArtifactsDirectory)) {
        [IO.Directory]::CreateDirectory($directory) | Out-Null
    }
    $script:CredentialPath = Join-Path $credential 'lambda-api-key.dpapi'
    $script:StatePath = Join-Path $script:RuntimeRoot 'instance.json'
    $script:LogPath = Join-Path $logs ('watcher_{0}.log' -f [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
}

function Acquire-WatcherLock {
    $path = Join-Path $script:RuntimeRoot 'watcher.lock'
    try {
        $script:LockStream = New-Object IO.FileStream(
            $path, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None
        )
    }
    catch { throw 'Another Lambda GH200 watcher is already running.' }
}

function Remove-StoredCredential {
    Initialize-Paths
    if ([IO.File]::Exists($script:CredentialPath)) {
        [IO.File]::Delete($script:CredentialPath)
        Write-Host "Deleted stored DPAPI credential: $script:CredentialPath"
    }
    else { Write-Host 'No stored DPAPI Lambda API key exists.' }
}

function Get-ApiKey {
    if ([IO.File]::Exists($script:CredentialPath)) {
        $protected = [IO.File]::ReadAllBytes($script:CredentialPath)
        $bytes = [System.Security.Cryptography.ProtectedData]::Unprotect(
            $protected, $null, [System.Security.Cryptography.DataProtectionScope]::CurrentUser
        )
        $key = [Text.Encoding]::UTF8.GetString($bytes)
        [Array]::Clear($bytes, 0, $bytes.Length)
    }
    else {
        $secure = Read-Host 'Lambda API key (stored with current-user DPAPI)' -AsSecureString
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        try { $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
        if ([string]::IsNullOrWhiteSpace($key)) { throw 'Lambda API key is empty.' }
        $bytes = [Text.Encoding]::UTF8.GetBytes($key)
        try {
            $protected = [System.Security.Cryptography.ProtectedData]::Protect(
                $bytes, $null, [System.Security.Cryptography.DataProtectionScope]::CurrentUser
            )
            [IO.File]::WriteAllBytes($script:CredentialPath, $protected)
        }
        finally { [Array]::Clear($bytes, 0, $bytes.Length) }
    }
    if ([string]::IsNullOrWhiteSpace($key)) { throw 'Stored Lambda API key is empty.' }
    $script:Secrets.Add($key)
    return $key
}

function Get-RetryAfterSeconds {
    param([AllowNull()][object]$Value)
    if ($null -eq $Value) { return 0.0 }
    $seconds = 0.0
    if ([double]::TryParse([string]$Value, [ref]$seconds)) {
        return [Math]::Max(0.0, $seconds)
    }
    $date = [DateTimeOffset]::MinValue
    if ([DateTimeOffset]::TryParse([string]$Value, [ref]$date)) {
        return [Math]::Max(0.0, ($date - [DateTimeOffset]::UtcNow).TotalSeconds)
    }
    return 0.0
}

function Get-HeaderValue {
    param(
        [AllowNull()][object]$Headers,
        [Parameter(Mandatory = $true)][string]$Name
    )
    if ($null -eq $Headers) { return $null }
    if ($null -ne $Headers.PSObject.Methods['TryGetValues']) {
        $values = $null
        if ($Headers.TryGetValues($Name, [ref]$values)) {
            return @($values)[0]
        }
        return $null
    }
    if ($null -ne $Headers.PSObject.Methods['Get']) {
        return $Headers.Get($Name)
    }
    if ($Headers -is [Collections.IDictionary] -and $Headers.Contains($Name)) {
        return $Headers[$Name]
    }
    return $null
}

function Invoke-LambdaApi {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('GET', 'POST')][string]$Method,
        [Parameter(Mandatory = $true)][string]$Path,
        [AllowNull()][object]$Body,
        [int]$MaximumAttempts = 5,
        [AllowNull()][scriptblock]$RequestInvoker,
        [AllowNull()][scriptblock]$Sleeper
    )
    if ($null -eq $RequestInvoker) {
        $RequestInvoker = { param([hashtable]$Parameters) Invoke-WebRequest @Parameters }
    }
    if ($null -eq $Sleeper) {
        $Sleeper = { param([double]$Seconds) Start-Sleep -Milliseconds ([int][Math]::Ceiling(1000 * $Seconds)) }
    }
    for ($attempt = 1; $attempt -le $MaximumAttempts; $attempt++) {
        $elapsed = ([DateTime]::UtcNow - $script:LastApiRequestUtc).TotalSeconds
        if ($elapsed -lt $script:ApiMinimumSeconds) {
            & $Sleeper ($script:ApiMinimumSeconds - $elapsed)
        }
        $parameters = @{
            Uri = $script:ApiBaseUri + $Path
            Method = $Method
            Headers = $script:ApiHeaders
            UseBasicParsing = $true
            TimeoutSec = 30
            ErrorAction = 'Stop'
        }
        if ($null -ne $Body) {
            $parameters.ContentType = 'application/json'
            $parameters.Body = $Body | ConvertTo-Json -Depth 8 -Compress
        }
        $script:LastApiRequestUtc = [DateTime]::UtcNow
        try {
            $response = & $RequestInvoker $parameters
            $parsed = $response.Content | ConvertFrom-Json
            if ($null -eq $parsed.PSObject.Properties['data']) {
                throw 'Lambda response has no data field.'
            }
            return $parsed.data
        }
        catch {
            $status = 0
            $retryAfter = $null
            if ($null -ne $_.Exception.Data['StatusCode']) {
                $status = [int]$_.Exception.Data['StatusCode']
                $retryAfter = $_.Exception.Data['RetryAfter']
            }
            elseif ($null -ne $_.Exception.Response) {
                $status = [int]$_.Exception.Response.StatusCode
                $retryAfter = Get-HeaderValue $_.Exception.Response.Headers 'Retry-After'
            }
            $retryable = $status -eq 429 -or ($Method -eq 'GET' -and ($status -eq 0 -or $status -ge 500))
            if ($retryable -and $attempt -lt $MaximumAttempts) {
                $delay = [Math]::Max((Get-RetryAfterSeconds $retryAfter), [Math]::Min(16.0, [Math]::Pow(2, $attempt - 1)))
                Write-Log "Lambda API retry $attempt/$MaximumAttempts after $delay seconds (HTTP $status)."
                & $Sleeper $delay
                continue
            }
            $exception = New-Object InvalidOperationException(
                "Lambda API $Method $Path failed (HTTP $status): $(Protect-Text $_.Exception.Message)"
            )
            $exception.Data['HttpStatus'] = $status
            $details = [string]$_.ErrorDetails.Message
            if ($details -match 'global/insufficient-capacity') {
                $exception.Data['LambdaCode'] = 'global/insufficient-capacity'
            }
            throw $exception
        }
    }
}

function Find-Gh200InstanceType {
    param([Parameter(Mandatory = $true)][object]$InstanceTypes)
    $items = @()
    if ($InstanceTypes -is [Collections.IDictionary]) { $items = @($InstanceTypes.Values) }
    else { $items = @($InstanceTypes.PSObject.Properties | ForEach-Object Value) }
    $candidates = @()
    foreach ($item in $items) {
        $type = Get-Value $item 'instance_type'
        $specs = Get-Value $type 'specs'
        $description = "$(Get-Value $type 'gpu_description') $(Get-Value $type 'description')"
        if ((Get-Value $specs 'gpus') -eq 1 -and $description -match '(?i)GH200') {
            $available = @((Get-Value $item 'regions_with_capacity_available') | Where-Object {
                (Get-Value $_ 'name') -eq $script:TargetRegion
            }).Count -gt 0
            $candidates += [pscustomobject]@{
                Name = [string](Get-Value $type 'name')
                PriceCentsPerHour = [int](Get-Value $type 'price_cents_per_hour')
                Available = $available
            }
        }
    }
    if ($candidates.Count -ne 1 -or [string]::IsNullOrWhiteSpace($candidates[0].Name)) {
        throw "Expected exactly one single-GPU GH200 type; found $($candidates.Count)."
    }
    return $candidates[0]
}

function Get-ExactlyOneNamed {
    param([object[]]$Items, [string]$Name, [string]$Kind)
    $matches = @($Items | Where-Object { (Get-Value $_ 'name') -eq $Name })
    if ($matches.Count -ne 1) { throw "Expected one $Kind named '$Name'; found $($matches.Count)." }
    return $matches[0]
}

function Get-TagMap {
    param([AllowNull()][object]$Tags)
    $map = @{}
    foreach ($tag in @($Tags)) { $map[[string](Get-Value $tag 'key')] = [string](Get-Value $tag 'value') }
    return $map
}

function Select-ManagedInstance {
    param([object[]]$Instances, [string]$InstanceTypeName)
    $matches = @($Instances | Where-Object {
        $tags = Get-TagMap (Get-Value $_ 'tags')
        $status = [string](Get-Value $_ 'status')
        (Get-Value $_ 'name') -eq $script:InstanceName -and
        (Get-Value (Get-Value $_ 'region') 'name') -eq $script:TargetRegion -and
        (Get-Value (Get-Value $_ 'instance_type') 'name') -eq $InstanceTypeName -and
        $tags['project'] -eq 'brazil-rv' -and
        $tags['managed-by'] -eq 'gh200-watcher' -and
        @('terminated', 'preempted') -notcontains $status
    })
    if ($matches.Count -gt 1) { throw 'Multiple matching nonterminal Brazil-RV GH200 instances exist.' }
    return $(if ($matches.Count -eq 1) { $matches[0] } else { $null })
}

function New-LaunchPayload {
    param([string]$InstanceTypeName, [object]$FileSystem)
    $fileSystemId = [string](Get-Value $FileSystem 'id')
    if ([string]::IsNullOrWhiteSpace($fileSystemId)) { throw 'Filesystem ID is missing.' }
    return [ordered]@{
        region_name = $script:TargetRegion
        instance_type_name = $InstanceTypeName
        ssh_key_names = @($script:SshKeyName)
        file_system_mounts = @([ordered]@{
            file_system_id = $fileSystemId
            mount_point = $script:FileSystemMount
        })
        hostname = $script:InstanceName
        name = $script:InstanceName
        tags = @(
            [ordered]@{ key = 'project'; value = 'brazil-rv' },
            [ordered]@{ key = 'purpose'; value = 'research-training' },
            [ordered]@{ key = 'managed-by'; value = 'gh200-watcher' }
        )
    }
}

function ConvertTo-ProcessArgument {
    param([AllowEmptyString()][string]$Argument)
    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') { return $Argument }
    $builder = New-Object Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ([int]$character -eq 92) { $backslashes++; continue }
        if ([int]$character -eq 34) {
            [void]$builder.Append(('\' * (2 * $backslashes + 1)))
            [void]$builder.Append('"'); $backslashes = 0; continue
        }
        if ($backslashes -gt 0) { [void]$builder.Append(('\' * $backslashes)); $backslashes = 0 }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) { [void]$builder.Append(('\' * (2 * $backslashes))) }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Invoke-Process {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [AllowNull()][string]$WorkingDirectory,
        [int]$TimeoutSeconds = 300,
        [int[]]$AllowedExitCodes = @(0),
        [switch]$AllowTimeout
    )
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $FilePath
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    if (-not [string]::IsNullOrEmpty($WorkingDirectory)) { $info.WorkingDirectory = $WorkingDirectory }
    if ($null -ne $info.PSObject.Properties['ArgumentList']) {
        foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
    }
    else { $info.Arguments = (@($Arguments | ForEach-Object { ConvertTo-ProcessArgument $_ })) -join ' ' }
    foreach ($name in @('LAMBDA_API_KEY', 'LAMBDA_CLOUD_API_KEY')) {
        if ($info.EnvironmentVariables.ContainsKey($name)) { $info.EnvironmentVariables.Remove($name) }
    }
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $timedOut = -not $process.WaitForExit($TimeoutSeconds * 1000)
        if ($timedOut -and -not $process.HasExited) { $process.Kill() }
        $process.WaitForExit()
        $result = [pscustomobject]@{
            ExitCode = $(if ($timedOut) { -1 } else { $process.ExitCode })
            TimedOut = $timedOut
            StdOut = $stdout.GetAwaiter().GetResult()
            StdErr = $stderr.GetAwaiter().GetResult()
        }
        if ($timedOut -and -not $AllowTimeout) { throw "Process timed out: $FilePath" }
        if (-not $timedOut -and $AllowedExitCodes -notcontains $result.ExitCode) {
            throw "Process failed with exit code $($result.ExitCode): $FilePath"
        }
        return $result
    }
    finally { $process.Dispose() }
}

function Get-SshArguments {
    param([string]$PrivateKeyPath, [string]$KnownHostsPath)
    return @(
        '-i', $PrivateKeyPath,
        '-o', 'IdentitiesOnly=yes',
        '-o', 'BatchMode=yes',
        '-o', 'PasswordAuthentication=no',
        '-o', 'StrictHostKeyChecking=accept-new',
        '-o', "UserKnownHostsFile=$KnownHostsPath",
        '-o', 'ConnectTimeout=10',
        '-o', 'ServerAliveInterval=30',
        '-o', 'ServerAliveCountMax=3'
    )
}

function Get-SshTools {
    param([object]$CloudKey)
    $private = Join-Path (Join-Path $env:USERPROFILE '.ssh') 'lambda_brazil_rv_ed25519'
    $public = $private + '.pub'
    foreach ($path in @($private, $public)) {
        if (-not [IO.File]::Exists($path)) { throw "Required SSH key is missing: $path" }
    }
    $localIdentity = (([IO.File]::ReadAllText($public) -split '\s+')[0..1] -join ' ')
    $cloudText = [string](Get-Value $CloudKey 'public_key')
    if (-not [string]::IsNullOrWhiteSpace($cloudText)) {
        $cloudIdentity = (($cloudText -split '\s+')[0..1] -join ' ')
        if ($localIdentity -ne $cloudIdentity) { throw 'Local and Lambda Brazil-RV SSH keys differ.' }
    }
    return [pscustomobject]@{
        PrivateKey = $private
        Ssh = (Get-Command ssh.exe -ErrorAction Stop).Source
        Scp = (Get-Command scp.exe -ErrorAction Stop).Source
    }
}

function New-LaunchArtifacts {
    $git = (Get-Command git.exe -ErrorAction SilentlyContinue)
    if ($null -eq $git) { $git = Get-Command git -ErrorAction Stop }
    $status = Invoke-Process $git.Source @('status', '--porcelain') $script:RepositoryRoot
    if (-not [string]::IsNullOrWhiteSpace($status.StdOut)) { throw 'Launch requires a clean worktree.' }
    $branch = (Invoke-Process $git.Source @('symbolic-ref', '--short', 'HEAD') $script:RepositoryRoot).StdOut.Trim()
    if ($branch -ne 'main') { throw 'Launch requires the main branch.' }
    $sha = (Invoke-Process $git.Source @('rev-parse', 'HEAD') $script:RepositoryRoot).StdOut.Trim().ToLowerInvariant()
    $directory = Join-Path $script:ArtifactsDirectory $sha
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $bundle = Join-Path $directory "brazil-rv_$sha.bundle"
    $bootstrap = Join-Path $directory 'lambda-gh200-bootstrap.sh'
    if (-not [IO.File]::Exists($bundle)) {
        Invoke-Process $git.Source @('bundle', 'create', $bundle, 'refs/heads/main') $script:RepositoryRoot 900 | Out-Null
    }
    $source = (Invoke-Process $git.Source @('show', "${sha}:ops/lambda-gh200-bootstrap.sh") $script:RepositoryRoot).StdOut
    [IO.File]::WriteAllText($bootstrap, $source.Replace("`r`n", "`n").Replace("`r", "`n"), (New-Object Text.UTF8Encoding($false)))
    $artifacts = [pscustomobject]@{
        GitSha = $sha
        BundlePath = $bundle
        BootstrapPath = $bootstrap
        BundleSha256 = (Get-FileHash $bundle -Algorithm SHA256).Hash.ToLowerInvariant()
        BootstrapSha256 = (Get-FileHash $bootstrap -Algorithm SHA256).Hash.ToLowerInvariant()
        GitPath = $git.Source
    }
    Assert-TransferArtifacts $artifacts
    return $artifacts
}

function Assert-TransferArtifacts {
    param([Parameter(Mandatory = $true)][object]$Artifacts)
    foreach ($path in @($Artifacts.BundlePath, $Artifacts.BootstrapPath)) {
        if (-not [IO.File]::Exists($path)) { throw "Transfer artifact is missing: $path" }
    }
    if ((Get-FileHash $Artifacts.BundlePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Artifacts.BundleSha256) {
        throw 'Git bundle hash mismatch.'
    }
    if ((Get-FileHash $Artifacts.BootstrapPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Artifacts.BootstrapSha256) {
        throw 'Bootstrap hash mismatch.'
    }
    if ($null -ne (Get-Value $Artifacts 'GitPath')) {
        Invoke-Process $Artifacts.GitPath @('bundle', 'verify', $Artifacts.BundlePath) $script:RepositoryRoot 900 | Out-Null
        $heads = (Invoke-Process $Artifacts.GitPath @('bundle', 'list-heads', $Artifacts.BundlePath) $script:RepositoryRoot).StdOut
        if ($heads -notmatch "(?m)^$($Artifacts.GitSha)\s+refs/heads/main\s*$") {
            throw 'Git bundle does not expose the expected main commit.'
        }
    }
}

function Test-ExpectedMount {
    param([object]$Instance, [string]$FileSystemId)
    return @((Get-Value $Instance 'file_system_mounts') | Where-Object {
        (Get-Value $_ 'file_system_id') -eq $FileSystemId -and
        (Get-Value $_ 'mount_point') -eq $script:FileSystemMount
    }).Count -eq 1
}

function Assert-ActiveInstance {
    param([object]$Instance, [string]$InstanceTypeName, [string]$FileSystemId)
    if ([string]::IsNullOrWhiteSpace([string](Get-Value $Instance 'ip'))) { throw 'Active instance has no IP.' }
    if ((Get-Value (Get-Value $Instance 'region') 'name') -ne $script:TargetRegion) { throw 'Instance region mismatch.' }
    if ((Get-Value (Get-Value $Instance 'instance_type') 'name') -ne $InstanceTypeName) { throw 'Instance type mismatch.' }
    if (-not (Test-ExpectedMount $Instance $FileSystemId)) { throw 'Instance filesystem mismatch.' }
    if (@(Get-Value $Instance 'ssh_key_names') -notcontains $script:SshKeyName) { throw 'Instance SSH key mismatch.' }
}

function Wait-ForActiveInstance {
    param([string]$InstanceId, [string]$InstanceTypeName, [string]$FileSystemId)
    $deadline = [DateTime]::UtcNow.AddMinutes(20)
    while ([DateTime]::UtcNow -lt $deadline) {
        $instance = Invoke-LambdaApi GET ("/instances/{0}" -f [Uri]::EscapeDataString($InstanceId)) $null
        $status = [string](Get-Value $instance 'status')
        Write-Log "Instance $InstanceId status: $status"
        if ($status -eq 'active') {
            Assert-ActiveInstance $instance $InstanceTypeName $FileSystemId
            return $instance
        }
        if (@('unhealthy', 'terminated', 'terminating', 'preempted') -contains $status) {
            throw "Instance entered failure status '$status'."
        }
        Start-Sleep -Seconds 5
    }
    throw 'Instance did not become active within 20 minutes.'
}

function Wait-ForSsh {
    param([object]$Tools, [string]$KnownHostsPath, [string]$IpAddress)
    $arguments = @(Get-SshArguments $Tools.PrivateKey $KnownHostsPath) + @("ubuntu@$IpAddress", 'true')
    $deadline = [DateTime]::UtcNow.AddMinutes(45)
    while ([DateTime]::UtcNow -lt $deadline) {
        $result = Invoke-Process $Tools.Ssh $arguments $null 15 (0..255) -AllowTimeout
        if (-not $result.TimedOut -and $result.ExitCode -eq 0) { return }
        Start-Sleep -Seconds 5
    }
    throw 'Authenticated SSH did not become ready within 45 minutes.'
}

function Save-InstanceState {
    param([object]$Instance, [string]$KnownHostsPath, [string]$SshCommand, [string]$RemoteLog)
    $state = [ordered]@{
        instance_id = [string](Get-Value $Instance 'id')
        ip = [string](Get-Value $Instance 'ip')
        status = [string](Get-Value $Instance 'status')
        known_hosts = $KnownHostsPath
        ssh_command = $SshCommand
        bootstrap_log = $RemoteLog
        updated_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    $temporary = $script:StatePath + '.tmp'
    [IO.File]::WriteAllText($temporary, ($state | ConvertTo-Json -Depth 4) + [Environment]::NewLine)
    Move-Item -LiteralPath $temporary -Destination $script:StatePath -Force
}

function Invoke-RemoteBootstrap {
    param([object]$Instance, [object]$Tools, [string]$KnownHostsPath, [object]$Artifacts)
    Assert-TransferArtifacts $Artifacts
    $ip = [string](Get-Value $Instance 'ip')
    $instanceId = [string](Get-Value $Instance 'id')
    $sshOptions = @(Get-SshArguments $Tools.PrivateKey $KnownHostsPath)
    $scpArguments = $sshOptions + @($Artifacts.BundlePath, $Artifacts.BootstrapPath, "ubuntu@${ip}:/home/ubuntu/")
    $uploaded = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $result = Invoke-Process $Tools.Scp $scpArguments $null 300 (0..255) -AllowTimeout
        if (-not $result.TimedOut -and $result.ExitCode -eq 0) { $uploaded = $true; break }
        Start-Sleep -Seconds 5
    }
    if (-not $uploaded) { throw 'SCP upload failed after three attempts.' }
    $arguments = $sshOptions + @(
        "ubuntu@$ip", 'bash', '/home/ubuntu/lambda-gh200-bootstrap.sh',
        $Artifacts.GitSha, $Artifacts.BundleSha256, $Artifacts.BootstrapSha256, $instanceId
    )
    $result = Invoke-Process $Tools.Ssh $arguments $null (90 * 60) (0..255) -AllowTimeout
    if (-not [string]::IsNullOrWhiteSpace($result.StdOut)) { Write-Host $result.StdOut.TrimEnd() }
    if ($result.TimedOut -or $result.ExitCode -ne 0) { throw 'Remote bootstrap failed.' }
    return "$script:FileSystemMount/quant-data/b3/processed/model_runs/_ops/bootstrap_gh200_${instanceId}.log"
}

function Send-OptionalNotification {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
    if ($Notify) { [System.Media.SystemSounds]::Asterisk.Play() }
}

function Start-NotifyWatcher {
    $previous = $null
    while ($true) {
        $type = Find-Gh200InstanceType (Invoke-LambdaApi GET '/instance-types' $null)
        if ($null -eq $previous -or [bool]$previous -ne [bool]$type.Available) {
            $state = $(if ($type.Available) { 'AVAILABLE' } else { 'unavailable' })
            Send-OptionalNotification "GH200 $($type.Name) in $script:TargetRegion is $state (`$$([decimal]$type.PriceCentsPerHour / 100)/hour)."
            $previous = [bool]$type.Available
        }
        Start-Sleep -Milliseconds ([int](1000 * $script:PollSeconds))
    }
}

function Start-LaunchWatcher {
    $artifacts = New-LaunchArtifacts
    $type = Find-Gh200InstanceType (Invoke-LambdaApi GET '/instance-types' $null)
    $fileSystem = Get-ExactlyOneNamed @((Invoke-LambdaApi GET '/file-systems' $null)) $script:FileSystemName 'filesystem'
    if ((Get-Value (Get-Value $fileSystem 'region') 'name') -ne $script:TargetRegion) { throw 'Filesystem is in the wrong region.' }
    $cloudKey = Get-ExactlyOneNamed @((Invoke-LambdaApi GET '/ssh-keys' $null)) $script:SshKeyName 'SSH key'
    $tools = Get-SshTools $cloudKey
    $instances = @((Invoke-LambdaApi GET '/instances' $null))
    $instance = Select-ManagedInstance $instances $type.Name
    $fileSystemId = [string](Get-Value $fileSystem 'id')
    if ($null -ne $instance -and -not (Test-ExpectedMount $instance $fileSystemId)) {
        throw 'Matching instance is attached to the wrong filesystem.'
    }
    if ($null -eq $instance -and [bool](Get-Value $fileSystem 'is_in_use')) {
        throw 'The intended filesystem is already in use by another instance.'
    }
    while ($null -eq $instance) {
        if (-not $type.Available) {
            Write-Log "Waiting for GH200 capacity in $script:TargetRegion."
            Start-Sleep -Milliseconds ([int](1000 * $script:PollSeconds))
            $type = Find-Gh200InstanceType (Invoke-LambdaApi GET '/instance-types' $null)
            continue
        }
        $instance = Select-ManagedInstance @((Invoke-LambdaApi GET '/instances' $null)) $type.Name
        if ($null -ne $instance) { break }
        Assert-TransferArtifacts $artifacts
        try {
            $result = Invoke-LambdaApi POST '/instance-operations/launch' (New-LaunchPayload $type.Name $fileSystem)
            $ids = @(Get-Value $result 'instance_ids')
            if ($ids.Count -ne 1) { throw 'Launch response did not contain exactly one instance ID.' }
            $instance = [pscustomobject]@{ id = [string]$ids[0]; status = 'booting' }
            Write-Log "Launch accepted for instance $($instance.id). Billing may be active."
        }
        catch {
            Start-Sleep -Seconds 3
            $instance = Select-ManagedInstance @((Invoke-LambdaApi GET '/instances' $null)) $type.Name
            if ($null -eq $instance) {
                if ($_.Exception.Data['LambdaCode'] -eq 'global/insufficient-capacity') {
                    Write-Log 'Capacity disappeared before launch; polling continues.'
                    $type.Available = $false
                    continue
                }
                throw
            }
            Write-Log "Adopted matching instance $((Get-Value $instance 'id')) after an ambiguous launch response."
        }
    }
    $instanceId = [string](Get-Value $instance 'id')
    $script:ActiveInstance = $instance
    $script:ActiveInstance = Wait-ForActiveInstance $instanceId $type.Name $fileSystemId
    $ip = [string](Get-Value $script:ActiveInstance 'ip')
    $knownHosts = Join-Path $script:KnownHostsDirectory $instanceId
    if (-not [IO.File]::Exists($knownHosts)) { [IO.File]::WriteAllText($knownHosts, '') }
    Wait-ForSsh $tools $knownHosts $ip
    $sshArguments = @(Get-SshArguments $tools.PrivateKey $knownHosts) + @("ubuntu@$ip")
    $sshCommand = (ConvertTo-ProcessArgument $tools.Ssh) + ' ' + ((@($sshArguments | ForEach-Object { ConvertTo-ProcessArgument $_ })) -join ' ')
    try { $remoteLog = Invoke-RemoteBootstrap $script:ActiveInstance $tools $knownHosts $artifacts }
    catch {
        Write-Warning "Bootstrap failed. Instance $instanceId at $ip remains running."
        Write-Host "SSH: $sshCommand"
        Write-Host "Watcher log: $script:LogPath"
        throw
    }
    Save-InstanceState $script:ActiveInstance $knownHosts $sshCommand $remoteLog
    Write-Host "Instance ID: $instanceId"
    Write-Host "IP: $ip"
    Write-Host "SSH: $sshCommand"
    Write-Host "Bootstrap log: $remoteLog"
    Write-Host "Watcher log: $script:LogPath"
    Write-Host 'The paid instance remains running. Training was not started.' -ForegroundColor Yellow
}

function Invoke-Main {
    Assert-Invocation $Mode ([bool]$IUnderstandBilling) ([bool]$SelfTest) ([bool]$ForgetStoredApiKey)
    if ($SelfTest) {
        & (Join-Path (Split-Path -Parent $PSCommandPath) 'test-lambda-gh200.ps1')
        if ($LASTEXITCODE -ne 0) { throw "Self-test failed with exit code $LASTEXITCODE." }
        return
    }
    if ($ForgetStoredApiKey) { Remove-StoredCredential; return }
    Initialize-Paths
    Acquire-WatcherLock
    $key = Get-ApiKey
    $script:ApiHeaders = @{ Authorization = "Bearer $key" }
    Write-Log "$Mode watcher started."
    try {
        if ($Mode -eq 'Notify') { Start-NotifyWatcher }
        else { Start-LaunchWatcher }
    }
    catch {
        Write-Log "FAILED: $($_.Exception.Message)"
        if ($null -ne $script:ActiveInstance) {
            Write-Warning "Instance $((Get-Value $script:ActiveInstance 'id')) remains running; no termination was attempted."
        }
        Write-Host "Watcher log: $script:LogPath"
        throw
    }
    finally {
        $script:ApiHeaders = $null
        $script:Secrets.Clear()
        if ($null -ne $script:LockStream) { $script:LockStream.Dispose(); $script:LockStream = $null }
    }
}

if ($MyInvocation.InvocationName -ne '.') { Invoke-Main }
