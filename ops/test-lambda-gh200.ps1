$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot 'lambda-gh200.ps1')

$script:Passed = 0

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Assert-Throws {
    param([scriptblock]$Action, [string]$Pattern)
    try { & $Action; throw 'Expected an exception.' }
    catch {
        if ($_.Exception.Message -eq 'Expected an exception.') { throw }
        if ($_.Exception.Message -notmatch $Pattern) {
            throw "Unexpected exception: $($_.Exception.Message)"
        }
    }
}

function Test-Case {
    param([string]$Name, [scriptblock]$Body)
    & $Body
    $script:Passed++
    Write-Host "PASS $Name"
}

Test-Case 'Launch requires explicit billing acknowledgement' {
    Assert-Throws { Assert-Invocation 'Launch' $false $false $false } 'IUnderstandBilling'
    Assert-Invocation 'Launch' $true $false $false
}

Test-Case 'Secrets are redacted from text' {
    $safe = Protect-Text 'request failed with secret-value' @('secret-value')
    Assert-True ($safe -eq 'request failed with <redacted>') 'Secret was not redacted.'
}

Test-Case 'Retry-After controls bounded API retry delay' {
    $state = [pscustomobject]@{ Count = 0 }
    $sleeps = New-Object 'System.Collections.Generic.List[double]'
    $request = {
        param([hashtable]$Parameters)
        $state.Count++
        if ($state.Count -eq 1) {
            $error = New-Object Exception('rate limited')
            $error.Data['StatusCode'] = 429
            $error.Data['RetryAfter'] = '3'
            throw $error
        }
        return [pscustomobject]@{ Content = '{"data":{"ok":true}}' }
    }.GetNewClosure()
    $sleep = { param([double]$Seconds) $sleeps.Add($Seconds) }.GetNewClosure()
    $script:ApiHeaders = @{}
    $script:LastApiRequestUtc = [DateTime]::MinValue
    $result = Invoke-LambdaApi GET '/test' $null 2 $request $sleep
    Assert-True ([bool]$result.ok) 'Retry did not return the successful response.'
    Assert-True ($state.Count -eq 2) 'API helper did not retry exactly once.'
    Assert-True (@($sleeps | Where-Object { $_ -ge 3.0 }).Count -eq 1) 'Retry-After delay was not honored.'
}

Test-Case 'Multiple matching instances are refused' {
    $instance = [pscustomobject]@{
        id = 'instance-1'
        name = 'brazil-rv-gh200'
        status = 'active'
        region = [pscustomobject]@{ name = 'us-east-3' }
        instance_type = [pscustomobject]@{ name = 'gpu_1x_gh200' }
        tags = @(
            [pscustomobject]@{ key = 'project'; value = 'brazil-rv' },
            [pscustomobject]@{ key = 'managed-by'; value = 'gh200-watcher' }
        )
    }
    $second = $instance.PSObject.Copy()
    $second.id = 'instance-2'
    Assert-Throws { Select-ManagedInstance @($instance, $second) 'gpu_1x_gh200' } 'Multiple matching'
}

Test-Case 'SSH options use explicit key and isolated known-hosts file' {
    $arguments = Get-SshArguments 'C:\keys\brazil-rv' 'C:\known\instance-1'
    Assert-True ($arguments -contains '-i') 'Explicit -i option is missing.'
    Assert-True ($arguments -contains 'C:\keys\brazil-rv') 'Explicit key path is missing.'
    Assert-True ($arguments -contains 'IdentitiesOnly=yes') 'IdentitiesOnly is missing.'
    Assert-True ($arguments -contains 'UserKnownHostsFile=C:\known\instance-1') 'Known-hosts isolation is missing.'
}

Test-Case 'Transfer hash mismatch is rejected' {
    $directory = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    try {
        $bundle = Join-Path $directory 'test.bundle'
        $bootstrap = Join-Path $directory 'bootstrap.sh'
        [IO.File]::WriteAllText($bundle, 'bundle')
        [IO.File]::WriteAllText($bootstrap, 'bootstrap')
        $artifacts = [pscustomobject]@{
            BundlePath = $bundle
            BootstrapPath = $bootstrap
            BundleSha256 = '0' * 64
            BootstrapSha256 = (Get-FileHash $bootstrap -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        Assert-Throws { Assert-TransferArtifacts $artifacts } 'bundle hash mismatch'
    }
    finally { Remove-Item -LiteralPath $directory -Recurse -Force }
}

Test-Case 'Operations never auto-terminate or start training' {
    $watcher = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'lambda-gh200.ps1'))
    $bootstrap = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'lambda-gh200-bootstrap.sh'))
    Assert-True ($watcher -notmatch '/instance-operations/terminate') 'Automatic termination endpoint is present.'
    Assert-True ($bootstrap -notmatch 'modeling\.train(?! --help)') 'Bootstrap starts training.'
    Assert-True ($bootstrap -notmatch 'pytest|modeling\.sanity') 'Bootstrap runs paid-instance model/test preflights.'
}

Write-Host "$script:Passed compact Lambda safety tests passed."
exit 0
