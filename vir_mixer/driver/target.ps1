# Preparation only until the user explicitly approves the selected system action.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('EnableTestMode','Sign','Trust','Install','Uninstall','RemoveTrust','DisableTestMode','VerifierOn','VerifierOff')][string]$Action,
    [string]$Package,
    [string]$PublishedInf,
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
if (-not $Apply) { throw "Preview only. Action $Action requires explicit approval and -Apply. No changes made." }
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Run in an elevated PowerShell on the approved TEST machine.' }
$kit = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10'
$version = '10.0.28000.0'
$signTool = "$kit\bin\$version\x64\signtool.exe"
$devcon = "$kit\Tools\$version\x64\devcon.exe"
$state = Join-Path $PSScriptRoot 'out\target-state'
New-Item -ItemType Directory -Force $state | Out-Null
function Native([string]$Tool, [string[]]$Arguments, [int[]]$Success = @(0)) {
    & $Tool @Arguments
    if ($LASTEXITCODE -notin $Success) { throw "$Tool failed: $LASTEXITCODE" }
    if ($LASTEXITCODE -in @(1,3010)) { Write-Warning 'Windows requests a restart. No automatic restart is performed.' }
}
function Certificate {
    $thumb = (Get-Content -LiteralPath "$state\thumbprint.txt" -Raw).Trim()
    if ($thumb -notmatch '^[0-9A-F]{40}$') { throw 'Invalid saved certificate thumbprint' }
    return $thumb
}
switch ($Action) {
    EnableTestMode {
        # Secure Boot may reject this. Do not disable Secure Boot automatically.
        Native bcdedit.exe @('/set','testsigning','on')
        Write-Host 'Restart is required before installation. Secure Boot settings were not modified.'
    }
    Sign {
        if (-not $Package) { throw '-Package must point to a validated unsigned package' }
        $Package = (Resolve-Path -LiteralPath $Package).Path
        $manifest = Get-Content "$Package\manifest.json" -Raw | ConvertFrom-Json
        foreach ($entry in $manifest.files.PSObject.Properties) {
            if ((Get-FileHash -LiteralPath (Join-Path $Package $entry.Name)).Hash -ne $entry.Value) { throw 'Unsigned package hash mismatch' }
        }
        if (Test-Path "$state\thumbprint.txt") {
            $thumb = Certificate
            $cert = Get-Item "Cert:\CurrentUser\My\$thumb"
        } else {
            $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject 'CN=VirMixer Development Test Only' -CertStoreLocation Cert:\CurrentUser\My -HashAlgorithm SHA256 -KeyLength 3072 -NotAfter (Get-Date).AddYears(1)
            Set-Content "$state\thumbprint.txt" $cert.Thumbprint
        }
        Export-Certificate -Cert $cert -FilePath "$state\VirMixerTest.cer" | Out-Null
        $signed = Join-Path $PSScriptRoot ('out\packages\Signed-' + [guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $signed | Out-Null
        Copy-Item "$Package\VirMixerAudio.sys","$Package\VirMixerAudio.inf" -Destination $signed
        # Embed-sign SYS first; regenerate CAT over that exact signed SYS; sign CAT last.
        Native $signTool @('sign','/fd','SHA256','/s','My','/sha1',$cert.Thumbprint,"$signed\VirMixerAudio.sys")
        Native "$kit\bin\$version\x86\Inf2Cat.exe" @("/driver:$signed",'/os:10_CO_X64,10_NI_X64,10_GE_X64')
        Native $signTool @('sign','/fd','SHA256','/s','My','/sha1',$cert.Thumbprint,"$signed\VirMixerAudio.cat")
        $hashes = @{}
        Get-ChildItem $signed -File | ForEach-Object { $hashes[$_.Name] = (Get-FileHash $_.FullName).Hash }
        @{files=$hashes; signed=$true; thumbprint=$cert.Thumbprint} | ConvertTo-Json -Depth 4 | Set-Content "$signed\manifest.json"
        Set-Content "$state\signed-package.txt" $signed
        Write-Host "Signed package: $signed. Trust has not yet been installed."
    }
    Trust {
        $thumb = Certificate
        $fileCert = New-Object Security.Cryptography.X509Certificates.X509Certificate2("$state\VirMixerTest.cer")
        if ($fileCert.Thumbprint -ne $thumb) { throw 'Certificate file mismatch' }
        Import-Certificate -FilePath "$state\VirMixerTest.cer" -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
        Import-Certificate -FilePath "$state\VirMixerTest.cer" -CertStoreLocation Cert:\LocalMachine\TrustedPublisher | Out-Null
    }
    Install {
        if (-not $Package) { $Package = (Get-Content "$state\signed-package.txt" -Raw).Trim() }
        $Package = (Resolve-Path -LiteralPath $Package).Path
        $manifest = Get-Content "$Package\manifest.json" -Raw | ConvertFrom-Json
        if (-not $manifest.signed -or $manifest.thumbprint -ne (Certificate)) { throw 'Expected our signed test package' }
        foreach ($entry in $manifest.files.PSObject.Properties) {
            if ((Get-FileHash -LiteralPath (Join-Path $Package $entry.Name)).Hash -ne $entry.Value) { throw 'Signed package hash mismatch' }
        }
        Native $signTool @('verify','/pa','/v',"$Package\VirMixerAudio.sys")
        Native $signTool @('verify','/pa','/v','/c',"$Package\VirMixerAudio.cat","$Package\VirMixerAudio.sys")
        Native $signTool @('verify','/pa','/v','/c',"$Package\VirMixerAudio.cat","$Package\VirMixerAudio.inf")
        $existing = @(Get-PnpDevice | Where-Object {
            $ids = (Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_HardwareIds -ErrorAction SilentlyContinue).Data
            $ids -contains 'Root\VirMixerAudio'
        })
        if ($existing.Count) { throw 'VirMixer devnode already exists; uninstall the old package first to avoid duplicates.' }
        Native $devcon @('install',"$Package\VirMixerAudio.inf",'Root\VirMixerAudio') @(0,1)
        Get-CimInstance Win32_PnPSignedDriver | Where-Object DeviceName -eq 'VirMixer Virtual Audio' | Select-Object DeviceID,InfName,DriverVersion | ConvertTo-Json | Set-Content "$state\installed.json"
        Write-Host 'Now inspect device status and execute PCM tests. Installation alone is not a PASS.'
    }
    Uninstall {
        if ($PublishedInf -notmatch '^oem\d+\.inf$') { throw 'Supply the exact installed -PublishedInf oemNN.inf from installed.json or pnputil /enum-drivers.' }
        $driver = Get-WindowsDriver -Online -Driver $PublishedInf
        if ($driver.ProviderName -ne 'VirMixer Project' -or [IO.Path]::GetFileName($driver.OriginalFileName) -ine 'VirMixerAudio.inf') { throw 'Refusing removal: package identity is not VirMixer' }
        Native $devcon @('remove','Root\VirMixerAudio') @(0,1)
        Native pnputil.exe @('/delete-driver',$PublishedInf,'/uninstall') @(0,3010)
    }
    RemoveTrust {
        $thumb = Certificate
        foreach ($store in 'LocalMachine\Root','LocalMachine\TrustedPublisher','CurrentUser\My') {
            $path = "Cert:\$store\$thumb"
            if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
        }
    }
    DisableTestMode { Native bcdedit.exe @('/set','testsigning','off'); Write-Host 'Restart required.' }
    VerifierOn { Native verifier.exe @('/standard','/driver','VirMixerAudio.sys'); Write-Host 'Separate approval required for this action; restart to begin Verifier testing.' }
    VerifierOff { Native verifier.exe @('/reset'); Write-Host 'Restart required. This resets all Driver Verifier settings.' }
}
