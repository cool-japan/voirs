$ErrorActionPreference = 'Stop'

$packageName = 'voirs'
$url = '{{BINARY_URL}}'
$checksum = '{{SHA256}}'
$checksumType = 'sha256'

$packageArgs = @{
    packageName   = $packageName
    url           = $url
    checksum      = $checksum
    checksumType  = $checksumType
    unzipLocation = Split-Path $MyInvocation.MyCommand.Definition
}

Install-ChocolateyZipPackage @packageArgs

# Add to PATH
$binPath = Join-Path (Split-Path $MyInvocation.MyCommand.Definition) 'voirs.exe'
Install-ChocolateyPath $binPath

Write-Host "VoiRS has been installed successfully!" -ForegroundColor Green
Write-Host "Run 'voirs --version' to verify the installation." -ForegroundColor Yellow