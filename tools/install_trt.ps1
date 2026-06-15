param(
  [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'NVIDIA\TensorRT'),
  [string]$TensorRtZipUrl = $env:TRT_ZIP_URL,
  [string]$TensorRtZipPath = $env:TRT_ZIP_PATH
)

$ErrorActionPreference = 'Stop'

function Write-JsonInfo {
  param([string]$message, [hashtable]$extra = @{})
  $payload = @{
    level = 'info'
    message = $message
    ts = (Get-Date).ToString('o')
  }
  foreach ($k in $extra.Keys) { $payload[$k] = $extra[$k] }
  Write-Output ($payload | ConvertTo-Json -Compress)
}

function Write-JsonError {
  param([string]$message, [hashtable]$extra = @{})
  $payload = @{
    level = 'error'
    message = $message
    ts = (Get-Date).ToString('o')
  }
  foreach ($k in $extra.Keys) { $payload[$k] = $extra[$k] }
  [Console]::Error.WriteLine(($payload | ConvertTo-Json -Compress))
}

function Exit-WithCode {
  param([int]$code, [string]$message, [bool]$isError = $false)
  if ($isError) {
    Write-JsonError $message @{ exitCode = $code }
  } else {
    Write-JsonInfo $message @{ exitCode = $code }
  }
  exit $code
}

try {
  if ($env:OS -ne 'Windows_NT') {
    Exit-WithCode 2 'install_trt.ps1 is only supported on Windows' $true
  }

  if ([string]::IsNullOrWhiteSpace($TensorRtZipUrl)) {
    $TensorRtZipUrl = 'https://developer.download.nvidia.com/compute/machine-learning/tensorrt/11.0.1/zip/TensorRT-11.0.1.0.Windows.win10.cuda-13.0.zip'
  }

  $tempRoot = Join-Path $env:TEMP ("trt-install-" + [guid]::NewGuid().ToString('N'))
  $downloadPath = Join-Path $tempRoot 'tensorrt.zip'
  $extractPath = Join-Path $tempRoot 'extract'

  New-Item -Path $tempRoot -ItemType Directory -Force | Out-Null
  New-Item -Path $extractPath -ItemType Directory -Force | Out-Null

  if (-not [string]::IsNullOrWhiteSpace($TensorRtZipPath)) {
    if (-not (Test-Path -LiteralPath $TensorRtZipPath)) {
      Exit-WithCode 2 "TRT_ZIP_PATH does not exist: $TensorRtZipPath" $true
    }
    Write-JsonInfo 'Using local TensorRT archive' @{ path = $TensorRtZipPath }
    Copy-Item -LiteralPath $TensorRtZipPath -Destination $downloadPath -Force
  } else {
    Write-JsonInfo 'Downloading TensorRT archive' @{ url = $TensorRtZipUrl }
    Invoke-WebRequest -Uri $TensorRtZipUrl -OutFile $downloadPath -UseBasicParsing
  }

  if (-not (Test-Path -LiteralPath $downloadPath)) {
    Exit-WithCode 1 'Failed to acquire TensorRT archive' $true
  }

  Write-JsonInfo 'Extracting TensorRT archive' @{ archive = $downloadPath }
  Expand-Archive -LiteralPath $downloadPath -DestinationPath $extractPath -Force

  $extractedRoot = Get-ChildItem -LiteralPath $extractPath -Directory | Select-Object -First 1
  if ($null -eq $extractedRoot) {
    Exit-WithCode 1 'Archive extraction completed but no payload directory found' $true
  }

  if (Test-Path -LiteralPath $InstallRoot) {
    Write-JsonInfo 'Removing existing TensorRT directory' @{ path = $InstallRoot }
    Remove-Item -LiteralPath $InstallRoot -Recurse -Force
  }

  New-Item -Path $InstallRoot -ItemType Directory -Force | Out-Null
  Write-JsonInfo 'Copying TensorRT payload' @{ source = $extractedRoot.FullName; target = $InstallRoot }
  Copy-Item -LiteralPath (Join-Path $extractedRoot.FullName '*') -Destination $InstallRoot -Recurse -Force

  $pathCandidates = @(
    (Join-Path $InstallRoot 'lib'),
    (Join-Path $InstallRoot 'bin')
  ) | Where-Object { Test-Path -LiteralPath $_ }

  $currentUserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  $pathParts = @()
  if (-not [string]::IsNullOrWhiteSpace($currentUserPath)) {
    $pathParts = $currentUserPath.Split(';', [System.StringSplitOptions]::RemoveEmptyEntries)
  }

  foreach ($candidate in $pathCandidates) {
    if (-not ($pathParts -contains $candidate)) {
      $pathParts += $candidate
      Write-JsonInfo 'Added TensorRT folder to user PATH' @{ path = $candidate }
    }
  }

  [Environment]::SetEnvironmentVariable('Path', ($pathParts -join ';'), 'User')

  $wheelCount = @(Get-ChildItem -LiteralPath $InstallRoot -Recurse -Filter '*.whl' -ErrorAction SilentlyContinue).Count
  if ($wheelCount -eq 0) {
    Exit-WithCode 1 "TensorRT installed but no wheel files found under $InstallRoot" $true
  }

  Write-JsonInfo 'TensorRT installation completed' @{ installRoot = $InstallRoot; wheels = $wheelCount }

  if (Test-Path -LiteralPath $tempRoot) {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
  }

  Exit-WithCode 0 'TensorRT install succeeded'
} catch {
  Write-JsonError 'TensorRT install failed' @{ detail = $_.Exception.Message }
  exit 1
}
