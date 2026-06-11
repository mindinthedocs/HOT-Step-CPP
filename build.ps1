# HOT-Step-CPP portable Windows build orchestrator.
#
# Run from either:
#   - the HOT-Step-CPP repository root
#   - a parent folder containing HOT-Step-CPP/
#
# The script discovers CUDA, CMake, Ninja, Visual Studio, and ccache from
# PATH and standard environment variables. It does not require Administrator
# privileges and does not write Machine environment variables.

[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [switch]$SkipNpmInstall,
    [switch]$SkipUiBuild,
    [switch]$CleanEngine,
    [switch]$UseNinja,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$overallSuccess = $true

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host $Message -ForegroundColor White
}

function Resolve-RepoRoot {
    param([string]$RequestedRoot)

    $candidates = @()
    if ($RequestedRoot) { $candidates += (Resolve-Path -LiteralPath $RequestedRoot).Path }
    $scriptRoot = Split-Path -Parent $PSCommandPath
    $candidates += $scriptRoot
    $candidates += (Join-Path $scriptRoot "HOT-Step-CPP")
    $candidates += (Get-Location).Path
    $candidates += (Join-Path (Get-Location).Path "HOT-Step-CPP")

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if ($candidate -and
            (Test-Path (Join-Path $candidate "engine\buildcuda.cmd")) -and
            (Test-Path (Join-Path $candidate "server\package.json")) -and
            (Test-Path (Join-Path $candidate "ui\package.json"))) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "Could not locate HOT-Step-CPP repo root. Pass -RepoRoot or run from the repo/parent directory."
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-CommandPath {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    return $null
}

function Get-SupportedCMakePath {
    $commands = Get-Command cmake -All -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -Unique
    foreach ($path in $commands) {
        try {
            $versionLine = (& $path --version 2>$null | Select-Object -First 1)
        } catch {
            continue
        }
        if ($versionLine -match 'cmake version\s+(\d+)\.(\d+)') {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 18)) {
                return $path
            }
        }
    }
    return $null
}

function Add-PathIfExists {
    param([string]$PathToAdd)
    if (-not $PathToAdd -or -not (Test-Path -LiteralPath $PathToAdd)) { return }
    $parts = $env:Path -split ';' | Where-Object { $_ }
    if ($parts -notcontains $PathToAdd) {
        $env:Path = "$PathToAdd;$env:Path"
    }
}

function Import-CudaPath {
    if (Test-Command "nvcc") { return }

    $cudaRoots = @(
        $env:CUDA_PATH,
        $env:CUDA_HOME,
        $env:CUDA_ROOT
    ) + (Get-ChildItem Env:CUDA_PATH_V* -ErrorAction SilentlyContinue | Sort-Object Name -Descending | ForEach-Object { $_.Value })

    foreach ($root in $cudaRoots | Where-Object { $_ } | Select-Object -Unique) {
        Add-PathIfExists (Join-Path $root "bin")
        if (Test-Command "nvcc") { return }
    }
}

function Import-VsDevEnvironment {
    if (Test-Command "cl") { return }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) {
        Write-Warning "vswhere.exe not found. If cl.exe is not already on PATH, install Visual Studio C++ Build Tools."
        return
    }

    $vcvars = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat" 2>$null | Select-Object -First 1
    if (-not $vcvars) {
        Write-Warning "Could not locate vcvars64.bat through vswhere. Install the Desktop development with C++ workload."
        return
    }

    Write-Host "   Loading MSVC environment: $vcvars" -ForegroundColor Cyan
    $envDump = & cmd.exe /s /c "`"$vcvars`" >nul && set"
    foreach ($line in $envDump) {
        $idx = $line.IndexOf('=')
        if ($idx -le 0) { continue }
        $name = $line.Substring(0, $idx)
        $value = $line.Substring($idx + 1)
        Set-Item -Path "Env:$name" -Value $value
    }
}

function Show-Tool {
    param([string]$Name, [string]$RequiredMessage)
    $path = Get-CommandPath $Name
    if ($path) {
        Write-Host "   $Name -> $path" -ForegroundColor Green
        return $true
    }
    Write-Host "   Missing $Name. $RequiredMessage" -ForegroundColor Red
    return $false
}

function Show-CMakeTool {
    $path = Get-SupportedCMakePath
    if ($path) {
        $version = (& $path --version | Select-Object -First 1).Trim()
        Write-Host "   cmake -> $path ($version)" -ForegroundColor Green
        return $true
    }
    Write-Host "   Missing cmake 3.18+. Install CMake 3.18 or newer, or add it to PATH." -ForegroundColor Red
    return $false
}

$repoDir = Resolve-RepoRoot -RequestedRoot $RepoRoot
Set-Location $repoDir

Write-Host "=== HOT-Step-CPP Portable Build ===" -ForegroundColor Green
Write-Host "Repo root: $repoDir" -ForegroundColor Cyan

Write-Step "[Step 1/6] Discovering toolchain from PATH and environment..."
Import-CudaPath
Import-VsDevEnvironment

$toolchainOk = $true
$toolchainOk = (Show-Tool "git" "Install Git or add it to PATH.") -and $toolchainOk
$toolchainOk = (Show-CMakeTool) -and $toolchainOk
$toolchainOk = (Show-Tool "ninja" "Install Ninja or add it to PATH. Ninja is required for CUDA builds.") -and $toolchainOk
$toolchainOk = (Show-Tool "nvcc" "Install CUDA Toolkit or set CUDA_PATH/CUDA_HOME to a CUDA root.") -and $toolchainOk
$toolchainOk = (Show-Tool "cl" "Install Visual Studio C++ Build Tools or run from a Developer PowerShell.") -and $toolchainOk
if (-not $toolchainOk) { exit 1 }

Write-Step "[Step 2/6] Checking Node.js..."
if (-not (Test-Command "node") -or -not (Test-Command "npm")) {
    Write-Host "   Missing node/npm. Install Node.js 18-23 and ensure it is on PATH." -ForegroundColor Red
    exit 1
}
$nodeVersion = (& node -p "process.versions.node").Trim()
$nodeMajor = [int]($nodeVersion.Split('.')[0])
if ($nodeMajor -lt 18 -or $nodeMajor -ge 24) {
    Write-Host "   Node.js $nodeVersion found. This repo declares support for >=18 <24; continuing, but npm may warn." -ForegroundColor Yellow
} else {
    Write-Host "   Node.js $nodeVersion OK." -ForegroundColor Green
}

Write-Step "[Step 3/6] Checking required ccache..."
if (Test-Command "ccache") {
    Write-Host "   ccache -> $(Get-CommandPath ccache)" -ForegroundColor Green
    if (-not $env:CCACHE_DIR) {
        $env:CCACHE_DIR = Join-Path $env:LOCALAPPDATA "ccache"
    }
    if (-not $env:CCACHE_MAXSIZE) {
        $env:CCACHE_MAXSIZE = "10G"
    }
    Write-Host "   CCACHE_DIR=$env:CCACHE_DIR" -ForegroundColor Cyan
} else {
    Write-Host "   Missing ccache. Install ccache or add it to PATH; CUDA builds require it." -ForegroundColor Red
    exit 1
}

Write-Step "[Step 4/6] Building CUDA engine..."
$engineDir = Join-Path $repoDir "engine"
$targetCmd = Join-Path $engineDir "buildcuda.cmd"
if (-not (Test-Path -LiteralPath $targetCmd)) {
    Write-Host "   Could not find $targetCmd" -ForegroundColor Red
    exit 1
}

$engineArgs = @()
if ($CleanEngine) { $engineArgs += "--clean" }
if ($UseNinja) { $engineArgs += "--ninja" }
if ($CheckOnly) { $engineArgs += "--check" }

Push-Location $engineDir
try {
    & cmd.exe /c "`"$targetCmd`" $($engineArgs -join ' ')"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   Engine compilation failed." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    if ($CheckOnly) {
        Write-Host "   Engine environment check succeeded." -ForegroundColor Green
    } else {
        Write-Host "   Engine compiled successfully." -ForegroundColor Green
    }
} finally {
    Pop-Location
}

if ($CheckOnly) {
    Write-Host ""
    Write-Host "=== HOT-Step-CPP build environment check completed ===" -ForegroundColor Green
    exit 0
}

Write-Step "[Step 5/6] Installing server and UI dependencies..."
if (-not $SkipNpmInstall) {
    Push-Location (Join-Path $repoDir "server")
    try { npm install; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } finally { Pop-Location }

    Push-Location (Join-Path $repoDir "ui")
    try { npm install; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } finally { Pop-Location }
} else {
    Write-Host "   Skipped npm install by request." -ForegroundColor Yellow
}

Write-Step "[Step 6/6] Building UI production bundle..."
if (-not $SkipUiBuild) {
    Push-Location (Join-Path $repoDir "ui")
    try {
        npx vite build
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   UI compilation failed." -ForegroundColor Red
            exit $LASTEXITCODE
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "   Skipped UI build by request." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== HOT-Step-CPP build completed ===" -ForegroundColor Green
Write-Host "Run application:" -ForegroundColor White
Write-Host "  cd '$repoDir'" -ForegroundColor Yellow
Write-Host "  .\server.cmd" -ForegroundColor Yellow
