param(
  [string]$InstallRoot = '',
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
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

# ---------------------------------------------------------------------------
# TensorRT detection — mirrors the Node.js backend logic in
# systemService.ts::getTensorRtPath() so both sides agree on where TRT lives.
#
# Candidate search order (same as Node.js):
#   1. TRT_PATH env var
#   2. ~/AppData/Local/NVIDIA/TensorRT   (Windows user-local)
#   3. TENSORRT_LIBS env var
#   4. engine/deps/tensorrt_libs/         (auto-detected by Model Manager)
#   5. Parent dir of #3 / #4
#   6. Fallback: existing venv has tensorrt importable
#
# Each candidate is validated by checking for marker files or wheel files
# (same as hasTensorRtArtifacts() / hasWheelUnder() in systemService.ts).
# ---------------------------------------------------------------------------

function Test-TensorRtArtifacts {
  <#
  .SYNOPSIS
    Returns $true if the directory contains TensorRT marker files or wheel files.
    Mirrors hasTensorRtArtifacts() + hasWheelUnder() in systemService.ts.
  #>
  param([string]$RootDir)

  if (-not (Test-Path -LiteralPath $RootDir -PathType Container)) {
    return $false
  }

  # Marker files: checked in root, bin/, and lib/ subdirs
  $markers = @('trtexec.exe', 'trtexec', 'nvinfer_10.dll', 'nvinfer.dll')
  foreach ($marker in $markers) {
    if (Test-Path -LiteralPath (Join-Path $RootDir $marker))            { return $true }
    if (Test-Path -LiteralPath (Join-Path $RootDir (Join-Path 'bin' $marker))) { return $true }
    if (Test-Path -LiteralPath (Join-Path $RootDir (Join-Path 'lib' $marker))) { return $true }
  }

  # Wheel file: look for the exact TensorRT 11 wheel
  return (Find-TensorRtWheel -Dir $RootDir -Depth 3)
}

function Find-TensorRtWheel {
  <#
  .SYNOPSIS
    Recursively search for the TensorRT wheel up to a given depth.
    Looks for tensorrt-11.0.0.114-cp312-none-win_amd64.whl specifically,
    with a broader tensorrt*.whl fallback (mirrors hasWheelUnder() in systemService.ts).
  #>
  param([string]$Dir, [int]$Depth)

  if ($Depth -lt 0) { return $false }
  if (-not (Test-Path -LiteralPath $Dir -PathType Container)) { return $false }

  try {
    $items = Get-ChildItem -LiteralPath $Dir -ErrorAction Stop
  } catch {
    return $false
  }

  foreach ($item in $items) {
    if ($item.PSIsContainer) {
      if ($Depth -gt 0 -and (Find-TensorRtWheel -Dir $item.FullName -Depth ($Depth - 1))) {
        return $true
      }
    } else {
      if ($item.Name -like '*.whl' -and $item.Name -match 'tensorrt') {
        return $true
      }
    }
  }
  return $false
}

function Resolve-TensorRtPath {
  <#
  .SYNOPSIS
    Find the TensorRT installation directory using the same candidate order
    as systemService.ts::getTensorRtPath(). Returns the first candidate that
    passes Test-TensorRtArtifacts, or $null.
  #>

  # Candidate 1: TRT_PATH env var
  if ($env:TRT_PATH) {
    $resolved = Resolve-Path -LiteralPath $env:TRT_PATH -ErrorAction SilentlyContinue
    if ($resolved -and (Test-TensorRtArtifacts $resolved.Path)) {
      Write-JsonInfo 'TensorRT found via TRT_PATH env' @{ path = $resolved.Path }
      return $resolved.Path
    }
  }

  # Candidate 2: User-local default  (~/AppData/Local/NVIDIA/TensorRT on Windows)
  $userLocalTrtPath = Join-Path $env:LOCALAPPDATA 'NVIDIA\TensorRT'
  if (Test-TensorRtArtifacts $userLocalTrtPath) {
    Write-JsonInfo 'TensorRT found in user-local path' @{ path = $userLocalTrtPath }
    return $userLocalTrtPath
  }

  # Candidate 3: TENSORRT_LIBS env var
  $configuredTrtDir = ''
  if ($env:TENSORRT_LIBS) {
    $resolved = Resolve-Path -LiteralPath $env:TENSORRT_LIBS -ErrorAction SilentlyContinue
    if ($resolved) { $configuredTrtDir = $resolved.Path }
  }

# Candidate 4: Auto-detected engine/deps/tensorrt_libs/
    if (-not $configuredTrtDir) {
      $autoDepsDir = Join-Path $RepoRoot (Join-Path 'engine' (Join-Path 'deps' 'tensorrt_libs'))
      
      # FIX: Wrap both Test-Path commands in parentheses
      if ((Test-Path -LiteralPath (Join-Path $autoDepsDir 'nvinfer_10.dll')) -or
          (Test-Path -LiteralPath (Join-Path $autoDepsDir 'libnvinfer.so.10'))) {
        $configuredTrtDir = $autoDepsDir
      }
    }

  if ($configuredTrtDir -and (Test-TensorRtArtifacts $configuredTrtDir)) {
    Write-JsonInfo 'TensorRT found via configured/auto-detected trtLibs' @{ path = $configuredTrtDir }
    return $configuredTrtDir
  }

  # Candidate 5: Parent of configuredTrtDir
  if ($configuredTrtDir) {
    $parentDir = Split-Path -LiteralPath $configuredTrtDir -Parent
    if ($parentDir -and (Test-TensorRtArtifacts $parentDir)) {
      Write-JsonInfo 'TensorRT found in parent of trtLibs dir' @{ path = $parentDir }
      return $parentDir
    }
  }

  # Candidate 6: Fallback — check if existing venv already has tensorrt importable
  $venvPython = Join-Path $RepoRoot (Join-Path '.venv' 'Scripts\python.exe')
  if (Test-Path -LiteralPath $venvPython) {
    $trtVer = & $venvPython -c "import tensorrt; print(tensorrt.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $trtVer) {
      Write-JsonInfo 'TensorRT found via venv Python import (fallback)' @{ version = $trtVer.Trim(); path = $userLocalTrtPath }
      return $userLocalTrtPath
    }
  }

  return $null
}

# ---------------------------------------------------------------------------
# Hard-coded dependency list — derived from all Python scripts in the repo.
# When adding a new third-party import to any .py file, update this table.
#
# Format: @( '<pip-package>[extras]', '<import-check-module>' )
#   - pip-package: what to pass to `pip install`
#   - import-check-module: the module name used in `import <name>` (for
#     verification); use $null if no import-time check is desired.
#
# TensorRT wheel:
#   The project uses TensorRT 11, specifically
#   tensorrt-11.0.0.114-cp312-none-win_amd64.whl
#   which must be present under the resolved TensorRT install directory.
# ---------------------------------------------------------------------------
$TrtWheelName = 'tensorrt-11.0.0.114-cp312-none-win_amd64.whl'

$PythonPackages = @(
  # Core numeric / scientific
  @( 'numpy',                          'numpy' ),
  @( 'torch',                          'torch' ),         # installed separately with --index-url
  @( 'einops',                         'einops' ),

  # HuggingFace ecosystem
  @( 'transformers',                   'transformers' ),
  @( 'safetensors',                    'safetensors' ),
  @( 'diffusers',                      'diffusers' ),

  # ONNX
  @( 'onnx',                           'onnx' ),
  @( 'onnxruntime-gpu',                'onnxruntime' ),

  # NVIDIA / CUDA 13 (tensorrt is installed separately from the local wheel)
  @( 'nvidia-cuda-runtime',            'cuda.bindings' ),  
  @( 'nvidia-cublas',                  'nvidia.cublas' ),  
  @( 'nvidia-cuda-nvrtc',              'nvidia.cuda.nvrtc' ),
  @( 'nvidia-modelopt[onnx]',          'modelopt' ),       

  # Audio
  @( 'librosa',                        'librosa' ),
  @( 'soundfile',                      'soundfile' ),

  # Utilities
  @( 'requests',                       'requests' ),
  @( 'gguf',                           'gguf' ),
  
  # Model dependencies
  @( 'vector-quantize-pytorch',        'vector_quantize_pytorch' )
)

# ---------------------------------------------------------------------------
# PyTorch CPU wheel index URL
# ---------------------------------------------------------------------------
$PyTorchCpuIndex = 'https://download.pytorch.org/whl/cpu'

try {
  if ($env:OS -ne 'Windows_NT') {
    Exit-WithCode 2 'setup_venv.ps1 is only supported on Windows' $true
  }

  # -----------------------------------------------------------------------
  # Resolve TensorRT installation directory.
  # If -InstallRoot was passed explicitly and is non-empty, use it.
  # Otherwise, auto-detect using the same logic as the Node.js backend.
  # -----------------------------------------------------------------------
  $trtPath = $InstallRoot
  if (-not $trtPath) {
    $trtPath = Resolve-TensorRtPath
  }

  if (-not $trtPath) {
    Exit-WithCode 2 'TensorRT not found. Install TensorRT before setting up Python venv. Searched: TRT_PATH env, ~/AppData/Local/NVIDIA/TensorRT, TENSORRT_LIBS env, engine/deps/tensorrt_libs/, venv import.' $true
  }

  Write-JsonInfo 'Using TensorRT installation' @{ path = $trtPath }

  # Look for the specific TensorRT 11 wheel first
  $trtWheel = Get-ChildItem -LiteralPath $trtPath -Recurse -Filter $TrtWheelName -ErrorAction SilentlyContinue |
    Select-Object -First 1

  if ($null -eq $trtWheel) {
    # Fallback: any tensorrt wheel for cp312 on win_amd64
    Write-JsonInfo "Exact wheel $TrtWheelName not found, searching for any TensorRT cp312 wheel" @{ path = $trtPath }
    $trtWheel = Get-ChildItem -LiteralPath $trtPath -Recurse -Filter 'tensorrt-*-cp312-none-win_amd64.whl' -ErrorAction SilentlyContinue |
      Sort-Object Name -Descending |
      Select-Object -First 1
  }

  if ($null -eq $trtWheel) {
    # Last resort: any tensorrt wheel at all
    Write-JsonInfo 'No cp312 TensorRT wheel found, searching for any tensorrt wheel' @{ path = $trtPath }
    $trtWheel = Get-ChildItem -LiteralPath $trtPath -Recurse -Filter 'tensorrt-*.whl' -ErrorAction SilentlyContinue |
      Sort-Object Name -Descending |
      Select-Object -First 1
  }

  if ($null -eq $trtWheel) {
    Exit-WithCode 2 "TensorRT wheel not found under $trtPath. Expected: $TrtWheelName" $true
  }

  $venvDir = Join-Path $RepoRoot '.venv'
  $venvPython = Join-Path $venvDir 'Scripts\python.exe'

  if (Test-Path -LiteralPath $venvDir) {
    Write-JsonInfo 'Removing existing venv' @{ path = $venvDir }
    Remove-Item -LiteralPath $venvDir -Recurse -Force
  }

  $python312 = $null
  $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
  if ($null -ne $pyLauncher) {
    & py -3.12 -c "import sys; print(sys.executable)" 2>$null | ForEach-Object {
      if (Test-Path -LiteralPath $_) { $python312 = $_ }
    }
  }
  if ($null -eq $python312) {
    $candidates = @(
      (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
      'C:\Program Files\Python312\python.exe'
    )
    foreach ($candidate in $candidates) {
      if (Test-Path -LiteralPath $candidate) {
        $python312 = $candidate
        break
      }
    }
  }
  if ($null -eq $python312) {
    Exit-WithCode 1 'Python 3.12 not found. Install Python 3.12 or register it with the py launcher.' $true
  }

  Write-JsonInfo 'Creating Python 3.12 virtual environment' @{ path = $venvDir; python = $python312 }
  & $python312 -m venv $venvDir
  if ($LASTEXITCODE -ne 0) {
    Exit-WithCode 1 'Failed to create Python 3.12 virtual environment' $true
  }

  if (-not (Test-Path -LiteralPath $venvPython)) {
    Exit-WithCode 1 "Venv python executable missing at $venvPython" $true
  }

  Write-JsonInfo 'Upgrading pip in virtual environment'
  & $venvPython -m pip install --upgrade pip
  if ($LASTEXITCODE -ne 0) {
    Exit-WithCode 1 'Failed to upgrade pip' $true
  }

  # -----------------------------------------------------------------------
  # Install PyTorch CPU Support
  # -----------------------------------------------------------------------
  Write-JsonInfo 'Installing PyTorch (CPU version)' @{ index = $PyTorchCpuIndex }
  & $venvPython -m pip install torch --index-url $PyTorchCpuIndex
  if ($LASTEXITCODE -ne 0) {
    Exit-WithCode 1 'Failed to install PyTorch (CPU)' $true
  }

  # -----------------------------------------------------------------------
  # Install TensorRT from the local wheel.
  # -----------------------------------------------------------------------
  Write-JsonInfo 'Installing TensorRT wheel into venv' @{ wheel = $trtWheel.FullName }
  & $venvPython -m pip install $trtWheel.FullName
  if ($LASTEXITCODE -ne 0) {
    Exit-WithCode 1 "Failed to install TensorRT wheel: $($trtWheel.FullName)" $true
  }

  # -----------------------------------------------------------------------
  # Install remaining Python dependencies.
  # Packages that were already pulled in as transitive deps (e.g. numpy via
  # torch) will be skipped quickly by pip.
  # -----------------------------------------------------------------------
  Write-JsonInfo 'Installing Python project dependencies' @{ count = $PythonPackages.Count }

  # Build a single pip-install argument list so pip can resolve conflicts
  # in one pass.
  $pipArgs = @()
  foreach ($pkg in $PythonPackages) {
    $pipName = $pkg[0]
    # Skip torch — already installed with CPU index above.
    # Skip tensorrt — already installed from local wheel above.
    if ($pipName -eq 'torch') { continue }
    $pipArgs += $pipName
  }

  & $venvPython -m pip install @pipArgs
  if ($LASTEXITCODE -ne 0) {
    Exit-WithCode 1 'Failed to install one or more Python packages' $true
  }

  Write-JsonInfo 'Python venv setup completed' @{ venv = $venvDir }
  Exit-WithCode 0 'setup-venv succeeded'
} catch {
  Write-JsonError 'setup-venv failed' @{ detail = $_.Exception.Message }
  exit 1
}