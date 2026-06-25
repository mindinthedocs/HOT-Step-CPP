@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem HOT-Step engine build (CUDA).
rem Discovers CUDA, CMake, Ninja, and MSVC from PATH and standard environment
rem variables. No hardcoded CUDA or Visual Studio install paths are required.

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build"
set "CLEAN_BUILD=0"
set "CHECK_ONLY=0"

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--clean" (
    set "CLEAN_BUILD=1"
    shift
    goto :parse_args
)
if /I "%~1"=="--ninja" (
    shift
    goto :parse_args
)
if /I "%~1"=="--vs" (
    echo ERROR: --vs is not supported. HOT-Step CUDA builds require Ninja.
    exit /b 2
)
if /I "%~1"=="--check" (
    set "CHECK_ONLY=1"
    shift
    goto :parse_args
)
echo ERROR: Unknown argument: %~1
echo Usage: buildcuda.cmd [--clean] [--check] [--ninja^|--vs]
exit /b 2

:args_done
cd /d "%SCRIPT_DIR%" || exit /b 1

call :ensure_cuda
if errorlevel 1 exit /b %errorlevel%
call :ensure_cmake
if errorlevel 1 exit /b %errorlevel%
call :ensure_msvc
if errorlevel 1 exit /b %errorlevel%
call :ensure_ninja
if errorlevel 1 exit /b %errorlevel%
call :ensure_ccache
if errorlevel 1 exit /b %errorlevel%

if not defined CCACHE_SLOPPINESS set "CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime,locale"
if not defined CCACHE_BASEDIR set "CCACHE_BASEDIR=%SCRIPT_DIR%"
if not defined CCACHE_COMPILERCHECK set "CCACHE_COMPILERCHECK=none"

echo.
echo === HOT-Step CUDA engine build ===
echo Engine dir: %SCRIPT_DIR%
echo Build dir : %BUILD_DIR%
echo CUDA_PATH : %CUDA_PATH%
where nvcc
if errorlevel 1 exit /b 1
echo CMake    : %CMAKE_EXE%
"%CMAKE_EXE%" --version | findstr /b /c:"cmake version"
where cl
if errorlevel 1 exit /b 1
where ninja
if errorlevel 1 exit /b 1
where ccache
if errorlevel 1 exit /b 1
echo Generator : Ninja
echo CUDA arch : native
echo ccache   : required
echo.

if "%CHECK_ONLY%"=="1" (
    echo Environment check completed. No build requested.
    exit /b 0
)

if "%CLEAN_BUILD%"=="1" (
    echo Removing "%BUILD_DIR%"
    rd /s /q "%BUILD_DIR%" 2>nul
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%" || exit /b 1

rem Discover TensorRT SDK (optional — TRT acceleration for DiT w8a8 path).
rem Checks TRT_ROOT or TENSORRT_ROOT env vars; passes -DTRT_ROOT to CMake if found.
set "CMAKE_TRT_OPT="
if defined TRT_ROOT (
    set "CMAKE_TRT_OPT=-DTRT_ROOT:PATH=%TRT_ROOT%"
    echo TRT_ROOT : %TRT_ROOT%
) else if defined TENSORRT_ROOT (
    set "CMAKE_TRT_OPT=-DTRT_ROOT:PATH=%TENSORRT_ROOT%"
    echo TENSORRT_ROOT : %TENSORRT_ROOT%
) else (
    echo TRT      : not set (set TRT_ROOT or TENSORRT_ROOT for TRT acceleration)
)

"%CMAKE_EXE%" -S "%SCRIPT_DIR%." -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=native ^
  -DCMAKE_C_COMPILER_LAUNCHER=ccache ^
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ^
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache ^
  -DCMAKE_POLICY_DEFAULT_CMP0141=OLD ^
  -DCMAKE_C_FLAGS="/W0" ^
  -DCMAKE_CXX_FLAGS="/W0" ^
  -DCMAKE_CUDA_FLAGS="-w -Xcompiler /W0" ^
  %CMAKE_TRT_OPT%
if errorlevel 1 exit /b %errorlevel%

set "BUILD_PARALLEL=%NUMBER_OF_PROCESSORS%"
if not defined BUILD_PARALLEL set "BUILD_PARALLEL=1"

"%CMAKE_EXE%" --build "%BUILD_DIR%" --parallel %BUILD_PARALLEL%
exit /b %errorlevel%

:ensure_cuda
where nvcc >nul 2>nul
if not errorlevel 1 exit /b 0

for %%V in ("%CUDA_PATH%" "%CUDA_HOME%" "%CUDA_ROOT%") do (
    if not "%%~V"=="" if exist "%%~V\bin\nvcc.exe" (
        set "CUDA_PATH=%%~V"
        set "PATH=%%~V\bin;%PATH%"
        exit /b 0
    )
)

for /f "tokens=1,* delims==" %%A in ('set CUDA_PATH_V 2^>nul') do (
    if exist "%%~B\bin\nvcc.exe" (
        set "CUDA_PATH=%%~B"
        set "PATH=%%~B\bin;%PATH%"
        exit /b 0
    )
)

echo ERROR: nvcc.exe was not found.
echo        Add CUDA bin to PATH or set CUDA_PATH/CUDA_HOME/CUDA_ROOT.
exit /b 1

:ensure_cmake
set "CMAKE_EXE="
for /f "tokens=*" %%C in ('where cmake 2^>nul') do (
    if not defined CMAKE_EXE (
        call :is_supported_cmake "%%~C"
        if not errorlevel 1 set "CMAKE_EXE=%%~C"
    )
)
if defined CMAKE_EXE exit /b 0
echo ERROR: cmake.exe 3.18 or newer was not found on PATH.
exit /b 1

:is_supported_cmake
set "CMAKE_CANDIDATE=%~1"
set "CMAKE_CANDIDATE_DIR=%~dp1"
set "CMAKE_CANDIDATE_EXE=%~nx1"
set "CMAKE_VERSION="
pushd "%CMAKE_CANDIDATE_DIR%" >nul 2>nul
if errorlevel 1 exit /b 1
for /f "tokens=3" %%V in ('%CMAKE_CANDIDATE_EXE% --version 2^>nul ^| findstr /b /c:"cmake version"') do (
    if not defined CMAKE_VERSION set "CMAKE_VERSION=%%V"
)
popd >nul
if not defined CMAKE_VERSION exit /b 1
set "CMAKE_MAJOR="
set "CMAKE_MINOR="
for /f "tokens=1,2 delims=." %%A in ("!CMAKE_VERSION!") do (
    set "CMAKE_MAJOR=%%A"
    set "CMAKE_MINOR=%%B"
)
if not defined CMAKE_MAJOR exit /b 1
if not defined CMAKE_MINOR set "CMAKE_MINOR=0"
set /a CMAKE_MAJOR_NUM=CMAKE_MAJOR >nul 2>nul
if errorlevel 1 exit /b 1
set /a CMAKE_MINOR_NUM=CMAKE_MINOR >nul 2>nul
if errorlevel 1 exit /b 1
if !CMAKE_MAJOR_NUM! GTR 3 exit /b 0
if !CMAKE_MAJOR_NUM! EQU 3 if !CMAKE_MINOR_NUM! GEQ 18 exit /b 0
exit /b 1

:ensure_ninja
where ninja >nul 2>nul
if not errorlevel 1 exit /b 0
echo ERROR: ninja.exe was not found on PATH. Ninja is required for CUDA builds.
exit /b 1

:ensure_ccache
where ccache >nul 2>nul
if not errorlevel 1 exit /b 0
echo ERROR: ccache.exe was not found on PATH. ccache is required for CUDA builds.
exit /b 1

:ensure_msvc
where cl >nul 2>nul
if not errorlevel 1 exit /b 0

set "VSWHERE_DIR=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
set "VSWHERE=%VSWHERE_DIR%\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: cl.exe was not found and vswhere.exe is unavailable.
    echo        Run from a Developer Command Prompt or install Visual Studio C++ Build Tools.
    exit /b 1
)

set "VCVARS="
pushd "%VSWHERE_DIR%" >nul
for /f "tokens=*" %%I in ('vswhere.exe -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat" 2^>nul') do (
    if not defined VCVARS set "VCVARS=%%I"
)
popd >nul

if not defined VCVARS (
    echo ERROR: Could not find vcvars64.bat through vswhere.
    echo        Install the "Desktop development with C++" workload.
    exit /b 1
)

echo Loading MSVC environment: "%VCVARS%"
call "%VCVARS%"
if errorlevel 1 exit /b %errorlevel%

where cl >nul 2>nul
if errorlevel 1 (
    echo ERROR: vcvars64.bat completed but cl.exe is still unavailable.
    exit /b 1
)
exit /b 0
