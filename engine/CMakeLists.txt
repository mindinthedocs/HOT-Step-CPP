cmake_minimum_required(VERSION 3.14)
project(acestep-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# version.h: embed git commit hash into all binaries.
# runs on every build, only rewrites if the hash changed.
set(VERSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/version.h")
add_custom_target(version ALL
    COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DOUTPUT=${VERSION_OUTPUT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/version.cmake"
    BYPRODUCTS "${VERSION_OUTPUT}"
    COMMENT "Checking git version"
)

# pthread: required explicitly on older glibc (< 2.34) where libpthread
# is not merged into libc. Modern distros link it implicitly but aarch64
# and older x86_64 toolchains need the explicit dependency.
find_package(Threads REQUIRED)

# Suppress MSVC fopen/sprintf deprecation warnings and Windows.h macro pollution.
# NOMINMAX: prevents Windows.h from defining min/max macros that collide with
#           std::min/std::max (causes C2589 errors in solvers/schedulers).
# WIN32_LEAN_AND_MEAN: reduces Windows.h header bloat.
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

# Static MSVC runtime (/MT) for portable release builds.
# Eliminates the VC++ Redistributable dependency for end users.
# Only enable during release builds: -DHOT_STEP_STATIC_RUNTIME=ON
option(HOT_STEP_STATIC_RUNTIME "Use static MSVC runtime (/MT) for portable builds" OFF)
if(HOT_STEP_STATIC_RUNTIME AND MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    message(STATUS "MSVC runtime: static (/MT)")
endif()

# Put executables and backend .so in the same directory (build root).
# Without this, ggml defaults to bin/ for .so but executables stay in root,
# and ggml_backend_load_all() can't find the backends at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# macOS rpath: make binaries relocatable (portable release support).
# Without this, CMake bakes the absolute build directory into LC_RPATH,
# which breaks on any machine other than the one that built it.
# @executable_path tells dyld to look for dylibs next to the binary.
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@executable_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_MACOSX_RPATH TRUE)
endif()

# Linux rpath: make binaries relocatable (portable release support).
# $ORIGIN tells the dynamic linker to search for .so files next to the binary.
if(UNIX AND NOT APPLE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# DiT tensor names can exceed default GGML_MAX_NAME of 64
add_compile_definitions(GGML_MAX_NAME=128)

# Harden: mark fread/fwrite/etc with warn_unused_result on all platforms
if(NOT MSVC)
    add_compile_definitions(_FORTIFY_SOURCE=2)
endif()

# CUDA architectures: cover Turing to Blackwell for distributed binaries (CI: CUDA 13.1 / 12.8).
# Users can override with -DCMAKE_CUDA_ARCHITECTURES=native for local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit QUIET)
    # Base arch list: Turing through Lovelace/Ada
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-virtual;86-real;89-real;90-real")
    if(CUDAToolkit_FOUND)
        # Blackwell sm_120a: CUDA 12.8+
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            list(APPEND CMAKE_CUDA_ARCHITECTURES "120a-real")
        endif()
        # Blackwell sm_121a: CUDA 12.9+ (same as upstream GGML)
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
            list(APPEND CMAKE_CUDA_ARCHITECTURES "121a-real")
        endif()
    endif()
endif()

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
add_subdirectory(ggml)

# cpp-httplib (HTTP server library, used by ace-server)
add_subdirectory(vendor/cpp-httplib)

# Shared compile options and ggml linkage
macro(link_ggml_backends target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
    )
    target_include_directories(${target} SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /wd4100 /wd4505)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow -Wconversion
                              -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    endif()
    target_link_libraries(${target} PRIVATE ggml Threads::Threads)
    if(TARGET ggml-base)
        target_link_libraries(${target} PRIVATE ggml-base)
    endif()
    foreach(backend cpu blas cuda metal vulkan)
        if(TARGET ggml-${backend})
            get_target_property(CURRENT_BACKEND_TYPE ggml-${backend} TYPE)
            if (CURRENT_BACKEND_TYPE STREQUAL "MODULE_LIBRARY")
                # DL mode: backend is loaded at runtime via dlopen,
                # skip all link-time deps.
                continue()
            endif()
            target_link_libraries(${target} PRIVATE ggml-${backend})
        endif()
    endforeach()
    add_dependencies(${target} version)
endmacro()

# yyjson (MIT, fast JSON parser/writer)
add_library(yyjson STATIC vendor/yyjson/yyjson.c)
target_include_directories(yyjson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/yyjson)
if(MSVC)
    target_compile_options(yyjson PRIVATE /W0)
else()
    target_compile_options(yyjson PRIVATE -w)
endif()

# Lua 5.4 (MIT, embedded scripting for plugin system)
# All .c files except lua.c (standalone interpreter) and luac.c (compiler)
file(GLOB LUA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/vendor/lua/*.c")
list(FILTER LUA_SOURCES EXCLUDE REGEX "(lua|luac)\\.c$")
add_library(lua54 STATIC ${LUA_SOURCES})
target_include_directories(lua54 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor/lua)
if(MSVC)
    target_compile_options(lua54 PRIVATE /W0)
else()
    target_compile_options(lua54 PRIVATE -w)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime (SuperSep stem separation + VAE-ORT TensorRT acceleration)
# ─────────────────────────────────────────────────────────────────────────────
# Pre-built ORT GPU package. Resolution order:
#   1. ORT_ROOT cmake variable
#   2. ONNXRUNTIME_ROOT environment variable
#   3. Auto-detect from engine/deps/onnxruntime/ (populated by buildall.cmd)
#
# CUDAToolkit detection: the find_package at L73 is conditional on
# CMAKE_CUDA_ARCHITECTURES, so on cached re-configures CUDAToolkit_FOUND
# may be unset. Ensure it's always available for CUDA EP support.
find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
    set(_CUDA_ROOT "")
    foreach(_CUDA_ENV CUDA_PATH CUDA_HOME)
        if(NOT _CUDA_ROOT AND DEFINED ENV{${_CUDA_ENV}} AND EXISTS "$ENV{${_CUDA_ENV}}/include/cuda_runtime_api.h")
            set(_CUDA_ROOT "$ENV{${_CUDA_ENV}}")
        endif()
    endforeach()
    if(NOT _CUDA_ROOT)
        file(GLOB _CUDA_ROOT_CANDIDATES "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v*")
        list(SORT _CUDA_ROOT_CANDIDATES)
        list(REVERSE _CUDA_ROOT_CANDIDATES)
        foreach(_CUDA_CANDIDATE ${_CUDA_ROOT_CANDIDATES})
            if(NOT _CUDA_ROOT AND EXISTS "${_CUDA_CANDIDATE}/include/cuda_runtime_api.h")
                set(_CUDA_ROOT "${_CUDA_CANDIDATE}")
            endif()
        endforeach()
    endif()
    if(_CUDA_ROOT)
        set(CUDAToolkit_FOUND TRUE)
        set(CUDAToolkit_INCLUDE_DIRS "${_CUDA_ROOT}/include")
        set(_CUDA_CUDART_LIB "${_CUDA_ROOT}/lib/x64/cudart.lib")
        set(_CUDA_CUDART_STATIC_LIB "${_CUDA_ROOT}/lib/x64/cudart_static.lib")
        if(_CUDA_ROOT MATCHES "v([0-9]+)\\.([0-9]+)")
            set(CUDAToolkit_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
            set(CUDAToolkit_VERSION_MAJOR "${CMAKE_MATCH_1}")
        endif()
        if(NOT TARGET CUDA::cudart AND EXISTS "${_CUDA_CUDART_LIB}")
            add_library(CUDA::cudart UNKNOWN IMPORTED)
            set_target_properties(CUDA::cudart PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${_CUDA_ROOT}/include"
                IMPORTED_IMPLIB "${_CUDA_CUDART_LIB}"
                IMPORTED_LOCATION "${_CUDA_CUDART_LIB}")
            file(GLOB _CUDA_CUDART_DLL "${_CUDA_ROOT}/bin/cudart64_*.dll")
            if(_CUDA_CUDART_DLL)
                list(GET _CUDA_CUDART_DLL 0 _CUDA_CUDART_DLL_FIRST)
                set_target_properties(CUDA::cudart PROPERTIES IMPORTED_LOCATION "${_CUDA_CUDART_DLL_FIRST}")
            endif()
        endif()
        if(NOT TARGET CUDA::cudart_static AND EXISTS "${_CUDA_CUDART_STATIC_LIB}")
            add_library(CUDA::cudart_static UNKNOWN IMPORTED)
            set_target_properties(CUDA::cudart_static PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${_CUDA_ROOT}/include"
                IMPORTED_LOCATION "${_CUDA_CUDART_STATIC_LIB}")
        endif()
        message(STATUS "[CUDA] Found Toolkit fallback ${CUDAToolkit_VERSION} at ${_CUDA_ROOT}")
    endif()
endif()
option(HOT_STEP_SUPERSEP "Enable SuperSep stem separation (requires ONNX Runtime)" ON)

set(ORT_ROOT "" CACHE PATH "Path to ONNX Runtime pre-built package")
if(NOT ORT_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ORT_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
# Auto-detect from deps directory (buildall.cmd downloads here)
if(NOT ORT_ROOT)
    set(_ORT_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime")
    if(EXISTS "${_ORT_DEPS_DIR}/include/onnxruntime_cxx_api.h")
        set(ORT_ROOT "${_ORT_DEPS_DIR}")
        message(STATUS "ORT auto-detected at ${ORT_ROOT}")
    endif()
endif()

set(SUPERSEP_ENABLED FALSE)
if(HOT_STEP_SUPERSEP AND ORT_ROOT)
    if(EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
        set(SUPERSEP_ENABLED TRUE)
    else()
        message(WARNING "ORT_ROOT set but onnxruntime_cxx_api.h not found at ${ORT_ROOT}/include")
    endif()
endif()

# SuperSep library (ONNX Runtime inference + STFT)
add_library(supersep STATIC src/supersep.cpp)
target_include_directories(supersep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/pocketfft
    ${CMAKE_CURRENT_BINARY_DIR}
)
if(MSVC)
    target_compile_options(supersep PRIVATE /W4 /wd4100 /wd4505 /wd4244 /wd4267)
else()
    target_compile_options(supersep PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-sign-conversion)
endif()

if(SUPERSEP_ENABLED)
    target_compile_definitions(supersep PUBLIC HOT_STEP_SUPERSEP)
    target_include_directories(supersep PUBLIC "${ORT_ROOT}/include")
    target_link_directories(supersep PUBLIC "${ORT_ROOT}/lib")
    target_link_libraries(supersep PUBLIC onnxruntime)
    message(STATUS "SuperSep: ENABLED (ORT at ${ORT_ROOT})")
else()
    message(STATUS "SuperSep: DISABLED (set ORT_ROOT to enable)")
endif()

if(CUDAToolkit_FOUND)
    target_compile_definitions(supersep PUBLIC GGML_USE_CUDA)
endif()

# macOS: CoreML EP requires Apple frameworks
if(APPLE AND SUPERSEP_ENABLED)
    target_link_libraries(supersep PUBLIC "-framework CoreML" "-framework Foundation")
endif()
# TensorRT Native SDK (DiT TRT acceleration + LoRA refitting)
# ─────────────────────────────────────────────────────────────────────────────
# Pre-built TRT SDK (headers + import libs). Resolution order:
#   1. TRT_ROOT CMake cache variable
#   2. TENSORRT_ROOT environment variable
#   3. engine/deps/tensorrt/
#
# TensorRT 11 is required for this backend. Windows package names are versioned
# (nvinfer_11.lib, nvonnxparser_11.lib); Linux packages are usually neutral
# (libnvinfer.so, libnvonnxparser.so). Prefer exact TRT 11 libraries when they
# exist, then fall back to neutral names after the header version check passes.
set(TRT_ENABLED FALSE)
set(_TRT_DEFAULT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/tensorrt")
set(TRT_ROOT "" CACHE PATH "Path to TensorRT 11 SDK root")

# Accept both the CMake cache variable and the two environment names used by
# scripts/docs. This is intentionally before falling back to engine/deps so a
# PowerShell setting like `$env:TRT_ROOT = "D:\ai\TensorRT"` works even when
# build orchestration does not forward -DTRT_ROOT explicitly. If an older CMake
# cache contains the built-in fallback path, treat it as unset so the env var can
# repair an existing build directory without --clean.
set(_TRT_ROOT_IS_DEFAULT_FALLBACK FALSE)
if(TRT_ROOT STREQUAL _TRT_DEFAULT_ROOT)
    set(_TRT_ROOT_IS_DEFAULT_FALLBACK TRUE)
endif()
if((NOT TRT_ROOT OR _TRT_ROOT_IS_DEFAULT_FALLBACK) AND DEFINED ENV{TRT_ROOT})
    set(TRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "Path to TensorRT 11 SDK root" FORCE)
endif()
if((NOT TRT_ROOT OR _TRT_ROOT_IS_DEFAULT_FALLBACK) AND DEFINED ENV{TENSORRT_ROOT})
    set(TRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "Path to TensorRT 11 SDK root" FORCE)
endif()
if(NOT TRT_ROOT)
    set(TRT_ROOT "${_TRT_DEFAULT_ROOT}" CACHE PATH "Path to TensorRT 11 SDK root" FORCE)
endif()

# Some local installs use a wrapper directory (e.g. D:/ai/TensorRT containing a
# TensorRT-11.x/ child). If the provided root does not directly contain the SDK
# include directory, auto-select the first child that does.
if(TRT_ROOT AND NOT EXISTS "${TRT_ROOT}/include/NvInfer.h")
    file(GLOB _TRT_ROOT_CHILDREN LIST_DIRECTORIES TRUE "${TRT_ROOT}/*")
    foreach(_TRT_ROOT_CHILD IN LISTS _TRT_ROOT_CHILDREN)
        if(EXISTS "${_TRT_ROOT_CHILD}/include/NvInfer.h")
            set(TRT_ROOT "${_TRT_ROOT_CHILD}" CACHE PATH "Path to TensorRT 11 SDK root" FORCE)
            break()
        endif()
    endforeach()
endif()

set(_TRT_INC_DIR "${TRT_ROOT}/include")
set(_TRT_LIB_DIR "${TRT_ROOT}/lib")
set(_TRT_BIN_DIR "${TRT_ROOT}/bin")
set(_TRT_VERSION_HEADER "${_TRT_INC_DIR}/NvInferVersion.h")
set(_TRT_NVINFER_LIB "")
set(_TRT_NVONNXPARSER_LIB "")

if(EXISTS "${_TRT_INC_DIR}/NvInfer.h" AND EXISTS "${_TRT_VERSION_HEADER}")
    function(_trt_read_define out_name define_name)
        file(STRINGS "${_TRT_VERSION_HEADER}" _line REGEX "^#define[ \t]+${define_name}[ \t]+[^ \t/]+")
        string(REGEX REPLACE "^#define[ \t]+${define_name}[ \t]+([^ \t/]+).*" "\\1" _value "${_line}")
        set(${out_name} "${_value}" PARENT_SCOPE)
    endfunction()

    _trt_read_define(TRT_VERSION_MAJOR NV_TENSORRT_MAJOR)
    _trt_read_define(TRT_VERSION_MINOR NV_TENSORRT_MINOR)
    _trt_read_define(TRT_VERSION_PATCH NV_TENSORRT_PATCH)
    foreach(_TRT_VERSION_PART MAJOR MINOR PATCH)
        if(NOT TRT_VERSION_${_TRT_VERSION_PART} MATCHES "^[0-9]+$")
            _trt_read_define(TRT_VERSION_${_TRT_VERSION_PART} TRT_${_TRT_VERSION_PART}_ENTERPRISE)
        endif()
    endforeach()
    if(NOT TRT_VERSION_MAJOR STREQUAL "11")
        message(FATAL_ERROR "[TRT] TensorRT 11.x is required; found ${TRT_VERSION_MAJOR}.${TRT_VERSION_MINOR}.${TRT_VERSION_PATCH} at ${TRT_ROOT}")
    endif()

    foreach(_TRT_NVINFER_NAME nvinfer_11 nvinfer)
        foreach(_TRT_LIB_EXT lib so dylib)
            if(NOT _TRT_NVINFER_LIB AND EXISTS "${_TRT_LIB_DIR}/${_TRT_NVINFER_NAME}.${_TRT_LIB_EXT}")
                set(_TRT_NVINFER_LIB "${_TRT_LIB_DIR}/${_TRT_NVINFER_NAME}.${_TRT_LIB_EXT}")
            endif()
        endforeach()
    endforeach()
    foreach(_TRT_NVONNXPARSER_NAME nvonnxparser_11 nvonnxparser)
        foreach(_TRT_LIB_EXT lib so dylib)
            if(NOT _TRT_NVONNXPARSER_LIB AND EXISTS "${_TRT_LIB_DIR}/${_TRT_NVONNXPARSER_NAME}.${_TRT_LIB_EXT}")
                set(_TRT_NVONNXPARSER_LIB "${_TRT_LIB_DIR}/${_TRT_NVONNXPARSER_NAME}.${_TRT_LIB_EXT}")
            endif()
        endforeach()
    endforeach()

    find_library(_TRT_NVINFER_LIB
        NAMES nvinfer_11 nvinfer
        PATHS "${_TRT_LIB_DIR}"
        NO_DEFAULT_PATH)
    find_library(_TRT_NVONNXPARSER_LIB
        NAMES nvonnxparser_11 nvonnxparser
        PATHS "${_TRT_LIB_DIR}"
        NO_DEFAULT_PATH)

    if(_TRT_NVINFER_LIB AND _TRT_NVONNXPARSER_LIB)
        set(TRT_ENABLED TRUE)
        message(STATUS "[TRT] Found TensorRT ${TRT_VERSION_MAJOR}.${TRT_VERSION_MINOR}.${TRT_VERSION_PATCH} at ${TRT_ROOT}")
        message(STATUS "[TRT] nvinfer: ${_TRT_NVINFER_LIB}")
        message(STATUS "[TRT] nvonnxparser: ${_TRT_NVONNXPARSER_LIB}")
    else()
        message(STATUS "[TRT] TensorRT headers found at ${TRT_ROOT}, but nvinfer/nvonnxparser libraries were not found in ${_TRT_LIB_DIR}")
    endif()
else()
    message(STATUS "[TRT] Not found (set TRT_ROOT or TENSORRT_ROOT to a TensorRT 11 SDK)")
    message(STATUS "[TRT] Checked root: ${TRT_ROOT}")
    message(STATUS "[TRT] Expected headers: ${_TRT_INC_DIR}/NvInfer.h and ${_TRT_VERSION_HEADER}")
endif()

# Core library (shared between binaries)
add_library(acestep-core STATIC
    src/request.cpp
    src/model-store.cpp
    src/pipeline-lm.cpp
    src/pipeline-synth.cpp
    src/pipeline-synth-ops.cpp
    src/pipeline-understand.cpp
)
target_link_libraries(acestep-core PUBLIC yyjson lua54 supersep)
link_ggml_backends(acestep-core)

# TRT native linkage for DiT acceleration (shared across platforms)
if(TRT_ENABLED)
    if(NOT CUDAToolkit_FOUND)
        message(FATAL_ERROR "[TRT] CUDA Toolkit headers/runtime are required for TensorRT builds")
    endif()
    target_compile_definitions(acestep-core PUBLIC HOT_STEP_TRT)
    target_compile_definitions(acestep-core PUBLIC
        HOT_STEP_TRT_VERSION_MAJOR=${TRT_VERSION_MAJOR}
        HOT_STEP_TRT_VERSION_MINOR=${TRT_VERSION_MINOR}
        HOT_STEP_TRT_VERSION_PATCH=${TRT_VERSION_PATCH})
    target_include_directories(acestep-core PUBLIC "${_TRT_INC_DIR}" ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(acestep-core PUBLIC "${_TRT_NVINFER_LIB}" "${_TRT_NVONNXPARSER_LIB}" CUDA::cudart)
    message(STATUS "[TRT] DiT TRT acceleration: ENABLED")

    # ─────────────────────────────────────────────────────────────────────────
    # HOT-Step custom TRT plugin library (libhotstep_plugins.so / hotstep_plugins.dll)
    #
    # Builds the ConvRotInt8Linear IPluginV3 plugin that fuses the w8a8 +
    # ConvRot path (online activation rotation + per-row INT8 quant + INT8 × INT8
    # matmul + dequant + bias) into a single TRT layer.
    #
    # The plugin .so is loaded by the engine runtime (dit-trt.h:
    # dit_trt_load_engine) via dlopen/LoadLibrary before any w8a8 engine is
    # deserialized. The static REGISTER_TENSORRT_PLUGIN macro in the .cpp
    # file registers the creator with TRT's plugin registry under the
    # "hotstep" namespace.
    #
    # Build-time-only dependency: the plugin .so is NOT linked into
    # acestep-core (it has its own TRT/CUDA symbols that would conflict
    # with the engine's). It's loaded purely via dlopen.
    # ─────────────────────────────────────────────────────────────────────────
    enable_language(CUDA)
    add_library(hotstep_plugins SHARED
        src/plugins/convrot_int8_linear_plugin.cpp
        src/plugins/convrot_int8_linear_kernel.cu
    )
    target_compile_definitions(hotstep_plugins PRIVATE HOT_STEP_TRT)
    # Plugin diagnostic logging is OFF by default for zero runtime overhead.
    # Enable at build time with: cmake -DHOTSTEP_PLUGIN_DEBUG=1 ...
    # This compiles in the full RAM-buffered logging subsystem (controlled
    # at runtime via the HOTSTEP_PLUGIN_DEBUG env var). Without it, all
    # log() calls are compiled to no-ops.
    option(HOTSTEP_PLUGIN_DEBUG "Enable HotStep plugin diagnostic logging" OFF)
    if(HOTSTEP_PLUGIN_DEBUG)
        target_compile_definitions(hotstep_plugins PRIVATE HOTSTEP_PLUGIN_DEBUG)
    endif()
    # On Windows, export hotstep_register_plugins via __declspec(dllexport)
    # (defined in the plugin .cpp via HOTSTEP_PLUGIN_EXPORT). On Linux/macOS
    # the default visibility is sufficient.
    if(WIN32)
        # Required for MSVC: resolve CUDA device code symbols from .cu objects
        # into the DLL. Without this the linker fails with unresolved externals.
        set_target_properties(hotstep_plugins PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
    target_include_directories(hotstep_plugins PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${_TRT_INC_DIR}"
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(hotstep_plugins PRIVATE
        "${_TRT_NVINFER_LIB}"
        CUDA::cudart
        CUDA::cuda_driver
    )
    # Install the shared library next to the engine binary so dlopen/LoadLibrary
    # finds it via $ORIGIN (Linux/macOS) or same-directory search (Windows).
    set_target_properties(hotstep_plugins PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    message(STATUS "[TRT] HOT-Step plugin library: $<TARGET_FILE_NAME:hotstep_plugins> (ConvRotInt8Linear)")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# TRT-LLM Executor (C++ Executor API for LM inference)
# ─────────────────────────────────────────────────────────────────────────────
# Pre-built TRT-LLM SDK (tensorrt_llm.dll + plugin DLL).
# Auto-detect from engine/trtllm-libs/.
# This is SEPARATE from HOT_STEP_TRT (raw NvInfer for DiT). Both can coexist.
#
# STATUS: DISABLED (2026-06-02). Native Windows TRT-LLM is not viable:
#   - FMHA/XQA cubin embedding requires GCC inline asm (INCBIN), impossible on MSVC
#   - Docker-built engines have Linux platform tags, can't deserialize on Windows
#   - TRT version mismatch between Docker (10.14) and Windows SDK (10.16)
#   - ONNX-rebuilt engines lack TRT-LLM tensor bindings (kv_cache_block_offsets etc.)
# The code remains intact behind #ifdef HOT_STEP_TRTLLM for future WSL2 or
# cross-platform engine support. To re-enable, set HOT_STEP_TRTLLM_ENABLE=ON.
option(HOT_STEP_TRTLLM_ENABLE "Enable TRT-LLM Executor (currently broken on native Windows)" OFF)
set(TRTLLM_ENABLED FALSE)
set(_TRTLLM_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtllm-libs")
set(_TRTLLM_INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/trtllm-include")
if(HOT_STEP_TRTLLM_ENABLE AND WIN32
         AND EXISTS "${_TRTLLM_LIBS_DIR}/tensorrt_llm.lib"
         AND EXISTS "${_TRTLLM_INC_DIR}/tensorrt_llm/executor/executor.h")
    set(TRTLLM_ENABLED TRUE)
    message(STATUS "[TRT-LLM] Found at ${_TRTLLM_LIBS_DIR}")
    target_compile_definitions(acestep-core PUBLIC HOT_STEP_TRTLLM)
    target_include_directories(acestep-core PUBLIC "${_TRTLLM_INC_DIR}")
    target_link_directories(acestep-core PUBLIC "${_TRTLLM_LIBS_DIR}")
    target_link_libraries(acestep-core PUBLIC tensorrt_llm)
    # NOTE: nvinfer_plugin_tensorrt_llm is loaded dynamically via LoadLibrary
    # in lm-trtllm.h to avoid pulling its dependency chain at process startup.
    message(STATUS "[TRT-LLM] LM Executor: ENABLED")
else()
    if(HOT_STEP_TRTLLM_ENABLE)
        message(STATUS "[TRT-LLM] Not found (set engine/trtllm-libs/ + trtllm-include/ to enable)")
    else()
        message(STATUS "[TRT-LLM] DISABLED (set -DHOT_STEP_TRTLLM_ENABLE=ON to re-enable)")
    endif()
endif()


# ace-synth: full pipeline (text-enc + cond + dit + vae + wav)
add_executable(ace-synth tools/ace-synth.cpp)
target_link_libraries(ace-synth PRIVATE acestep-core)
link_ggml_backends(ace-synth)
# CUDA runtime for TRT DiT path (cudaMalloc, cudaMemcpy, etc.)
if(CUDAToolkit_FOUND)
    if(GGML_STATIC)
        target_link_libraries(ace-synth PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(ace-synth PRIVATE CUDA::cudart)
    endif()
endif()

# test-prune-bundle: behavior test for store_prune_bundle_except (TRT bundle group-evict).
# Recompiles model-store.cpp with HOT_STEP_MODEL_STORE_TEST so the test seam
# (store_test_install_synthetic / store_test_has_entry) is available. This object
# satisfies every model-store symbol, so the production copy in acestep-core is
# not pulled from the archive; loader symbols (qw3lm_load, etc.) still resolve
# from acestep-core. Not wired into any default build; built/run on demand.
add_executable(test-prune-bundle tests/test-prune-bundle.cpp src/model-store.cpp)
target_compile_definitions(test-prune-bundle PRIVATE HOT_STEP_MODEL_STORE_TEST)
target_link_libraries(test-prune-bundle PRIVATE acestep-core)
link_ggml_backends(test-prune-bundle)
if(CUDAToolkit_FOUND)
    if(GGML_STATIC)
        target_link_libraries(test-prune-bundle PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(test-prune-bundle PRIVATE CUDA::cudart)
    endif()
endif()

# test-dit-trt-store: store-side ownership test for the MODEL_DIT_TRT model-store
# integration. Recompiles model-store.cpp with HOT_STEP_MODEL_STORE_TEST so the
# synthetic-entry seam is available, and links acestep-core (which propagates
# HOT_STEP_TRT PUBLIC, activating the MODEL_DIT_TRT eviction logic in this
# object). No real DiT engine is loaded — a live store-managed TRT-DiT load
# requires a real DiT bundle. Guarded on TRT_ENABLED so the eviction behavior
# is deterministic. Built/run on demand.
if(TRT_ENABLED)
    add_executable(test-dit-trt-store tests/test-dit-trt-store.cpp src/model-store.cpp)
    target_compile_definitions(test-dit-trt-store PRIVATE HOT_STEP_MODEL_STORE_TEST)
    target_link_libraries(test-dit-trt-store PRIVATE acestep-core)
    link_ggml_backends(test-dit-trt-store)
    if(CUDAToolkit_FOUND)
        if(GGML_STATIC)
            target_link_libraries(test-dit-trt-store PRIVATE CUDA::cudart_static)
        else()
            target_link_libraries(test-dit-trt-store PRIVATE CUDA::cudart)
        endif()
    endif()
endif()

# test-trt-bundle-manifest: all-three-or-nothing fail-closed gate test for the
# native TRT bundle manifest reader (trt-bundle-manifest.h). Header-only gate;
# links acestep-core for the yyjson parser symbols. Not TRT-gated (the manifest
# reader has no TRT/CUDA dependency). Built/run on demand; not in default build.
add_executable(test-trt-bundle-manifest tests/test-trt-bundle-manifest.cpp)
target_link_libraries(test-trt-bundle-manifest PRIVATE acestep-core)
link_ggml_backends(test-trt-bundle-manifest)
if(CUDAToolkit_FOUND)
    if(GGML_STATIC)
        target_link_libraries(test-trt-bundle-manifest PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(test-trt-bundle-manifest PRIVATE CUDA::cudart)
    endif()
endif()

# test-convrot-int8-plugin: unit test for the ConvRotInt8Linear TRT plugin.
# Verifies the CUDA kernel math against NumPy reference test vectors and the
# plugin creator's registration with TRT's plugin registry. Only built when
# TRT is enabled (the plugin library must be on LD_LIBRARY_PATH / next to the
# exe at runtime so the static REGISTER_TENSORRT_PLUGIN can register the creator).
if(TRT_ENABLED)
    add_executable(test-convrot-int8-plugin tests/test-convrot-int8-plugin.cpp)
    target_compile_definitions(test-convrot-int8-plugin PRIVATE HOT_STEP_TRT)
    target_include_directories(test-convrot-int8-plugin PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${_TRT_INC_DIR}"
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(test-convrot-int8-plugin PRIVATE
        hotstep_plugins
        "${_TRT_NVINFER_LIB}"
        CUDA::cudart
    )
    # On Windows, copy the plugin DLL and TRT runtime DLLs next to the test
    # executable so they can be found at runtime.
    if(WIN32)
        add_custom_command(TARGET test-convrot-int8-plugin POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:hotstep_plugins>"
                "$<TARGET_FILE_DIR:test-convrot-int8-plugin>"
            COMMENT "Copying hotstep_plugins DLL for test"
        )
        foreach(_trt_dll nvinfer_11.dll nvinfer.dll nvonnxparser_11.dll nvonnxparser.dll)
            if(EXISTS "${_TRT_BIN_DIR}/${_trt_dll}")
                add_custom_command(TARGET test-convrot-int8-plugin POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_TRT_BIN_DIR}/${_trt_dll}"
                        "$<TARGET_FILE_DIR:test-convrot-int8-plugin>/${_trt_dll}"
                    COMMENT "Copying ${_trt_dll} for test"
                )
            endif()
        endforeach()
    endif()
endif()

# test-text-enc-trt: parity + evict/reactivate test for the native TRT text
# encoder (text-enc-trt.h). Links acestep-core, which propagates HOT_STEP_TRT,
# the TRT include dir, and nvinfer (PUBLIC). Only built when TRT is enabled.
# Run on demand against the gitignored .enc-build fixtures; not in default build.
if(TRT_ENABLED)
    add_executable(test-text-enc-trt tests/test-text-enc-trt.cpp)
    target_link_libraries(test-text-enc-trt PRIVATE acestep-core)
    link_ggml_backends(test-text-enc-trt)
    if(CUDAToolkit_FOUND)
        if(GGML_STATIC)
            target_link_libraries(test-text-enc-trt PRIVATE CUDA::cudart_static)
        else()
            target_link_libraries(test-text-enc-trt PRIVATE CUDA::cudart)
        endif()
    endif()
endif()

# test-cond-enc-trt: parity + evict/reactivate test for the native TRT condition
# encoder (cond-enc-trt.h). Same structure as test-text-enc-trt. Links
# acestep-core (propagates HOT_STEP_TRT + the TRT include dir + nvinfer PUBLIC).
# Only built when TRT is enabled. Run on demand against the gitignored .enc-build
# fixtures; not in the default build.
if(TRT_ENABLED)
    add_executable(test-cond-enc-trt tests/test-cond-enc-trt.cpp)
    target_link_libraries(test-cond-enc-trt PRIVATE acestep-core)
    link_ggml_backends(test-cond-enc-trt)
    if(CUDAToolkit_FOUND)
        if(GGML_STATIC)
            target_link_libraries(test-cond-enc-trt PRIVATE CUDA::cudart_static)
        else()
            target_link_libraries(test-cond-enc-trt PRIVATE CUDA::cudart)
        endif()
    endif()
endif()

# test-lyric-embed-lut: parity oracle for the TRT-path lyric-embedding lookup
# (lyric-embed-lut.h) vs the GGML qwen3_embed_lookup against the same embed
# table. Links acestep-core (propagates HOT_STEP_TRT + the GGML text encoder).
# Only built when TRT is enabled. Run on demand against the on-disk bundle +
# GGUF text encoder; not in the default build.
if(TRT_ENABLED)
    add_executable(test-lyric-embed-lut tests/test-lyric-embed-lut.cpp)
    target_link_libraries(test-lyric-embed-lut PRIVATE acestep-core)
    link_ggml_backends(test-lyric-embed-lut)
    if(CUDAToolkit_FOUND)
        if(GGML_STATIC)
            target_link_libraries(test-lyric-embed-lut PRIVATE CUDA::cudart_static)
        else()
            target_link_libraries(test-lyric-embed-lut PRIVATE CUDA::cudart)
        endif()
    endif()
endif()

# ace-lm: LLM inference (CoT + audio codes)
add_executable(ace-lm tools/ace-lm.cpp)
target_link_libraries(ace-lm PRIVATE acestep-core)
link_ggml_backends(ace-lm)
# CUDA runtime for TRT LM path (cudaMalloc, cudaMemcpy, etc.)
if(CUDAToolkit_FOUND)
    if(GGML_STATIC)
        target_link_libraries(ace-lm PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(ace-lm PRIVATE CUDA::cudart)
    endif()
endif()

if(SUPERSEP_ENABLED)
    add_executable(ace-ort-cache tools/ace-ort-cache.cpp)
    target_include_directories(ace-ort-cache PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        "${ORT_ROOT}/include"
    )
    target_link_directories(ace-ort-cache PRIVATE "${ORT_ROOT}/lib")
    target_link_libraries(ace-ort-cache PRIVATE onnxruntime Threads::Threads)
    if(CUDAToolkit_FOUND)
        target_compile_definitions(ace-ort-cache PRIVATE GGML_USE_CUDA)
        if(GGML_STATIC)
            target_link_libraries(ace-ort-cache PRIVATE CUDA::cudart_static)
        else()
            target_link_libraries(ace-ort-cache PRIVATE CUDA::cudart)
        endif()
    endif()
    add_dependencies(ace-ort-cache version)
endif()

# webui: convert tools/webui/public/index.html.gz to a C header for embedding.
# the .gz is committed to git so the C++ build works without npm.
# to update: cd tools/webui && npm install && npm run build, then rebuild ace-server.
set(WEBUI_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/tools/public/index.html.gz")
set(WEBUI_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/index.html.gz.hpp")
add_custom_command(
    OUTPUT "${WEBUI_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${WEBUI_INPUT}" "-DOUTPUT=${WEBUI_OUTPUT}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/xxd.cmake"
    DEPENDS "${WEBUI_INPUT}"
    COMMENT "Embedding webui into index.html.gz.hpp"
)
set_source_files_properties(${WEBUI_OUTPUT} PROPERTIES GENERATED TRUE)

# hot-step-server: HOT-Step HTTP server (LM + synth endpoints + embedded webui)
# NOTE: upstream ace-server.cpp is kept as reference but NOT compiled.
#       Our server binary is hot-step-server.cpp with extension layer support.
add_executable(ace-server tools/hot-step-server.cpp ${WEBUI_OUTPUT})
target_include_directories(ace-server PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ace-server PRIVATE acestep-core httplib supersep)
link_ggml_backends(ace-server)
# CUDA runtime for GET /vram (cudaMemGetInfo)
if(NOT CUDAToolkit_FOUND)
    find_package(CUDAToolkit QUIET)
endif()
if(CUDAToolkit_FOUND)
    target_compile_definitions(ace-server PRIVATE GGML_USE_CUDA)
    if(GGML_STATIC)
        target_link_libraries(ace-server PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(ace-server PRIVATE CUDA::cudart)
    endif()
    # Copy CUDA runtime DLL to build dir (needed for portable release builds —
    # end users don't have the CUDA Toolkit, so cudart64_*.dll must ship with the binary)
    if(WIN32)
        get_target_property(_CUDART_LOC CUDA::cudart IMPORTED_LOCATION)
        if(_CUDART_LOC AND _CUDART_LOC MATCHES "\\.dll$")
            add_custom_command(TARGET ace-server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_CUDART_LOC}"
                    "$<TARGET_FILE_DIR:ace-server>"
                COMMENT "Copying cudart DLL for portable release"
            )
        endif()
    endif()
    # Write CUDA version marker next to ace-server (server reads this to
    # select the correct runtime DLLs — cuBLAS/cudart/cuDNN for 12 vs 13)
    file(GENERATE OUTPUT "$<TARGET_FILE_DIR:ace-server>/.cuda-version"
         CONTENT "${CUDAToolkit_VERSION_MAJOR}")
endif()

# Copy ONNX Runtime DLLs next to binaries that can load ORT models.
if(SUPERSEP_ENABLED AND WIN32)
    set(_ORT_DLL_DIR "${ORT_ROOT}/lib")
    foreach(_ort_target ace-synth ace-server ace-ort-cache)
        foreach(_dll onnxruntime.dll onnxruntime_providers_shared.dll onnxruntime_providers_cuda.dll onnxruntime_providers_tensorrt.dll)
            if(EXISTS "${_ORT_DLL_DIR}/${_dll}")
                add_custom_command(TARGET ${_ort_target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_ORT_DLL_DIR}/${_dll}"
                        "$<TARGET_FILE_DIR:${_ort_target}>/${_dll}"
                    COMMENT "Copying ${_dll} for ${_ort_target}"
                )
            endif()
        endforeach()
    endforeach()
endif()

# Copy ONNX Runtime dylibs next to ace-server at build time (macOS)
if(SUPERSEP_ENABLED AND APPLE)
    set(_ORT_LIB_DIR "${ORT_ROOT}/lib")
    file(GLOB _ORT_DYLIBS "${_ORT_LIB_DIR}/libonnxruntime*.dylib")
    foreach(_dylib ${_ORT_DYLIBS})
        add_custom_command(TARGET ace-server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dylib}" "$<TARGET_FILE_DIR:ace-server>"
            COMMENT "Copying ${_dylib}"
        )
    endforeach()
endif()

# Copy ONNX Runtime shared libs next to ace-server at build time (Linux)
if(SUPERSEP_ENABLED AND UNIX AND NOT APPLE)
    set(_ORT_LIB_DIR "${ORT_ROOT}/lib")
    file(GLOB _ORT_SOLIBS "${_ORT_LIB_DIR}/libonnxruntime*.so*")
    foreach(_solib ${_ORT_SOLIBS})
        add_custom_command(TARGET ace-server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_solib}" "$<TARGET_FILE_DIR:ace-server>"
            COMMENT "Copying ${_solib}"
        )
    endforeach()
endif()

# Copy TRT runtime DLLs + hotstep_plugins DLL next to ace-server on Windows.
# TRT DLLs live in ${TRT_ROOT}/bin/; the plugin DLL is a build artifact.
# Without this, LoadLibrary("hotstep_plugins.dll") in dit-trt.h fails because
# Windows can't resolve transitive DLL deps (nvinfer_11.dll, cudart64_*.dll).
if(TRT_ENABLED AND WIN32)
    # Copy TRT runtime DLLs from the SDK bin/ directory
    foreach(_trt_dll nvinfer_11.dll nvinfer.dll nvonnxparser_11.dll nvonnxparser.dll)
        if(EXISTS "${_TRT_BIN_DIR}/${_trt_dll}")
            add_custom_command(TARGET ace-server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_TRT_BIN_DIR}/${_trt_dll}"
                    "$<TARGET_FILE_DIR:ace-server>/${_trt_dll}"
                COMMENT "Copying ${_trt_dll} for ace-server"
            )
        endif()
    endforeach()
    # Copy the hotstep_plugins DLL (built by the hotstep_plugins target)
    add_custom_command(TARGET ace-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:hotstep_plugins>"
            "$<TARGET_FILE_DIR:ace-server>/"
        COMMENT "Copying hotstep_plugins DLL for ace-server"
    )
endif()

add_executable(ace-understand tools/ace-understand.cpp)
target_link_libraries(ace-understand PRIVATE acestep-core)
link_ggml_backends(ace-understand)

# quantize: GGUF requantizer (BF16 -> K-quants)
add_executable(quantize tools/quantize.cpp)
link_ggml_backends(quantize)

# neural-codec: Oobleck VAE neural audio codec (encode/decode WAV <-> latent)
add_executable(neural-codec tools/neural-codec.cpp)
link_ggml_backends(neural-codec)

# mp3-codec: MP3 encoder/decoder (standalone, no ggml needed)
# The mp3/ headers are header-only and usable by ace-synth too via #include "mp3/mp3enc.h"
add_executable(mp3-codec tools/mp3-codec.cpp)
target_include_directories(mp3-codec PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(mp3-codec version)
if(MSVC)
    target_compile_options(mp3-codec PRIVATE /W4 /wd4100 /wd4505)
else()
    target_compile_options(mp3-codec PRIVATE -Wall -Wextra -Wconversion
                          -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    target_link_libraries(mp3-codec PRIVATE m)
endif()
target_link_libraries(mp3-codec PRIVATE Threads::Threads)

# mastering: reference-based audio mastering (standalone, no ggml needed)
# Implements the matchering algorithm using pocketfft for FFT.
add_executable(mastering tools/mastering.cpp)
target_include_directories(mastering PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/pocketfft
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_dependencies(mastering version)
if(MSVC)
    target_compile_options(mastering PRIVATE /W4 /wd4100 /wd4505 /wd4244 /wd4267)
else()
    target_compile_options(mastering PRIVATE -Wall -Wextra -Wconversion
                          -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    target_link_libraries(mastering PRIVATE m)
endif()
target_link_libraries(mastering PRIVATE Threads::Threads)

# ─────────────────────────────────────────────────────────────────────────────
# VST3 Hosting Library + vst-host tool
# ─────────────────────────────────────────────────────────────────────────────
# We compile only the necessary VST3 SDK source files for hosting (loading
# plugins, processing audio, state save/restore). We do NOT use the SDK's
# own CMakeLists.txt because it requires cmake 3.25+ and pulls in the full
# build system including plugin examples and VSTGUI.

set(VST3SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/vst3sdk")

# Static library: vst3-hosting
# Platform-specific hosting sources:
#   Windows: module_win32.cpp + threadchecker_win32.cpp
#   macOS:   module_mac.mm (ARC) + threadchecker_mac.mm
#   Linux:   module_linux.cpp + threadchecker_linux.cpp
if(APPLE)
    set(VST3_PLATFORM_SOURCES
        ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module_mac.mm
        ${VST3SDK_ROOT}/public.sdk/source/common/threadchecker_mac.mm
    )
    # module_mac.mm requires Objective-C ARC
    set_source_files_properties(
        ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module_mac.mm
        ${VST3SDK_ROOT}/public.sdk/source/common/threadchecker_mac.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
elseif(WIN32)
    set(VST3_PLATFORM_SOURCES
        ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module_win32.cpp
        ${VST3SDK_ROOT}/public.sdk/source/common/threadchecker_win32.cpp
    )
else()
    set(VST3_PLATFORM_SOURCES
        ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module_linux.cpp
        ${VST3SDK_ROOT}/public.sdk/source/common/threadchecker_linux.cpp
    )
endif()

add_library(vst3-hosting STATIC
    # Base library
    ${VST3SDK_ROOT}/base/source/baseiids.cpp
    ${VST3SDK_ROOT}/base/source/fobject.cpp
    ${VST3SDK_ROOT}/base/source/fdebug.cpp
    ${VST3SDK_ROOT}/base/source/fstreamer.cpp
    ${VST3SDK_ROOT}/base/source/fbuffer.cpp
    ${VST3SDK_ROOT}/base/source/updatehandler.cpp
    ${VST3SDK_ROOT}/base/source/timer.cpp
    ${VST3SDK_ROOT}/base/source/fstring.cpp

    # Pluginterfaces
    ${VST3SDK_ROOT}/pluginterfaces/base/funknown.cpp
    ${VST3SDK_ROOT}/pluginterfaces/base/ustring.cpp
    ${VST3SDK_ROOT}/pluginterfaces/base/coreiids.cpp
    ${VST3SDK_ROOT}/pluginterfaces/base/conststringtable.cpp

    # Public SDK hosting (cross-platform + platform-specific)
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/module.cpp
    ${VST3_PLATFORM_SOURCES}
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/hostclasses.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/plugprovider.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/processdata.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/parameterchanges.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/pluginterfacesupport.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/eventlist.cpp
    ${VST3SDK_ROOT}/public.sdk/source/vst/hosting/connectionproxy.cpp

    # Public SDK common (memory streams for state I/O)
    ${VST3SDK_ROOT}/public.sdk/source/common/memorystream.cpp
    ${VST3SDK_ROOT}/public.sdk/source/common/commoniids.cpp
    ${VST3SDK_ROOT}/public.sdk/source/common/pluginview.cpp
    ${VST3SDK_ROOT}/public.sdk/source/common/commonstringconvert.cpp

    # VST interface ID definitions (all DEF_CLASS_IID symbols)
    ${VST3SDK_ROOT}/public.sdk/source/vst/vstinitiids.cpp

    # VST utilities
    ${VST3SDK_ROOT}/public.sdk/source/vst/utility/stringconvert.cpp
)

target_include_directories(vst3-hosting PUBLIC
    ${VST3SDK_ROOT}
    ${VST3SDK_ROOT}/pluginterfaces
    ${VST3SDK_ROOT}/public.sdk
)

# The SDK defines DEVELOPMENT=1 for debug builds
target_compile_definitions(vst3-hosting PRIVATE
    $<$<CONFIG:Debug>:DEVELOPMENT=1>
    $<$<CONFIG:Release>:RELEASE=1>
)

if(MSVC)
    # Suppress noisy SDK warnings
    target_compile_options(vst3-hosting PRIVATE /W0)
else()
    target_compile_options(vst3-hosting PRIVATE -w)
endif()

# macOS: VST3 hosting needs Cocoa + CoreFoundation frameworks
if(APPLE)
    find_library(COCOA_FRAMEWORK Cocoa)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
    target_link_libraries(vst3-hosting PRIVATE
        ${COCOA_FRAMEWORK}
        ${COREFOUNDATION_FRAMEWORK}
    )
endif()

# vst-host: standalone VST3 host tool (GUI + offline processing + chain)
add_executable(vst-host tools/vst-host.cpp)
target_include_directories(vst-host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${VST3SDK_ROOT}
)
target_link_libraries(vst-host PRIVATE vst3-hosting yyjson Threads::Threads)
if(MSVC)
    target_compile_options(vst-host PRIVATE /W4 /wd4100 /wd4505)
    target_compile_definitions(vst-host PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    target_compile_options(vst-host PRIVATE -Wall -Wextra -Wconversion
                          -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
endif()
add_dependencies(vst-host version)

# macOS: vst-host also needs Apple frameworks
if(APPLE)
    target_link_libraries(vst-host PRIVATE ${COCOA_FRAMEWORK} ${COREFOUNDATION_FRAMEWORK})
endif()
