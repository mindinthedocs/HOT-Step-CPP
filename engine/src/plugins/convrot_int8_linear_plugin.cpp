/*
 * convrot_int8_linear_plugin.cpp — IPluginV3 implementation for ConvRotInt8Linear.
 *
 * Runtime execution family:
 *
 *   All M, including M == 1:
 *     Launch a two-kernel Triton pipeline:
 *       1. kernel1_convrot_quant  — rotate + quantize activations into
 *          reusable workspace (X_q + X_scale)
 *       2. kernel2_gemm_dequant   — reuse that workspace across all N tiles
 *
 *   A previous dedicated fused M==1 Triton kernel was removed after profiling
 *   showed the two-kernel BK64/BM128/BN128 path is faster on the real M==1
 *   workload too. Keeping one execution family also shrinks the cubin header
 *   and removes M-dependent runtime dispatch state.
 *
 * Rotation design (butterfly, not dense H-matrix matmul):
 *   The Hadamard rotation H_{GROUP_SIZE} is decomposed into log_4(GROUP_SIZE)
 *   successive H_4 butterfly stages, all performed in registers. This avoids
 *   materializing a dense [GROUP_SIZE x GROUP_SIZE] H matrix in shared memory.
 *
 * Cubin dispatch:
 *   At onShapeChange(), the plugin verifies the concrete runtime M/K/N and
 *   ensures the two compiled Triton cubins for the serialized group size, bias,
 *   and dtype configuration are loaded. Runtime M no longer changes the cubin
 *   family; M==1 uses the same two-kernel path as larger batches.
 */

#ifdef HOT_STEP_TRT

#include "convrot_int8_linear_plugin.h"
#include "convrot_int8_linear_kernel.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CONVROT_INT8_KERNEL_CUBIN_HEADER_AVAILABLE
#define CONVROT_INT8_KERNEL_CUBIN_HEADER_AVAILABLE __has_include("assets/convrot_int8_kernel_cubin.h")
#endif
#if CONVROT_INT8_KERNEL_CUBIN_HEADER_AVAILABLE
#include "assets/convrot_int8_kernel_cubin.h"
#endif


#if CONVROT_INT8_KERNEL_CUBIN_HEADER_AVAILABLE
// ── Cubin header version guard ─────────────────────────────────────────────
// The cubin header must declare CONVROT_INT8_CUBIN_HEADER_VERSION >= 4.
// Version 3 added per-cubin tile dimensions; version 4 additionally reflects
// the intentionally smaller cubin inventory: G256 only (plus optional G0 sentinel), and no specialized M1
// cubins. Referencing removed G0/G256/M1 symbols here would make the plugin
// silently depend on an oversized stale generated header.
//
// Version 1: original (no _SHARED, no _BLOCK_M/_BLOCK_N)
// Version 2: added _SHARED constants (shared-mem pre-flight check)
// Version 3: added _BLOCK_M / _BLOCK_N (per-cubin tile selection)
// Version 4: G64-only two-kernel inventory; M==1 uses the two-kernel path
// Version 5: generated launch descriptors/stubs.  The C++ plugin no longer
//            reconstructs Triton launch geometry from loose macros.
// Version 6: GROUP_SIZE=256, TC-based rotation (H_16⊗H_16), decoupled BLOCK_K,
//            persistent grid-stride launch, transposed X_scale layout.
#ifndef CONVROT_INT8_CUBIN_HEADER_VERSION
#  error "Cubin header is missing CONVROT_INT8_CUBIN_HEADER_VERSION. Re-run tools/onnx-export/extract_jit_cubins_autotune.py to regenerate engine/src/plugins/assets/convrot_int8_kernel_cubin.h."
#elif CONVROT_INT8_CUBIN_HEADER_VERSION < 6
#  error "Cubin header version >= 6 required (GROUP_SIZE=256, TC rotation, persistent launch). Re-run tools/onnx-export/extract_jit_cubins_autotune.py to regenerate engine/src/plugins/assets/convrot_int8_kernel_cubin.h."
#endif
#ifndef CONVROT_INT8_HAS_GENERATED_LAUNCH_STUBS
#  error "Cubin header is missing generated launch stubs/descriptors. Re-run tools/onnx-export/extract_jit_cubins_autotune.py to regenerate engine/src/plugins/assets/convrot_int8_kernel_cubin.h."
#endif
#else
// The generated cubin header is intentionally not committed because it is very
// large. This branch still compiles for host-side checks without it; runtime
// cubin lookup helpers below return empty payloads until the header is generated
// with tools/onnx-export/extract_jit_cubins_autotune.py.
#endif
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#if !CONVROT_INT8_KERNEL_CUBIN_HEADER_AVAILABLE
namespace hotstep::convrot_int8_generated {
enum class ConvRotKernelStage : int32_t { kQuant = 1, kGemm = 2 };
struct ConvRotCubinDesc {
    char const* logical_name{};
    unsigned char const* data{};
    size_t size{};
    char const* function_name{};
    ConvRotKernelStage stage{};
    int32_t group_size{};
    bool has_bias{};
    int32_t input_dtype_id{};
    int32_t output_dtype_id{};
    int32_t block_m{1};
    int32_t block_n{1};
    int32_t block_k{1};
    int32_t group_m{1};
    uint32_t block_x{1};
    uint32_t block_y{1};
    uint32_t block_z{1};
    size_t shared_bytes{};
    int32_t num_warps{};
    int32_t num_stages{};
    int32_t maxnreg{};
};
inline constexpr ConvRotCubinDesc const* findConvRotCubin(
    ConvRotKernelStage, int32_t, bool, int32_t, int32_t) { return nullptr; }
inline uint32_t ceilDivU32(int32_t x, int32_t y) {
    return static_cast<uint32_t>((x + y - 1) / y);
}
inline CUresult launchConvRotQuant(ConvRotCubinDesc const&, CUfunction, CUstream,
    void const*, void*, void*, int32_t, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t) { return CUDA_ERROR_INVALID_VALUE; }
inline CUresult launchConvRotGemm(ConvRotCubinDesc const&, CUfunction, CUstream,
    void*, void*, int8_t const*, float const*, void const*, void*, int32_t,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t) { return CUDA_ERROR_INVALID_VALUE; }
} // namespace hotstep::convrot_int8_generated
#endif

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN 1
#  define NOMINMAX 1
#  include <windows.h>  // OutputDebugStringW, MultiByteToWideChar
#endif

// ── Runtime CUDA arch detection (global anonymous namespace) ───────────────
//
// queryDeviceComputeCapability() is declared in a *global* anonymous
// namespace (not inside `namespace hotstep`) because it is called from both:
//   * `namespace hotstep` members (initTriton, plugin class methods), and
//   * the `extern "C"` `hotstep_register_plugins()` entry point at the
//     bottom of this file, which lives at global scope.
//
// An anonymous-namespace member declared inside `namespace hotstep` is only
// findable by unqualified lookup from *within* `namespace hotstep`. The
// extern "C" entry point can't see it, which previously caused MSVC to emit
// "identifier not found" at the DLL-load log site. A global anonymous
// namespace is visible from every namespace in the TU, so both call sites
// resolve correctly.
//
// The plugin's 16-bit I/O boundary dtype is selected at cubin-generation
// time (BF16 on sm >= 80, FP16 on sm_75). At runtime we query the active
// device's compute capability once and stash it in a function-local static
// so we can:
//   1. Log it at plugin-DLL load time for diagnostic visibility.
//   2. Warn if the engine's serialized boundary dtype doesn't match the
//      GPU's arch (e.g. an FP16 engine loaded on sm_80+ still works, but
//      the user should re-export as BF16 for the wider exponent range).
//
// Returns the compute capability as major*10 + minor (e.g. 86 for sm_86),
// or 0 if the device query failed (we treat 0 as "unknown arch" and skip
// the arch/dtype mismatch warning rather than failing).
namespace {
int32_t queryDeviceComputeCapability() {
    static int32_t cached = -1;
    static std::once_flag flag;
    std::call_once(flag, []() {
        int32_t dev = 0;
        if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) {
            // No active CUDA context — try cudaGetDevice as a fallback.
            if (cudaGetDevice(&dev) != cudaSuccess) {
                cached = 0;
                return;
            }
        }
        int major = 0, minor = 0;
        CUresult const rc_major = cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        CUresult const rc_minor = cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        if (rc_major != CUDA_SUCCESS || rc_minor != CUDA_SUCCESS ||
            major <= 0 || minor < 0) {
            // Fall back to the runtime API if the driver API query failed.
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
                major = prop.major;
                minor = prop.minor;
            } else {
                cached = 0;
                return;
            }
        }
        cached = static_cast<int32_t>(major) * 10 + static_cast<int32_t>(minor);
    });
    return cached;
}
}  // namespace

namespace hotstep {

// ── Diagnostic logging ──────────────────────────────────────────────────────
//
// Compile-time gating: the entire diag subsystem is compiled in ONLY when
// HOTSTEP_PLUGIN_DEBUG is defined at build time (e.g. via
//   cmake -DHOTSTEP_PLUGIN_DEBUG=1 ...
// or by adding `target_compile_definitions(hotstep_plugins PRIVATE HOTSTEP_PLUGIN_DEBUG)`
// to CMakeLists.txt for debug builds).
//
// When HOTSTEP_PLUGIN_DEBUG is NOT defined, log() and flush() are inline
// no-ops. The compiler eliminates the call site entirely — including
// argument evaluation — so there is ZERO runtime overhead from the dozens of
// log calls scattered through onShapeChange() / enqueue() / initTriton().
// This is critical for production throughput: the DiT inference loop calls
// these methods 359 times per step, and even ~100ns of per-call overhead
// from runtime env-var checks would add up to ~36us per step (negligible
// here, but the principle matters for hot paths).
//
// When HOTSTEP_PLUGIN_DEBUG IS defined, the full RAM-buffered logging
// subsystem is compiled in, with runtime control via the HOTSTEP_PLUGIN_DEBUG
// environment variable (set to "1", "immediate", or "stdout"). See the
// long comment below for details.
//
// TRT 11's PluginV3Runner::execute assertion at line 252 fires whenever any
// plugin method returns non-success. TRT swallows plugin stderr, leaving the
// user with only the bare assertion. To work around this, when diagnostics
// are enabled, every message is written to a RAM buffer with three sinks:
//
//   1. A RAM buffer (fast — pure memcpy under mutex, ~100ns/call)
//   2. stderr (cheap, may be swallowed by TRT)
//   3. OutputDebugStringW on Windows (kernel call, ~1us, always visible in VS)
//
// The RAM buffer is flushed to disk (hotstep_plugin_diag.log) only when:
//   - The buffer crosses kFlushThreshold (default 1 MB)
//   - An error-path message is logged (contains FAILURE/REJECT/failed)
//   - The DLL is unloaded (static destructor)
//
// This keeps per-call cost at ~5-10us (mostly OutputDebugStringW) instead of
// the ~500us of file I/O per call. For a DiT inference pass with ~100 plugin
// invocations, that's ~1ms total overhead — negligible.
//
// Performance gating:
//   Diagnostics are OFF by default — zero overhead. Enable by setting
//   HOTSTEP_PLUGIN_DEBUG=1 in the environment before loading the plugin DLL.
//   The enabled() check is a single atomic bool read (~1ns).
//
//   Set HOTSTEP_PLUGIN_DEBUG=immediate to force a disk flush on EVERY log
//   call — useful for debugging crashes that prevent the static destructor
//   from running (e.g., TRT assertion aborts before DLL unload).
namespace diag {

#ifdef HOTSTEP_PLUGIN_DEBUG
// ──────────────────────────────────────────────────────────────────────────
// Full diagnostic logging implementation (compiled only when
// HOTSTEP_PLUGIN_DEBUG is defined at build time).
// ──────────────────────────────────────────────────────────────────────────

constexpr size_t kFlushThreshold = 1ull * 1024ull * 1024ull;  // 1 MB

std::mutex& logMutex() {
    static std::mutex m;
    return m;
}

std::string logPath() {
    char const* env = std::getenv("HOTSTEP_PLUGIN_LOG");
    if (env && env[0] != '\0') return std::string(env);
    // Default: hotstep_plugin_diag.log in the host process's CWD.
    // Set HOTSTEP_PLUGIN_LOG to redirect (e.g., to share the host's main log
    // by pointing this at the same path the host process logs to).
    return std::string("hotstep_plugin_diag.log");
}

// If true, also emit every log line to stdout (not just stderr). Useful when
// the host process redirects stdout to its main log file — plugin diagnostics
// will then be co-located with the rest of the application's output.
// Enable with HOTSTEP_PLUGIN_DEBUG=stdout  (implies enabled() too).
bool echoToStdout() {
    static bool cached = false;
    static std::once_flag flag;
    std::call_once(flag, []() {
        char const* env = std::getenv("HOTSTEP_PLUGIN_DEBUG");
        cached = (env != nullptr && std::strcmp(env, "stdout") == 0);
    });
    return cached;
}

bool enabled() {
    static bool cached = false;
    static std::once_flag flag;
    std::call_once(flag, []() {
        char const* env = std::getenv("HOTSTEP_PLUGIN_DEBUG");
        // Any non-empty value except "0" enables diagnostics. The special
        // values "immediate" and "stdout" select extra modes (immediate:
        // per-call disk flush; stdout: also echo to stdout).
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0');
    });
    return cached;
}

// "immediate" mode: flush to disk after every log call. Useful when the
// process might crash before the static destructor runs.
bool immediateMode() {
    static bool cached = false;
    static std::once_flag flag;
    std::call_once(flag, []() {
        char const* env = std::getenv("HOTSTEP_PLUGIN_DEBUG");
        cached = (env != nullptr && std::strcmp(env, "immediate") == 0);
    });
    return cached;
}

// RAM buffer — pre-allocated to 256 KB, grows as needed. Lives for the DLL's
// lifetime. All log calls append to this; flushes to disk are batched.
std::string& logBuffer() {
    static std::string buf;
    static std::once_flag flag;
    std::call_once(flag, []() {
        try { buf.reserve(256 * 1024); } catch (...) {}
    });
    return buf;
}

// Persistent log file stream — opened lazily on first flush, kept open.
std::ofstream& logStream() {
    static std::ofstream f;
    static std::once_flag flag;
    std::call_once(flag, []() {
        try { f.open(logPath(), std::ios::app); } catch (...) {}
    });
    return f;
}

// Internal: flush RAM buffer to disk. Caller MUST hold logMutex().
void flushToDiskLocked() {
    std::string& buf = logBuffer();
    if (buf.empty()) return;
    try {
        std::ofstream& f = logStream();
        if (f.is_open()) {
            f.write(buf.data(), static_cast<std::streamsize>(buf.size()));
            f.flush();
        }
    } catch (...) {
        // Swallow — logging must never throw.
    }
    buf.clear();
}

// Quick scan of a formatted message to decide whether to force a flush.
// Error-path messages (FAILURE/REJECT/failed) should reach disk immediately
// so they survive a process crash.
bool isErrorPath(char const* msg) {
    return std::strstr(msg, "FAILURE") != nullptr ||
           std::strstr(msg, "REJECT")  != nullptr ||
           std::strstr(msg, "failed")  != nullptr ||
           std::strstr(msg, "BUILD-TIME") != nullptr;
}

void log(char const* fmt, ...) {
    if (!enabled()) return;

    // 1. Format into stack buffer (no lock needed).
    char buf[2048];
    int len = 0;

    std::time_t now = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::gmtime(&now));
    len += std::snprintf(buf + len, sizeof(buf) - len, "[HotStep %s] ", ts);

    va_list args;
    va_start(args, fmt);
    len += std::vsnprintf(buf + len, sizeof(buf) - len, fmt, args);
    va_end(args);

    if (len > 0 && buf[len - 1] != '\n') {
        if (len < (int)sizeof(buf) - 1) {
            buf[len++] = '\n';
            buf[len] = '\0';
        }
    }

    bool const force_flush = isErrorPath(buf) || immediateMode();

    // 2. Append to RAM buffer + optionally flush, all under one mutex hold.
    bool need_flush = false;
    {
        std::lock_guard<std::mutex> lock(logMutex());

        // Append to RAM buffer (fast — string::append is amortized memcpy).
        try {
            logBuffer().append(buf, static_cast<size_t>(len));
        } catch (...) {
            // Swallow — logging must never throw.
        }

        if (force_flush || logBuffer().size() >= kFlushThreshold) {
            need_flush = true;
        }
    }

    // 3. stderr — cheap, may be swallowed by TRT.
    std::fputs(buf, stderr);
    std::fflush(stderr);

    // 3b. stdout — only in "stdout" mode. Use this when the host process
    //     redirects stdout to its main log file: plugin diagnostics will be
    //     co-located with the rest of the application's output, in order.
    if (echoToStdout()) {
        std::fputs(buf, stdout);
        std::fflush(stdout);
    }

    // 4. Windows debugger — cheap kernel call (~1us), always visible.
#if defined(_WIN32)
    int wlen = MultiByteToWideChar(CP_UTF8, 0, buf, -1, nullptr, 0);
    if (wlen > 0) {
        std::wstring wbuf(static_cast<size_t>(wlen), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, buf, -1, &wbuf[0], wlen);
        OutputDebugStringW(wbuf.c_str());
    }
#endif

    // 5. Flush to disk if threshold crossed, error path, or immediate mode.
    //    Done OUTSIDE the mutex-held section above only conceptually — we
    //    re-acquire the mutex here. This is fine because the work inside is
    //    just file I/O which is rare (once per 1MB or once per error).
    if (need_flush) {
        std::lock_guard<std::mutex> lock(logMutex());
        flushToDiskLocked();
    }
}

// Public flush API — exposed so callers can force a flush on demand (e.g.,
// from a signal handler or before a deliberate abort).
void flush() {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lock(logMutex());
    flushToDiskLocked();
}

// Static destructor — flushes the RAM buffer to disk at DLL unload. Ensures
// the last few KB of buffered messages (below kFlushThreshold) are not lost.
//
// STATIC DESTRUCTION ORDER: g_buffer_flusher is a namespace-scope static,
// constructed at DLL load. logBuffer()/logStream()/logMutex() are function-
// local statics, constructed on first log() call (which happens AFTER DLL
// load, during the first plugin method). So the destruction order is:
//   1. logStream(), logBuffer(), logMutex() destroyed (function-local, LIFO)
//   2. g_buffer_flusher destroyed (namespace-scope)
//
// This means g_buffer_flusher's destructor would access DEAD logStream()/
// logBuffer()/logMutex() objects — a use-after-free. To avoid this, the
// destructor does NOTHING. The RAM buffer is flushed incrementally during
// normal operation (on error paths, on 1 MB threshold, or in immediate mode),
// so the only data lost is the last few KB of buffered messages below the
// threshold — acceptable for a debug-only diagnostic log.
struct BufferFlusher {
    ~BufferFlusher() {
        // Intentionally empty — see comment above. Doing file I/O in a static
        // destructor is unsafe because function-local statics (logStream,
        // logBuffer, logMutex) may already be destroyed.
    }
};
BufferFlusher g_buffer_flusher;

#else  // !HOTSTEP_PLUGIN_DEBUG
// ──────────────────────────────────────────────────────────────────────────
// No-op stubs (compiled when HOTSTEP_PLUGIN_DEBUG is NOT defined).
//
// These are inline functions with empty bodies. Any modern compiler at -O1 or
// higher will eliminate the call site entirely, including all argument
// evaluation (since the arguments are unused and the functions have no side
// effects). This gives ZERO runtime overhead for production builds.
//
// logPath() returns an empty string — it's only used as an argument to log(),
// which is itself a no-op, so the return value is never observed. The stub
// exists solely so the symbol resolves at compile time (MSVC does name lookup
// before dead-code elimination, so even unused arguments must reference
// declared symbols).
// ──────────────────────────────────────────────────────────────────────────

inline void log(char const* /*fmt*/, ...) noexcept {}
inline void flush() noexcept {}
inline std::string logPath() noexcept { return std::string{}; }

#endif  // HOTSTEP_PLUGIN_DEBUG

}  // namespace diag

namespace {

unsigned int getDeviceMaxDynamicSharedMem() {
    static unsigned int cached = 0;
    static std::once_flag flag;
    std::call_once(flag, []() {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, dev);
        cached = static_cast<unsigned int>(prop.sharedMemPerBlockOptin);
        if (cached == 0) cached = 48 * 1024; // fallback
    });
    return cached;
}

// ── Global CUmodule cache ──────────────────────────────────────────────────
//
// PROBLEM: TRT 11's IPluginV3 lifecycle calls attachToContext() → clone() for
// EACH plugin instance at EACH inference step. The clone() method creates a
// new plugin with no Triton modules cached yet. When onShapeChange() runs on
// the cloned
// instance, it calls initTriton() → cuModuleLoadData(), which parses a ~2 MB
// ELF cubin and loads the SASS into GPU memory. With 359 ConvRotInt8Linear
// instances in the DiT engine, that's 359 cuModuleLoadData calls per step,
// each taking ~100–400 ms — totaling 36–144 s per step. This is the dominant
// overhead in the DiT inference loop (150 s/step observed vs. <2 s/step with
// the old cuBLASLt path).
//
// SOLUTION: Cache the loaded CUmodule globally, keyed by the cubin data
// pointer. The cubin data is a `constexpr unsigned char[]` array in .rodata,
// so its address is stable for the DLL's lifetime. Multiple plugin instances
// with the same (group_size, has_bias, dtype) config share the same
// CUmodule and CUfunction. The cache is:
//   - Thread-safe (guarded by a mutex, but only held during lookup/insert —
//     never during kernel execution).
//   - Lazy (modules are loaded on first use, not at DLL load).
//   - Leaked at process exit (NOT cleaned up by a static destructor — see
//     the comment above moduleCache() for why static destructor cleanup
//     causes access violations). The CUDA driver reclaims all GPU memory
//     when the process exits.
//
// With this cache, the first DiT step loads only the unique two-kernel cubins
// needed by the engine (for example, K1_G256_FP32IO plus the G256 FP32IO K2
// BIAS/NOBIAS variants when both bias modes occur).
// Subsequent steps hit the cache — zero cuModuleLoadData calls, zero
// cuModuleGetFunction calls, zero cuFuncSetAttribute calls. initTriton()
// becomes a single mutex + hash lookup (~200 ns).
//
// The cache key is the raw cubin data pointer (void const*). This is safe
// because:
//   1. All cubins are constexpr arrays in the cubin header — their addresses
//      are fixed for the DLL's lifetime.
//   2. Two plugins with the same config call selectCubinK1()/selectCubinK2()
//      which return the SAME constexpr array pointer.
//   3. The pointer is never dereferenced after the lookup — we only use it
//      as a hash key.
struct CachedModule {
    CUmodule module;
    CUfunction func;
};

std::mutex& moduleCacheMutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<void const*, CachedModule>& moduleCache() {
    static std::unordered_map<void const*, CachedModule> cache;
    return cache;
}

// NOTE: We intentionally do NOT unload cached CUmodules at DLL unload / process
// exit. Attempting cuModuleUnload() in a static destructor causes an access
// violation (0xC0000005) on Windows because:
//
//   1. TRT's own teardown runs first and destroys the CUDA context it created.
//      Calling cuModuleUnload() after the context is gone is undefined behavior
//      and crashes with an access violation.
//   2. Static destruction order: moduleCache() is a function-local static
//      (constructed on first cache miss), while any namespace-scope cleaner
//      would construct at DLL load — meaning the cleaner destructs AFTER the
//      map, accessing a dead object (another use-after-free).
//
// Leaking CUmodule handles during process exit is completely harmless: the
// CUDA driver reclaims all GPU memory when the process exits. The total leak
// is ~4 unique cubins × a few KB of driver-side metadata = negligible.
//
// If DLL hot-reload (unload without process exit) is ever needed, the cleanup
// must be done via an explicit teardown function called BEFORE TRT destroys
// its context — never via a static destructor.

constexpr size_t kCUBLASLT_WORKSPACE_BYTES = 32ull * 1024ull * 1024ull;

size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

char* alignPtr(char* ptr, size_t alignment) {
    uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
    raw = (raw + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
    return reinterpret_cast<char*>(raw);
}

int64_t flattenedRows(nvinfer1::Dims const& dims) {
    if (dims.nbDims < 1) return 0;
    int64_t rows = 1;
    for (int32_t i = 0; i < dims.nbDims - 1; ++i) {
        if (dims.d[i] <= 0) return 0;
        rows *= dims.d[i];
    }
    return rows;
}

int32_t lastDim(nvinfer1::Dims const& dims) {
    if (dims.nbDims < 1) return 0;
    return dims.d[dims.nbDims - 1];
}

int32_t normalizeDtypeId(int32_t value, int32_t defaultValue = 10) {
    // ONNX TensorProto enum: FLOAT=1, FLOAT16=10, BFLOAT16=16.
    if (value == 1 || value == 10 || value == 16) return value;
    return defaultValue;
}

int32_t parseDtypeId(char const* value, int32_t defaultValue = 10) {
    if (value == nullptr || value[0] == '\0') return defaultValue;
    std::string s(value);
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (s == "FP16" || s == "FLOAT16" || s == "HALF") return 10;
    if (s == "BF16" || s == "BFLOAT16") return 16;
    if (s == "FP32" || s == "FLOAT" || s == "FLOAT32") return 1;
    return defaultValue;
}

nvinfer1::DataType trtTypeFromDtypeId(int32_t dtypeId) {
    if (dtypeId == 10) return nvinfer1::DataType::kHALF;
    if (dtypeId == 16) return nvinfer1::DataType::kBF16;
    return nvinfer1::DataType::kFLOAT;
}

int32_t kernelDtypeFromTrt(nvinfer1::DataType dtype) {
    if (dtype == nvinfer1::DataType::kHALF) return 1;
    if (dtype == nvinfer1::DataType::kBF16) return 2;
    return 0;
}

char const* dtypeNameFromId(int32_t dtypeId) {
    if (dtypeId == 10) return "FP16";
    if (dtypeId == 16) return "BF16";
    return "FP32";
}

bool isSupportedPluginBoundaryDtypePair(int32_t input_dtype_id, int32_t output_dtype_id) {
    // Only FP32IO (ONNX dtype id 1) is compiled by default. The 16-bit
    // boundary cubin variants (FP16IO/BF16IO) were removed to simplify the
    // build and because the default HOTSTEP_W8A8_PLUGIN_IO_DTYPE=FP32 uses
    // FP32 everywhere anyway.
    return (input_dtype_id == 1 && output_dtype_id == 1);  // FP32IO
}

bool readPluginString(nvinfer1::PluginField const& f, char* dst, size_t dstSize) {
    if (dst == nullptr || dstSize == 0) return false;
    dst[0] = '\0';
    if (f.data == nullptr || f.length <= 0) return false;
    auto const* src = static_cast<char const*>(f.data);
    size_t maxLen = static_cast<size_t>(f.length);
    size_t n = 0;
    while (n < maxLen && src[n] != '\0') ++n;
    n = std::min(n, dstSize - 1);
    std::memcpy(dst, src, n);
    dst[n] = '\0';
    return true;
}

bool validGroupSize(int32_t groupSize, int32_t K) {
    // Version 6 supports GROUP_SIZE=256 per CONVROT_OPTIMAL_TWO_KERNEL_SPEC.md.
    // groupSize=0 is also accepted as a no-rotation sentinel for testing.
    if (K <= 0) return false;
    if (groupSize == 0) return true;
    return groupSize == 256 && K % 256 == 0;
}

int32_t activationBlockKForGroupSize(int32_t groupSize) {
    return (groupSize == 0) ? 1 : groupSize;
}

int32_t activationScaleGroups(int32_t K, int32_t groupSize) {
    int32_t const blockK = activationBlockKForGroupSize(groupSize);
    return (K + blockK - 1) / blockK;
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────
// ConvRotInt8LinearPlugin
// ──────────────────────────────────────────────────────────────────────────

ConvRotInt8LinearPlugin::ConvRotInt8LinearPlugin(
    int32_t group_size, int32_t in_features, int32_t out_features, int32_t has_bias,
    int32_t input_dtype_id, int32_t output_dtype_id, std::string preferred_format)
    : m_group_size(group_size),
      m_in_features(in_features),
      m_out_features(out_features),
      m_has_bias(has_bias),
      m_input_dtype_id(normalizeDtypeId(input_dtype_id)),
      m_output_dtype_id(normalizeDtypeId(output_dtype_id)),
      m_preferred_format(std::move(preferred_format)) {}

ConvRotInt8LinearPlugin::ConvRotInt8LinearPlugin(void const* data, size_t length) {
    if (length != getSerializationSize()) return;
    uint8_t const* d = static_cast<uint8_t const*>(data);
    int32_t input_dtype_id = 10, output_dtype_id = 10;
    std::memcpy(&m_group_size,   d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&m_in_features,  d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&m_out_features, d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&m_has_bias,     d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&input_dtype_id,  d, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(&output_dtype_id, d, sizeof(int32_t)); d += sizeof(int32_t);
    m_input_dtype_id = normalizeDtypeId(input_dtype_id);
    m_output_dtype_id = normalizeDtypeId(output_dtype_id);
}

ConvRotInt8LinearPlugin::~ConvRotInt8LinearPlugin() {
    destroyTriton();
}

// ── IPluginV3 ──────────────────────────────────────────────────────────────

nvinfer1::IPluginCapability* ConvRotInt8LinearPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept {
    if (type == nvinfer1::PluginCapabilityType::kBUILD)
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    if (type == nvinfer1::PluginCapabilityType::kRUNTIME)
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    return static_cast<nvinfer1::IPluginV3OneCore*>(this);
}

// ── IPluginV3OneCore ──────────────────────────────────────────────────────

char const* ConvRotInt8LinearPlugin::getPluginName() const noexcept {
    return kCONVROT_INT8_LINEAR_PLUGIN_NAME;
}
char const* ConvRotInt8LinearPlugin::getPluginVersion() const noexcept {
    return kCONVROT_INT8_LINEAR_PLUGIN_VERSION;
}
char const* ConvRotInt8LinearPlugin::getPluginNamespace() const noexcept {
    return m_namespace.c_str();
}
// Note: setPluginNamespace() removed in TRT 11; namespace is set via creator.

// ── IPluginV3OneBuild ──────────────────────────────────────────────────────

int32_t ConvRotInt8LinearPlugin::getNbOutputs() const noexcept { return 1; }

int32_t ConvRotInt8LinearPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const*, int32_t) const noexcept {
    if (nbOutputs < 1) return -1;
    if (!isSupportedPluginBoundaryDtypePair(m_input_dtype_id, m_output_dtype_id)) return -1;
    outputTypes[0] = trtTypeFromDtypeId(m_output_dtype_id);
    return 0;
}

int32_t ConvRotInt8LinearPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const*, int32_t,
    nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    if (nbOutputs < 1 || nbInputs < 1) return -1;
    auto const& x_dims = inputs[0];
    int32_t ndim = x_dims.nbDims;
    if (ndim < 1) return -1;
    outputs[0].nbDims = ndim;
    for (int32_t i = 0; i < ndim - 1; ++i)
        outputs[0].d[i] = x_dims.d[i];
    outputs[0].d[ndim - 1] = exprBuilder.constant(m_out_features);
    return 0;
}

bool ConvRotInt8LinearPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs, int32_t nbOutputs) noexcept {
    if (pos < 0 || pos >= nbInputs + nbOutputs) return false;
    if (!isSupportedPluginBoundaryDtypePair(m_input_dtype_id, m_output_dtype_id)) return false;

    // The current kernel indexes tensors as contiguous row-major memory. Keep
    // kLINEAR as the only accepted physical layout until the packed-layout
    // kernels land; the v2 dtype/tactic contract is independent of that future
    // optimization. The ONNX-side preferred_format attribute is still parsed
    // and serialized so future kernels can enable packed formats without graph
    // changes.
    if (inOut[pos].desc.format != nvinfer1::PluginFormat::kLINEAR) return false;

    auto type = inOut[pos].desc.type;
    auto inputType = trtTypeFromDtypeId(m_input_dtype_id);
    auto outputType = trtTypeFromDtypeId(m_output_dtype_id);
    if (pos < nbInputs) {
        switch (pos) {
            case 0: return type == inputType;                        // x
            case 1: return type == nvinfer1::DataType::kINT8;        // weight_q
            case 2: return type == nvinfer1::DataType::kFLOAT;       // weight_scale
            case 3: return m_has_bias && type == nvinfer1::DataType::kFLOAT; // bias stays FP32 (1D sane rule)
            default: return false;
        }
    }
    return type == outputType;  // output y
}

int32_t ConvRotInt8LinearPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    (void)out;
    hotstep::diag::log("[ConvRotInt8Linear] configurePlugin ENTER (BUILD phase): "
                     "nbInputs=%d, nbOutputs=%d, group_size=%d, in_features=%d, "
                     "out_features=%d, has_bias=%d, input_dtype=%s, output_dtype=%s\n",
                     nbInputs, nbOutputs,
                     m_group_size, m_in_features, m_out_features, m_has_bias,
                     dtypeNameFromId(m_input_dtype_id), dtypeNameFromId(m_output_dtype_id));
    if (nbInputs < (m_has_bias ? 4 : 3) || nbOutputs < 1) return -1;
    int64_t optM = flattenedRows(in[0].opt);
    int32_t optK = lastDim(in[0].opt);
    if (optM <= 0 || optK <= 0 || optK != m_in_features) return -1;
    if (!isSupportedPluginBoundaryDtypePair(m_input_dtype_id, m_output_dtype_id)) return -1;
    if (!validGroupSize(m_group_size, optK)) return -1;
    if (optM > std::numeric_limits<int32_t>::max()) return -1;
    m_M = static_cast<int32_t>(optM);
    m_K = optK;
    m_N = m_out_features;

    // Eagerly initialize the Triton cubin during the build phase so a cubin
    // load failure surfaces as a build-time error with a clear stderr message
    // (via initTriton's diagnostics below), rather than as a runtime assertion
    // inside PluginV3Runner::execute at C:\_src\runtime\gpu\cuda\pluginV3Runner.cpp:252.
    //
    // initTriton() will reject unsupported (group_size, has_bias, input_dtype,
    // output_dtype) combinations with a detailed message explaining which
    // combination was requested and which combinations are in the generated
    // cubin header. The DiT exporter (tools/onnx-export/export_dit.py) emits
    // FP32 plugin boundary dtypes by default, so the cubin header MUST
    // contain the FP32IO variant (see extract_jit_cubins_autotune.py).
    destroyTriton();
    if (!initTriton()) {
        hotstep::diag::log(
                     "[ConvRotInt8Linear] BUILD-TIME FAILURE: initTriton() failed during "
                     "configurePlugin() for group_size=%d, in_features=%d, "
                     "out_features=%d, has_bias=%d, input_dtype=%s, output_dtype=%s. "
                     "See prior stderr for details.\n",
                     m_group_size, m_in_features, m_out_features, m_has_bias,
                     dtypeNameFromId(m_input_dtype_id),
                     dtypeNameFromId(m_output_dtype_id));
        return -1;
    }
    return 0;
}

// Note: destroy() removed in TRT 11; the destructor handles cleanup.

size_t ConvRotInt8LinearPlugin::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs,
                                                 int32_t nbInputs,
                                                 nvinfer1::DynamicPluginTensorDesc const* outputs,
                                                 int32_t nbOutputs) const noexcept {
    (void)nbInputs;
    (void)outputs;
    (void)nbOutputs;

    // TensorRT's V3 build-time contract provides concrete profile bounds in
    // min/opt/max, while desc.dims may still contain wildcards. Workspace must
    // therefore be sized from the profile's MAX shape, not from desc.dims.
    int64_t maxM = flattenedRows(inputs[0].max);
    if (maxM <= 0) maxM = flattenedRows(inputs[0].opt);
    // M==1 uses the same two-kernel path as every other runtime shape, so it
    // still needs workspace for the single quantized activation row and scale.
    if (maxM <= 0) return 0;

    int32_t K = lastDim(inputs[0].max);
    if (K <= 0) K = m_in_features;
    if (K <= 0) return 0;

    int32_t const scale_groups = activationScaleGroups(K, m_group_size);
    size_t x_q_size = alignUp(static_cast<size_t>(maxM) * static_cast<size_t>(K), 16);
    // X_scale is now FP32 (was FP16/uint16_t) — FP16 storage caused measurable
    // quality reduction vs the ggml/non-Triton baselines.  The per-group scale
    // is applied to every element of the INT32 partial accumulator, so even
    // small FP16 rounding errors compound across K_groups.
    size_t x_scale_size = alignUp(
        static_cast<size_t>(maxM) * static_cast<size_t>(scale_groups) * sizeof(float),
        16);
    return x_q_size + x_scale_size;
}

// ── Custom Tactics (IPluginV3OneBuild) ──────────────────────────────────────

int32_t ConvRotInt8LinearPlugin::getNbTactics() noexcept {
    // Simplest correct sm86-first behavior: advertise exactly one custom tactic
    // that is guaranteed to exist in the generated cubin header. TensorRT still
    // retains implicit default tactic 0.
    return 1;
}

int32_t ConvRotInt8LinearPlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept {
    if (tactics == nullptr || nbTactics < 1) return 1;
    // TensorRT reserves tactic 0 for the implicit default path; advertised
    // custom tactic IDs must be unique and non-zero.
    // getValidTactics returns an error code, not the number of tactics.
    tactics[0] = 2;  // 128x128 Triton tile
    return 0;
}

char const* ConvRotInt8LinearPlugin::getTimingCacheID() noexcept {
    return "ConvRotInt8Linear.v2";
}

int32_t ConvRotInt8LinearPlugin::getFormatCombinationLimit() noexcept {
    // One physical layout combination today (kLINEAR), with dtype selected by
    // input_dtype/output_dtype. Packed-layout tactics can raise this later
    // without changing the ONNX custom-op schema.
    return 1;
}

char const* ConvRotInt8LinearPlugin::getMetadataString() noexcept {
    // Return a string describing the plugin config (for engine inspector).
    // This is called once; the buffer must live as long as the plugin.
    static thread_local std::string meta;
    std::ostringstream oss;
    oss << "ConvRotInt8Linear.v2(gs=" << m_group_size
        << ",K=" << m_in_features
        << ",N=" << m_out_features
        << ",bias=" << m_has_bias
        << ",in=" << dtypeNameFromId(m_input_dtype_id)
        << ",out=" << dtypeNameFromId(m_output_dtype_id)
        << ",fmt=" << m_preferred_format
        << ",tactic=" << m_tactic << ")";
    meta = oss.str();
    return meta.c_str();
}

// ── IPluginV3OneRuntime ────────────────────────────────────────────────────

int32_t ConvRotInt8LinearPlugin::setTactic(int32_t tactic) noexcept {
    // TensorRT will call this with default tactic 0 or one of the advertised
    // custom tactics. For the current sm86-first implementation, only tactic 2
    // is advertised and both 0 and 2 resolve to the same 128x128 kernel.
    if (tactic != 0 && tactic != 2) {
        hotstep::diag::log("[ConvRotInt8Linear] setTactic(%d) REJECTED — only 0 and 2 are valid.\n",
                         tactic);
        return -1;
    }
    m_tactic = tactic;
    return 0;
}

int32_t ConvRotInt8LinearPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    hotstep::diag::log("[ConvRotInt8Linear] onShapeChange ENTER: nbInputs=%d, nbOutputs=%d, "
                     "serialized(group_size=%d, in_features=%d, out_features=%d, has_bias=%d, "
                     "input_dtype=%s, output_dtype=%s, quant=%p, gemm=%p, m_tactic=%d)\n",
                     nbInputs, nbOutputs,
                     m_group_size, m_in_features, m_out_features, m_has_bias,
                     dtypeNameFromId(m_input_dtype_id), dtypeNameFromId(m_output_dtype_id),
                     m_module_quant, m_module_gemm, m_tactic);

    if (nbInputs < (m_has_bias ? 4 : 3) || nbOutputs < 1) {
        hotstep::diag::log("[ConvRotInt8Linear] onShapeChange REJECT: nbInputs=%d < required=%d "
                         "OR nbOutputs=%d < 1. m_has_bias=%d.\n",
                         nbInputs, (m_has_bias ? 4 : 3), nbOutputs, m_has_bias);
        return -1;
    }
    // Extract M, K, N from the descriptors.
    (void)out;
    auto const& x_desc = in[0].dims;
    int64_t runtimeM = flattenedRows(x_desc);
    if (runtimeM <= 0 || runtimeM > std::numeric_limits<int32_t>::max()) {
        hotstep::diag::log("[ConvRotInt8Linear] onShapeChange REJECT: runtimeM=%lld is out of "
                         "range (must be in [1, %d]).\n",
                         (long long)runtimeM, std::numeric_limits<int32_t>::max());
        return -1;
    }
    m_M = static_cast<int32_t>(runtimeM);
    m_K = lastDim(x_desc);
    m_N = m_out_features;

    hotstep::diag::log("[ConvRotInt8Linear] onShapeChange shapes: M=%d, K=%d (expected=%d), N=%d, "
                     "x_desc.nbDims=%d, x_desc.d=[",
                     m_M, m_K, m_in_features, m_N, x_desc.nbDims);
    for (int32_t i = 0; i < x_desc.nbDims; ++i) {
        hotstep::diag::log("%s%d", (i ? "," : ""), x_desc.d[i]);
    }
    hotstep::diag::log("]\n");

    if (m_K != m_in_features) {
        hotstep::diag::log("[ConvRotInt8Linear] onShapeChange REJECT: runtime K=%d != serialized "
                         "in_features=%d. The ONNX input shape does not match the build-time "
                         "contract.\n",
                         m_K, m_in_features);
        return -1;
    }
    if (!validGroupSize(m_group_size, m_K)) {
        hotstep::diag::log("[ConvRotInt8Linear] onShapeChange REJECT: group_size=%d not valid "
                         "for K=%d (must divide K evenly, or be 0).\n",
                         m_group_size, m_K);
        return -1;
    }
    if (m_M <= 0 || m_K <= 0 || m_N <= 0) {
        hotstep::diag::log("[ConvRotInt8Linear] onShapeChange REJECT: non-positive dim "
                         "(M=%d, K=%d, N=%d).\n", m_M, m_K, m_N);
        return -1;
    }

    // M==1 intentionally uses the same two-kernel path as all larger runtime
    // shapes. configurePlugin() may have eagerly loaded the cubins for the
    // profile's OPT shape, so onShapeChange() only needs to ensure the cached
    // two-kernel family is present for the concrete runtime dtype/bias config.
    bool const have_two_kernel = (m_kernelFunc_quant != nullptr && m_kernelFunc_gemm != nullptr);

    if (!have_two_kernel) {
        destroyTriton();
        if (!initTriton()) {
            hotstep::diag::log(
                         "[ConvRotInt8Linear] onShapeChange FAILURE: initTriton() failed for "
                         "group_size=%d, in_features=%d, out_features=%d, has_bias=%d, "
                         "M=%d, K=%d, N=%d, input_dtype=%s, output_dtype=%s.\n",
                         m_group_size, m_in_features, m_out_features, m_has_bias,
                         m_M, m_K, m_N,
                         dtypeNameFromId(m_input_dtype_id),
                         dtypeNameFromId(m_output_dtype_id));
            hotstep::diag::flush();
            return -1;
        }
    }
    hotstep::diag::log("[ConvRotInt8Linear] onShapeChange OK — returning 0.\n");
    return 0;
}

int32_t ConvRotInt8LinearPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs,
    void* workspace, cudaStream_t stream) noexcept {
    try {
        (void)inputDesc;
        (void)outputDesc;
        int32_t M = m_M;
        int32_t K = m_K;
        int32_t N = m_N;
        if (M <= 0 || K <= 0 || N <= 0) return -1;

        void const* x_ptr = inputs[0];
        int8_t const* wq_ptr = static_cast<int8_t const*>(inputs[1]);
        float const* ws_ptr = static_cast<float const*>(inputs[2]);
        void const* bias_ptr = m_has_bias ? inputs[3] : nullptr;
        void* y_ptr = outputs[0];

        int32_t stride_xm = K;
        int32_t stride_xk = 1;
        int32_t stride_wn = K;
        int32_t stride_wk = 1;
        int32_t stride_ym = N;
        int32_t stride_yn = 1;

        // Generated launch stubs in convrot_int8_kernel_cubin.h own the exact
        // Triton launch ABI, including trailing scratch-pointer parameters.

        if (!m_kernelFunc_quant || !m_kernelFunc_gemm || workspace == nullptr) {
#ifdef HOTSTEP_DIAGNOSTICS
            hotstep::diag::log("[ConvRotInt8Linear] enqueue REJECT: quant=%p, gemm=%p, workspace=%p\n",
                             (void*)m_kernelFunc_quant, (void*)m_kernelFunc_gemm, workspace);
            hotstep::diag::flush();
#endif
            return -1;
        }

        size_t const x_q_size = alignUp(static_cast<size_t>(M) * static_cast<size_t>(K), 16);
        void* xq_ptr = workspace;
        void* xs_ptr = static_cast<char*>(workspace) + x_q_size;

        int32_t stride_xqm = K;
        int32_t stride_xqk = 1;
        // X_scale workspace layout: TRANSPOSED [n_groups, M] row-major.
        // (Per CONVROT_OPTIMAL_TWO_KERNEL_SPEC.md §1.3: makes per-row writes/reads
        // coalesced — consecutive rm values become adjacent 4-byte floats.)
        // stride_xsm = 1 (M is the inner stride), stride_xsg = M (group is outer).
        int32_t stride_xsm = 1;
        int32_t stride_xsg = M;

        auto const* quant_desc = static_cast<hotstep::convrot_int8_generated::ConvRotCubinDesc const*>(m_desc_quant);
        auto const* gemm_desc = static_cast<hotstep::convrot_int8_generated::ConvRotCubinDesc const*>(m_desc_gemm);
        if (quant_desc == nullptr || gemm_desc == nullptr) return -1;

        CUresult status = hotstep::convrot_int8_generated::launchConvRotQuant(
            *quant_desc,
            static_cast<CUfunction>(m_kernelFunc_quant),
            static_cast<CUstream>(stream),
            x_ptr,
            xq_ptr,
            xs_ptr,
            M,
            K,
            m_num_sms,              // NUM_SMS (persistent grid-stride loop)
            stride_xm,
            stride_xk,
            stride_xqm,
            stride_xqk,
            stride_xsm,
            stride_xsg);
        if (status != CUDA_SUCCESS) {
#ifdef HOTSTEP_DIAGNOSTICS
            char const* err_str = "unknown";
            cuGetErrorString(status, &err_str);
            cudaError_t async_status = cudaPeekAtLastError();
            int shared_bytes_static = 0;
            cuFuncGetAttribute(&shared_bytes_static,
                               CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                               static_cast<CUfunction>(m_kernelFunc_quant));
            int num_regs = 0, max_threads_per_block = 0;
            cuFuncGetAttribute(&num_regs,
                               CU_FUNC_ATTRIBUTE_NUM_REGS,
                               static_cast<CUfunction>(m_kernelFunc_quant));
            cuFuncGetAttribute(&max_threads_per_block,
                               CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                               static_cast<CUfunction>(m_kernelFunc_quant));
            hotstep::diag::log(
                         "[ConvRotInt8Linear] enqueue FAILURE: quant cuLaunchKernel failed: %s "
                         "(code=%d, tactic=%d, group_size=%d, kernel=%s, "
                         "persistent grid=(%u,1,1) [num_sms=%u, num_pid_m=%u, num_k_groups=%u], "
                         "block=(%u,%u,%u), shared_mem_requested=%u, shared_mem_static=%d bytes, "
                         "num_regs=%d, max_threads_per_block=%d, cudaPeekAtLastError=%d (%s), "
                         "M=%d, K=%d, N=%d, input_dtype=%s, output_dtype=%s)\n",
                         err_str, status, m_tactic, m_group_size, quant_desc->logical_name,
                         hotstep::convrot_int8_generated::persistentGridU32(
                             hotstep::convrot_int8_generated::ceilDivU32(M, quant_desc->block_m) *
                             static_cast<uint32_t>(K / m_group_size),
                             static_cast<uint32_t>(m_num_sms)),
                         static_cast<uint32_t>(m_num_sms),
                         hotstep::convrot_int8_generated::ceilDivU32(M, quant_desc->block_m),
                         static_cast<uint32_t>(K / m_group_size),
                         quant_desc->block_x, quant_desc->block_y, quant_desc->block_z,
                         static_cast<unsigned int>(quant_desc->shared_bytes), shared_bytes_static, num_regs, max_threads_per_block,
                         static_cast<int>(async_status), cudaGetErrorString(async_status),
                         M, K, N,
                         dtypeNameFromId(m_input_dtype_id),
                         dtypeNameFromId(m_output_dtype_id));
            hotstep::diag::flush();
#endif
            return -1;
        }

        status = hotstep::convrot_int8_generated::launchConvRotGemm(
            *gemm_desc,
            static_cast<CUfunction>(m_kernelFunc_gemm),
            static_cast<CUstream>(stream),
            xq_ptr,
            xs_ptr,
            wq_ptr,
            ws_ptr,
            bias_ptr,
            y_ptr,
            M,
            N,
            K,
            m_num_sms,              // NUM_SMS (persistent grid-stride loop)
            stride_xqm,
            stride_xqk,
            stride_xsm,
            stride_xsg,
            stride_wn,
            stride_wk,
            stride_ym,
            stride_yn);
        if (status != CUDA_SUCCESS) {
            return -1;
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

nvinfer1::IPluginV3* ConvRotInt8LinearPlugin::clone() noexcept {
    auto* p = new ConvRotInt8LinearPlugin(m_group_size, m_in_features,
                                          m_out_features, m_has_bias,
                                          m_input_dtype_id, m_output_dtype_id,
                                          m_preferred_format);
    p->m_namespace = m_namespace;
    p->m_tactic = m_tactic;
    return p;
}

nvinfer1::IPluginV3* ConvRotInt8LinearPlugin::attachToContext(
    nvinfer1::IPluginResourceContext* ctx) noexcept {
    // TRT 11 requires this. For our simple plugin that doesn't use
    // plugin resources, just return a clone.
    (void)ctx;
    return clone();
}

nvinfer1::PluginFieldCollection const* ConvRotInt8LinearPlugin::getFieldsToSerialize() noexcept {
    m_fields.clear();
    m_fields.push_back({kFIELD_GROUP_SIZE,   &m_group_size,   nvinfer1::PluginFieldType::kINT32, 1});
    m_fields.push_back({kFIELD_IN_FEATURES,  &m_in_features,  nvinfer1::PluginFieldType::kINT32, 1});
    m_fields.push_back({kFIELD_OUT_FEATURES, &m_out_features, nvinfer1::PluginFieldType::kINT32, 1});
    m_fields.push_back({kFIELD_HAS_BIAS,     &m_has_bias,     nvinfer1::PluginFieldType::kINT32, 1});
    static thread_local std::string input_dtype;
    static thread_local std::string output_dtype;
    input_dtype = dtypeNameFromId(m_input_dtype_id);
    output_dtype = dtypeNameFromId(m_output_dtype_id);
    static thread_local int32_t input_dtype_id;
    static thread_local int32_t output_dtype_id;
    input_dtype_id = m_input_dtype_id;
    output_dtype_id = m_output_dtype_id;
    m_fields.push_back({kFIELD_INPUT_DTYPE,  input_dtype.c_str(),  nvinfer1::PluginFieldType::kCHAR, static_cast<int32_t>(input_dtype.size() + 1)});
    m_fields.push_back({kFIELD_OUTPUT_DTYPE, output_dtype.c_str(), nvinfer1::PluginFieldType::kCHAR, static_cast<int32_t>(output_dtype.size() + 1)});
    m_fields.push_back({kFIELD_INPUT_DTYPE_ID,  &input_dtype_id,  nvinfer1::PluginFieldType::kINT32, 1});
    m_fields.push_back({kFIELD_OUTPUT_DTYPE_ID, &output_dtype_id, nvinfer1::PluginFieldType::kINT32, 1});
    m_fields.push_back({kFIELD_PREFERRED_FORMAT, m_preferred_format.c_str(), nvinfer1::PluginFieldType::kCHAR, static_cast<int32_t>(m_preferred_format.size() + 1)});
    m_fc.nbFields = static_cast<int32_t>(m_fields.size());
    m_fc.fields = m_fields.data();
    return &m_fc;
}

size_t ConvRotInt8LinearPlugin::getSerializationSize() const noexcept {
    return 6 * sizeof(int32_t);  // group_size, in_features, out_features, has_bias, input_dtype_id, output_dtype_id
}

void ConvRotInt8LinearPlugin::serialize(void* buffer) const noexcept {
    uint8_t* d = static_cast<uint8_t*>(buffer);
    std::memcpy(d, &m_group_size,   sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_in_features,  sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_out_features, sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_has_bias,     sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_input_dtype_id,  sizeof(int32_t)); d += sizeof(int32_t);
    std::memcpy(d, &m_output_dtype_id, sizeof(int32_t)); d += sizeof(int32_t);
}

// ── Triton / CUDA Driver initialization ─────────────────────────────────────

namespace {

// Compile-time-valid group sizes for the phase-1 butterfly kernel.
// The current generated Triton inventory supports group_size=256 (and optional G0 sentinel); there
// is no legacy G64 cubin.

#if CONVROT_INT8_KERNEL_CUBIN_HEADER_AVAILABLE
using GeneratedCubinDesc = hotstep::convrot_int8_generated::ConvRotCubinDesc;

GeneratedCubinDesc const* selectCubinK1(int32_t group_size,
                                        int32_t input_dtype_id,
                                        int32_t output_dtype_id) {
    return hotstep::convrot_int8_generated::findConvRotCubin(
        hotstep::convrot_int8_generated::ConvRotKernelStage::kQuant,
        group_size,
        false,
        input_dtype_id,
        output_dtype_id);
}

GeneratedCubinDesc const* selectCubinK2(int32_t group_size,
                                        bool has_bias,
                                        int32_t input_dtype_id,
                                        int32_t output_dtype_id) {
    return hotstep::convrot_int8_generated::findConvRotCubin(
        hotstep::convrot_int8_generated::ConvRotKernelStage::kGemm,
        group_size,
        has_bias,
        input_dtype_id,
        output_dtype_id);
}
#else
struct GeneratedCubinDesc {
    char const* logical_name;
    unsigned char const* data;
    size_t size;
    char const* function_name;
    size_t shared_bytes;
};
GeneratedCubinDesc const* selectCubinK1(int32_t, int32_t, int32_t) { return nullptr; }
GeneratedCubinDesc const* selectCubinK2(int32_t, bool, int32_t, int32_t) { return nullptr; }
#endif

bool loadOrReuseKernel(GeneratedCubinDesc const* cubin,
                       void** module_out,
                       void** func_out) {
    if (cubin == nullptr || cubin->data == nullptr || cubin->size == 0) return false;

    unsigned int const device_max_smem = getDeviceMaxDynamicSharedMem();
    if (cubin->shared_bytes > device_max_smem) {
        hotstep::diag::log(
            "[ConvRotInt8Linear] Triton cubin %s requires %zu bytes of dynamic shared memory, "
            "but the device only supports %u bytes.\n",
            cubin->logical_name, cubin->shared_bytes, device_max_smem);
        return false;
    }

    CachedModule cached{};
    {
        std::lock_guard<std::mutex> lock(moduleCacheMutex());
        auto& cache = moduleCache();
        auto it = cache.find(cubin->data);
        if (it == cache.end()) {
            CUmodule new_module = nullptr;
            CUresult status = cuModuleLoadData(&new_module, cubin->data);
            if (status != CUDA_SUCCESS) {
                char const* err_str = "unknown";
                cuGetErrorString(status, &err_str);
                hotstep::diag::log("[ConvRotInt8Linear] cuModuleLoadData(%s) failed: %s (%d).\n",
                                   cubin->logical_name, err_str, static_cast<int>(status));
                return false;
            }

            CUfunction new_func = nullptr;
            status = cuModuleGetFunction(&new_func, new_module, cubin->function_name);
            if (status != CUDA_SUCCESS) {
                char const* err_str = "unknown";
                cuGetErrorString(status, &err_str);
                hotstep::diag::log("[ConvRotInt8Linear] cuModuleGetFunction(%s/%s) failed: %s (%d).\n",
                                   cubin->logical_name, cubin->function_name, err_str, static_cast<int>(status));
                cuModuleUnload(new_module);
                return false;
            }

            if (cubin->shared_bytes > 48 * 1024) {
                status = cuFuncSetAttribute(
                    new_func,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    static_cast<int>(cubin->shared_bytes));
                if (status != CUDA_SUCCESS) {
                    char const* err_str = "unknown";
                    cuGetErrorString(status, &err_str);
                    hotstep::diag::log(
                        "[ConvRotInt8Linear] cuFuncSetAttribute(%s, %zu) failed: %s (%d).\n",
                        cubin->logical_name, cubin->shared_bytes, err_str, static_cast<int>(status));
                    cuModuleUnload(new_module);
                    return false;
                }
            }

            it = cache.emplace(cubin->data, CachedModule{new_module, new_func}).first;
        }
        cached = it->second;
    }

    if (module_out) *module_out = cached.module;
    if (func_out) *func_out = cached.func;
    return true;
}

} // namespace

bool ConvRotInt8LinearPlugin::initTriton() {
    if (!isSupportedPluginBoundaryDtypePair(m_input_dtype_id, m_output_dtype_id)) {
        hotstep::diag::log(
            "[ConvRotInt8Linear] No compiled Triton cubin for input_dtype=%s, output_dtype=%s. "
            "Only FP32->FP32 plugin boundary dtype is supported.\n",
            dtypeNameFromId(m_input_dtype_id), dtypeNameFromId(m_output_dtype_id));
        return false;
    }

    if (m_kernelFunc_quant != nullptr && m_kernelFunc_gemm != nullptr &&
        m_desc_quant != nullptr && m_desc_gemm != nullptr) {
        return true;
    }

    GeneratedCubinDesc const* quant_cubin = selectCubinK1(
        m_group_size, m_input_dtype_id, m_output_dtype_id);
    GeneratedCubinDesc const* gemm_cubin = selectCubinK2(
        m_group_size, m_has_bias != 0, m_input_dtype_id, m_output_dtype_id);
    if (quant_cubin == nullptr || gemm_cubin == nullptr) {
        hotstep::diag::log(
            "[ConvRotInt8Linear] Missing generated cubin descriptor for group_size=%d, "
            "has_bias=%d, input_dtype=%s, output_dtype=%s.\n",
            m_group_size, m_has_bias,
            dtypeNameFromId(m_input_dtype_id), dtypeNameFromId(m_output_dtype_id));
        return false;
    }

    bool const ok_quant = loadOrReuseKernel(
        quant_cubin,
        &m_module_quant,
        &m_kernelFunc_quant);
    bool const ok_gemm = ok_quant && loadOrReuseKernel(
        gemm_cubin,
        &m_module_gemm,
        &m_kernelFunc_gemm);
    if (!ok_gemm) return false;

    m_desc_quant = quant_cubin;
    m_desc_gemm = gemm_cubin;

    // ── Query device SM count for persistent-kernel grid computation ──
    // H_16 is generated in-kernel (no host buffer needed).  NUM_SMS is passed
    // as a runtime int32 arg to both kernels and used for the persistent
    // grid-stride loop.
    if (m_num_sms == 0) {
        int32_t dev = 0;
        cuCtxGetDevice(&dev);
        int sms = 0;
        CUresult const sm_rc = cuDeviceGetAttribute(
            &sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
        if (sm_rc != CUDA_SUCCESS || sms <= 0) {
            hotstep::diag::log("[ConvRotInt8Linear] Failed to query SM count: %d, using 1\n", sm_rc);
            m_num_sms = 1;
        } else {
            m_num_sms = static_cast<int32_t>(sms);
        }
    }

    return true;
}

void ConvRotInt8LinearPlugin::destroyTriton() {
    m_module_quant = nullptr;
    m_kernelFunc_quant = nullptr;
    m_desc_quant = nullptr;

    m_module_gemm = nullptr;
    m_kernelFunc_gemm = nullptr;
    m_desc_gemm = nullptr;

    m_num_sms = 0;
}

// ──────────────────────────────────────────────────────────────────────────
// ConvRotInt8LinearPluginCreator
// ──────────────────────────────────────────────────────────────────────────

ConvRotInt8LinearPluginCreator::ConvRotInt8LinearPluginCreator() {
    m_fields.emplace_back(kFIELD_GROUP_SIZE,   nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_IN_FEATURES,  nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_OUT_FEATURES, nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_HAS_BIAS,     nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_INPUT_DTYPE,  nullptr, nvinfer1::PluginFieldType::kCHAR, 0);
    m_fields.emplace_back(kFIELD_OUTPUT_DTYPE, nullptr, nvinfer1::PluginFieldType::kCHAR, 0);
    m_fields.emplace_back(kFIELD_INPUT_DTYPE_ID,  nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_OUTPUT_DTYPE_ID, nullptr, nvinfer1::PluginFieldType::kINT32, 1);
    m_fields.emplace_back(kFIELD_PREFERRED_FORMAT, nullptr, nvinfer1::PluginFieldType::kCHAR, 0);
    m_fc.nbFields = static_cast<int32_t>(m_fields.size());
    m_fc.fields = m_fields.data();
}

char const* ConvRotInt8LinearPluginCreator::getPluginName() const noexcept {
    return kCONVROT_INT8_LINEAR_PLUGIN_NAME;
}
char const* ConvRotInt8LinearPluginCreator::getPluginVersion() const noexcept {
    return kCONVROT_INT8_LINEAR_PLUGIN_VERSION;
}
char const* ConvRotInt8LinearPluginCreator::getPluginNamespace() const noexcept {
    return m_namespace.c_str();
}
// Note: setPluginNamespace() removed in TRT 11; namespace set via constructor.
nvinfer1::PluginFieldCollection const* ConvRotInt8LinearPluginCreator::getFieldNames() noexcept {
    return &m_fc;
}

nvinfer1::IPluginV3* ConvRotInt8LinearPluginCreator::createPlugin(
    char const*, nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase) noexcept {
    int32_t group_size = 0, in_features = 0, out_features = 0, has_bias = 0;
    int32_t input_dtype_id = 10;
    int32_t output_dtype_id = 10;
    std::string preferred_format = "HWC8";
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        auto const& f = fc->fields[i];
        if (f.name == nullptr || f.data == nullptr) continue;
        if (std::strcmp(f.name, kFIELD_GROUP_SIZE) == 0)
            group_size = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_IN_FEATURES) == 0)
            in_features = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_OUT_FEATURES) == 0)
            out_features = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_HAS_BIAS) == 0)
            has_bias = *static_cast<int32_t const*>(f.data);
        else if (std::strcmp(f.name, kFIELD_INPUT_DTYPE) == 0) {
            if (f.type == nvinfer1::PluginFieldType::kINT32) {
                input_dtype_id = (*static_cast<int32_t const*>(f.data) != 0) ? 10 : 1;
            } else {
                char tmp[16];
                if (readPluginString(f, tmp, sizeof(tmp))) input_dtype_id = parseDtypeId(tmp, input_dtype_id);
            }
        } else if (std::strcmp(f.name, kFIELD_OUTPUT_DTYPE) == 0) {
            if (f.type == nvinfer1::PluginFieldType::kINT32) {
                output_dtype_id = (*static_cast<int32_t const*>(f.data) != 0) ? 10 : 1;
            } else {
                char tmp[16];
                if (readPluginString(f, tmp, sizeof(tmp))) output_dtype_id = parseDtypeId(tmp, output_dtype_id);
            }
        } else if (std::strcmp(f.name, kFIELD_INPUT_DTYPE_ID) == 0) {
            input_dtype_id = normalizeDtypeId(*static_cast<int32_t const*>(f.data), input_dtype_id);
        } else if (std::strcmp(f.name, kFIELD_OUTPUT_DTYPE_ID) == 0) {
            output_dtype_id = normalizeDtypeId(*static_cast<int32_t const*>(f.data), output_dtype_id);
        } else if (std::strcmp(f.name, kFIELD_PREFERRED_FORMAT) == 0) {
            char tmp[32];
            if (readPluginString(f, tmp, sizeof(tmp))) preferred_format = tmp;
        }
    }
    return new ConvRotInt8LinearPlugin(group_size, in_features, out_features, has_bias,
                                       input_dtype_id, output_dtype_id, preferred_format);
}

}  // namespace hotstep

// ──────────────────────────────────────────────────────────────────────────
// Static registration
// ──────────────────────────────────────────────────────────────────────────

// TRT 11: REGISTER_TENSORRT_PLUGIN macro changed and doesn't work
// the same way. Use manual registration via hotstep_register_plugins() below.
// REGISTER_TENSORRT_PLUGIN(hotstep::ConvRotInt8LinearPluginCreator);

// Portable DLL export macro for the plugin entry point.
// On Windows, __declspec(dllexport) is required for GetProcAddress() to find
// the symbol. On Linux/macOS, default visibility is sufficient.
#if defined(_WIN32)
#  define HOTSTEP_PLUGIN_EXPORT __declspec(dllexport)
#else
#  define HOTSTEP_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
HOTSTEP_PLUGIN_EXPORT int hotstep_register_plugins() {
    try {
        // ── Cubin header state log ──────────────────────────────────────────
        // Emit the cubin inventory at DLL load time so the user can verify
        // (via hotstep_plugin_diag.log) which dtype variants are actually
        // compiled into the DLL. This catches the common failure mode where
        // the user updated the C++ source but forgot to re-run
        // extract_jit_cubins_autotune.py to regenerate the header with the
        // new DTYPE_CONFIGS — in that case the CONVROT_INT8_HAS_DTYPE_*
        // markers below will be missing and dtypeSuffixFor() will return
        // nullptr for every plugin invocation.
        hotstep::diag::log("=== HotStep plugin DLL loaded ===\n");
        hotstep::diag::log("Cubin header state:\n");
#if defined(CONVROT_INT8_HAS_DTYPE_FP32IO)
        hotstep::diag::log("  CONVROT_INT8_HAS_DTYPE_FP32IO = 1 (FP32-in/FP32-out cubins present)\n");
#else
        hotstep::diag::log("  CONVROT_INT8_HAS_DTYPE_FP32IO = UNDEFINED (no FP32-in/FP32-out cubins)\n");
#  error "Cubin header is missing CONVROT_INT8_HAS_DTYPE_FP32IO. Re-run tools/onnx-export/extract_jit_cubins_autotune.py to regenerate engine/src/plugins/assets/convrot_int8_kernel_cubin.h."
#endif
        hotstep::diag::log("If FP32IO is UNDEFINED, you forgot to re-run "
                         "tools/onnx-export/extract_jit_cubins_autotune.py to regenerate "
                         "engine/src/plugins/assets/convrot_int8_kernel_cubin.h after "
                         "updating the C++ plugin source.\n");
        hotstep::diag::log("Log file path: %s (override with HOTSTEP_PLUGIN_LOG env var)\n",
                         hotstep::diag::logPath().c_str());

        // TRT 11: getPluginRegistry() is an extern "C" free function declared
        // in NvInferRuntime.h (included transitively via NvInfer.h).
        // It is in the global namespace, NOT in nvinfer1.
        auto* registry = getPluginRegistry();
        if (!registry) {
            hotstep::diag::log("hotstep_register_plugins: getPluginRegistry() returned null.\n");
            return 1;
        }
        static hotstep::ConvRotInt8LinearPluginCreator creator;
        registry->registerCreator(creator, hotstep::kCONVROT_INT8_LINEAR_PLUGIN_NAMESPACE);
        hotstep::diag::log("hotstep_register_plugins: ConvRotInt8Linear v2 creator registered OK.\n");
        return 0;
    } catch (...) {
        hotstep::diag::log("hotstep_register_plugins: C++ exception thrown during registration.\n");
        return 2;
    }
}
}  // extern "C"

#endif  // HOT_STEP_TRT
