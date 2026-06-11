// test-prune-bundle.cpp
// Behavior test for store_prune_bundle_except (model-store TRT bundle group-evict).
//
// Populates a ModelStore GPU cache with synthetic entries spanning two trt-bundles
// directories plus a non-bundle (GGUF) entry, prunes keeping one bundle, and asserts:
//   - other-bundle refcount-0 entries are evicted (deleter ran, removed from cache)
//   - keep-bundle entries are untouched
//   - the non-bundle entry is untouched
//   - a refcount>0 other-bundle entry triggers the fatal abort path
//
// The fatal path is verified by re-invoking this exe with arg "abort-case" in a
// child process and asserting it terminates abnormally (abort), so the harness
// process itself survives.
//
// Build: linked as a CMake target (test-prune-bundle) against acestep-core with
// HOT_STEP_MODEL_STORE_TEST defined. See engine/CMakeLists.txt.

#include "model-store.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <sys/wait.h>
#  include <unistd.h>
#endif

static int g_failures = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);           \
            g_failures++;                                                       \
        } else {                                                               \
            fprintf(stderr, "ok: %s\n", (msg));                                 \
        }                                                                       \
    } while (0)

// Path layout used by the test. Mixed separators on purpose to prove the
// bundle-dir derivation is separator-agnostic.
static const char * BUNDLE_A_DIT  = "models/trt-bundles/v15-base/dit.engine";
static const char * BUNDLE_A_TEXT = "models/trt-bundles/v15-base/text_encoder.engine";
static const char * BUNDLE_A_COND = "models\\trt-bundles\\v15-base\\cond_encoder.engine";
static const char * BUNDLE_B_DIT  = "models/trt-bundles/v15-turbo/dit.engine";
static const char * BUNDLE_B_TEXT = "models/trt-bundles/v15-turbo/text_encoder.engine";
static const char * NON_BUNDLE    = "models/acestep-v15-q8.gguf";
static const char * KEEP_BUNDLE   = "models/trt-bundles/v15-base";  // bundle dir itself

// The abort-case child: a single refcount>0 entry in the OTHER bundle must abort.
static int run_abort_case() {
    ModelStore * s = store_create(EVICT_STRICT);
    int          calls = 0;
    store_test_install_synthetic(s, MODEL_DIT_TRT, BUNDLE_A_DIT, 100, 0, &calls);
    store_test_install_synthetic(s, MODEL_DIT_TRT, BUNDLE_B_DIT, 200, 1, &calls);  // in use, other bundle
    store_prune_bundle_except(s, KEEP_BUNDLE);  // expected to abort()
    // Must never reach here.
    fprintf(stderr, "FAIL: abort-case did not abort\n");
    store_free(s);
    return 0;
}

// Spawn this same exe with arg "abort-case" and report whether it aborted.
static bool child_aborts(const char * self) {
#ifdef _WIN32
    std::string cmd = std::string("\"") + self + "\" abort-case";
    STARTUPINFOA        si = {};
    PROCESS_INFORMATION pi = {};
    si.cb = sizeof(si);
    // Suppress the Windows abort() error dialog in the child so CI doesn't hang.
    std::string env = "";
    char * cmdline = _strdup(cmd.c_str());
    BOOL ok = CreateProcessA(nullptr, cmdline, nullptr, nullptr, FALSE,
                             CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    free(cmdline);
    if (!ok) {
        fprintf(stderr, "FAIL: CreateProcess for abort-case failed (%lu)\n", GetLastError());
        return false;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    // abort() on Windows yields exit code 3 (or STATUS_* abnormal); any
    // non-zero exit means the prune did NOT return normally.
    return code != 0;
#else
    pid_t pid = fork();
    if (pid == 0) {
        execl(self, self, "abort-case", (char *) nullptr);
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    // abort -> terminated by signal (SIGABRT), or non-zero exit.
    if (WIFSIGNALED(status)) {
        return true;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) != 0;
#endif
}

int main(int argc, char ** argv) {
    // Quiet the Windows abort() dialog for the child invocation.
#ifdef _WIN32
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    if (argc > 1 && strcmp(argv[1], "abort-case") == 0) {
        return run_abort_case();
    }

    // ── Happy path: prune keeps bundle A + non-bundle, evicts bundle B ──
    ModelStore * s = store_create(EVICT_STRICT);

    int calls_a_dit = 0, calls_a_text = 0, calls_a_cond = 0;
    int calls_b_dit = 0, calls_b_text = 0;
    int calls_non   = 0;

    store_test_install_synthetic(s, MODEL_DIT_TRT,      BUNDLE_A_DIT,  111, 0, &calls_a_dit);
    store_test_install_synthetic(s, MODEL_TEXT_ENC_TRT, BUNDLE_A_TEXT, 222, 0, &calls_a_text);
    store_test_install_synthetic(s, MODEL_COND_ENC_TRT, BUNDLE_A_COND, 333, 0, &calls_a_cond);
    store_test_install_synthetic(s, MODEL_DIT_TRT,      BUNDLE_B_DIT,  444, 0, &calls_b_dit);
    store_test_install_synthetic(s, MODEL_TEXT_ENC_TRT, BUNDLE_B_TEXT, 555, 0, &calls_b_text);
    store_test_install_synthetic(s, MODEL_DIT,          NON_BUNDLE,    666, 0, &calls_non);

    const size_t total_before = 111 + 222 + 333 + 444 + 555 + 666;
    CHECK(store_gpu_module_count(s) == 6, "6 entries installed");
    CHECK(store_vram_bytes(s) == total_before, "byte total before prune correct");

    store_prune_bundle_except(s, KEEP_BUNDLE);

    // Bundle B (other bundle, refcount 0) must be fully evicted.
    CHECK(!store_test_has_entry(s, MODEL_DIT_TRT,      BUNDLE_B_DIT),  "bundle B dit evicted");
    CHECK(!store_test_has_entry(s, MODEL_TEXT_ENC_TRT, BUNDLE_B_TEXT), "bundle B text evicted");
    CHECK(calls_b_dit == 1,  "bundle B dit deleter ran exactly once");
    CHECK(calls_b_text == 1, "bundle B text deleter ran exactly once");

    // Bundle A (keep) + non-bundle must be untouched.
    CHECK(store_test_has_entry(s, MODEL_DIT_TRT,      BUNDLE_A_DIT),  "bundle A dit kept");
    CHECK(store_test_has_entry(s, MODEL_TEXT_ENC_TRT, BUNDLE_A_TEXT), "bundle A text kept");
    CHECK(store_test_has_entry(s, MODEL_COND_ENC_TRT, BUNDLE_A_COND), "bundle A cond kept (mixed-sep path)");
    CHECK(store_test_has_entry(s, MODEL_DIT,          NON_BUNDLE),    "non-bundle GGUF kept");
    CHECK(calls_a_dit == 0 && calls_a_text == 0 && calls_a_cond == 0, "bundle A deleters never ran");
    CHECK(calls_non == 0, "non-bundle deleter never ran");

    CHECK(store_gpu_module_count(s) == 4, "4 entries remain after prune");
    const size_t total_after = 111 + 222 + 333 + 666;
    CHECK(store_vram_bytes(s) == total_after, "byte total after prune == kept bundle + non-bundle");

    store_free(s);

    // ── Fatal path: other-bundle refcount>0 must abort ──
    CHECK(child_aborts(argv[0]), "prune aborts on refcount>0 other-bundle entry");

    if (g_failures == 0) {
        fprintf(stderr, "\nPASS: store_prune_bundle_except behaves correctly\n");
        return 0;
    }
    fprintf(stderr, "\nFAILED with %d failure(s)\n", g_failures);
    return 1;
}
