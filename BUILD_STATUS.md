# Fork build status — verified on Kaggle CPUs (2026-11-24)

Kernel `jprinceraj/inkengine-fork-build`, fork `main` @ `646c1a5`, GCC 13.3,
CMake with `GGML_CUDA=OFF`, `LLAMA_BUILD_TESTS=OFF`, `LLAMA_CURL=OFF`.

| Check | Result |
|---|---|
| Clone `-b main` and component presence | PASS |
| CMake configure + build target `llama-cli` | **BUILD OK** |
| `llama-cli --version` | `1.0.0-dev (build 1, commit 646c1a5)` |
| `tools/roret` ctest | PASS (1/1 ctest: test_union_cache) |
| `tools/pibs` ctest | rc 0 |
| `tools/odsl` ctest | rc 0 |
| Overall kernel | **COMPLETE / KAGGLE FORK BUILD PASS** |

Fixes required along the way (all committed):
- kernel cloned upstream default branch `master` instead of our `main` → `-b main`;
- ODSL depended on the sibling `inkengine` repo → LUT and vectors vendored here;
- upstream `*.bin` ignore hid the vectors → force-added; local build dirs untracked.

`tools/roret/test_union_cache.cpp` is now compiled and registered with ctest:
clean default and Release builds both report 1/1 real test, and `-UNDEBUG`
keeps its `assert()` calls active in Release (Release would otherwise define
`NDEBUG` and silently disable the test oracle).

The fork is buildable and all loop-specific prototypes are independently
verifiable on a fresh machine. No speedup is claimed yet; see each tool's
RESULTS.md for the honest numeric models.

## Re-verified 2026-11-24 (kernel v6, HEAD 149a963)

- Full `llama-cli` build: OK
- roret / pibs / odsl ctest: rc 0 each
- audit fixture + gate verifier: 13 passed in 10.30s
- Overall: **COMPLETE / KAGGLE FORK BUILD PASS**
- Note: audit tests require loopkit cloned at `/tmp/loopkit/src`; the kernel now does that.
