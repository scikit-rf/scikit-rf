# PR #1437: BLAS-kernel-sensitive vector fitting timeout

## Outcome and CI mitigation

A same-runner experiment reproduced an auto_fit timeout with OpenBLAS selecting SkylakeX and completed the same fit after selecting Haswell. Code testing now sets `OPENBLAS_CORETYPE=Haswell` at the Tests job level, before Python imports NumPy, for all five Python versions (3.10-3.14). The original full pytest command and assertions remain unchanged. The matrix retains `fail-fast: false` and the test step's 25-minute limit.

This is an environment workaround, not a change to the fitting algorithm or proof that OpenBLAS is mathematically incorrect. Temporary stack/probe scripts, diagnostic self-tests and diagnostic artifact steps have been removed. The evidence below and the earlier diagnostic commit remain available for review.

## Investigation sequence

1. [Run 34366803271](https://github.com/scikit-rf/scikit-rf/actions/runs/34366803271): Python 3.10 and 3.11 exceeded the six-hour job limit. Exit 143 reflected termination, not the initial cause. Buffered output did not identify the active computation.
2. [Run 34768467753](https://github.com/scikit-rf/scikit-rf/actions/runs/34768467753): external stack snapshots repeatedly located both versions inside test collection, executing `test_dc_enforcement -> auto_fit -> _pole_relocation -> numpy.linalg.qr`. Standalone imports were fast, while single-process collection also exceeded its limit.
3. [Run 34815178623](https://github.com/scikit-rf/scikit-rf/actions/runs/34815178623): 3.10 passed on AMD EPYC 7763 with Haswell; 3.11 timed out on EPYC 9V45 with SkylakeX. A standalone 3.11 fit completed 216,006 QR calls before timeout, and its single-thread variant completed 269,505 calls before timeout. Successful fits required 416 calls. Individual QR calls were fast, so the problem was excessive repeated fitting work, not one blocked QR call. CPU and library differences still confounded this comparison.
4. [Run 34875102901](https://github.com/scikit-rf/scikit-rf/actions/runs/34875102901), diagnostic commit [06f8af98](https://github.com/scikit-rf/scikit-rf/commit/06f8af98382826df01582f48f5479753be903bce): 3.10 failed on Intel Xeon Platinum 8573C. This time the same job ran automatic selection and forced Haswell sequentially in fresh processes, producing the controlled evidence below. Python 3.11-3.14, lint, CodeQL and separate notebook testing passed.

The changing outcome across runs does not make a Python version intrinsically faulty. Hardware and backend selection vary between assigned runners, and both Intel and AMD hosts have exhibited the issue.

## Same-runner experiment

[Tests (3.10)](https://github.com/scikit-rf/scikit-rf/actions/runs/34875102901/job/104111582420) used:

- Intel Xeon Platinum 8573C, four available logical CPUs.
- Python 3.10.21, official setup-python build, GCC 13.3.0.
- NumPy 2.2.6, SciPy 1.15.3; NumPy's OpenBLAS 0.3.29 shared library.
- Four BLAS threads in both probe processes.
- Identical fitting source and parsed frequency/S-parameter hashes.
- The four-port, 601-sample `skrf/tests/cst_example_4ports.s4p` input used by `test_dc_enforcement`.

| Measurement | Automatic SkylakeX | Forced Haswell |
|---|---|---|
| Original full pytest | Timed out after 601.59 s, diagnostic exit 124 | Not rerun in that job |
| Standalone fit | Timed out after 91.37 s | Fit completed in 0.79812 s; process 1.50 s |
| QR calls | 86,592 at last complete loop checkpoint | 416 total |
| Pole relocations | 5,412 at last checkpoint | 26 total |
| Outer-loop checks | 1,804, all continuing | 7, including exit check |
| Final model / RMS error | No completed fit | Order 15 / 0.10711003163326185 |

The ordinary tests ran before the probes. Probe failure preserved the job's failure status; no failures were converted to passes. These results demonstrate the standalone fit workaround; they did not establish a full-suite Haswell pass on that specific runner.

## Why the observed fit did not terminate

The outer loop continues while all three conditions are true:

```python
error_peak > target_error and model_order < model_order_max and delta_eps > alpha
```

At the last SkylakeX checkpoint (index 1803):

| Condition | Value | Outcome |
|---|---|---|
| Error exceeds target | 2.898380853245624 > 0.01 | Continue |
| Model order below limit | 17 < 100 | Continue |
| Error change exceeds threshold | 0.8593627251391232 > 0.03 | Continue |

From checkpoint 4 onward, order remained 17 for 1,800 checks. Late iterations repeatedly removed two pole entries and added two. Tail errors alternated around 1.68-2.55; the model neither grew to its limit nor settled enough to satisfy the change threshold. All recorded pole/error/change values remained finite. The outer loop has no total iteration cap. The finite observation period establishes sustained iteration, not a mathematical proof of an infinite loop.

Haswell exited at checkpoint 6 because `delta_eps=0.02674559084162509 < alpha=0.03`. It did not reach the target error. Its order was 19 at that checkpoint; final skimming produced order 15.

The first recorded numerical difference was already present at checkpoint 0, after initial relocation: errors were 2.898380853245624 vs 2.6378245967491827, and poles/condition numbers differed. The first recorded decision/state difference occurred at checkpoint 2 in the preceding error-band indices: `[600, 431, 361, 275]` vs `[600, 427, 358, 263, 190]`. These observations identify diverging fitting paths, not the first differing low-level floating-point operation.

The loop's `error_peak` variable is initialized before the loop but not refreshed alongside `error_peak_history`. This deserves separate review. However, the minimum recomputed error in the failing trace was 1.5298181279785206, also above 0.01, so updating that variable alone would not stop any of the observed checkpoints.

Collection is an additional issue: the test module executes a unittest suite at import time, so workers do this numerical work before ordinary pytest test execution. Removing that import side effect would improve collection behavior, but the standalone probe demonstrates it would not itself resolve the fitting issue. Neither algorithm nor collection code is changed by this mitigation.

## Local reproduction and validation

WSL Ubuntu 24.04 on an Intel i7-12700H was tested with the official setup-python Python 3.10.21, 3.11.16 and 3.12.14 builds and matching numerical dependency versions. Original full-suite runs passed. Haswell is the local automatic selection. The WSL CPU exposes no AVX-512 capability; forcing SkylakeX caused SIGILL, not the CI timeout, and was not counted as a reproduction.

The diagnostic observer recorded each outer-loop condition without changing locals. Six real probe integrations retained reference RMS results, 416 QR calls and 26 relocations. Diagnostic unit checks covered missing/truncated artifacts, first differences, thread mismatches and same-kernel controls. External-supervisor checks covered success, failure, timeout and child-process cleanup. These temporary checks are removed after collecting their evidence.

The final mitigation is validated with the full original command under `OPENBLAS_CORETYPE=Haswell`:

```bash
export OPENBLAS_CORETYPE=Haswell
export PYTEST_XDIST_AUTO_NUM_WORKERS=2
python -m pytest -n auto --timeout=60 -v \
  --junitxml=test-results/junit-VERSION.xml --junit-prefix=VERSION --dist loadscope
```

Final local full-suite results under forced Haswell:

| Python | Result |
|---|---|
| 3.10 | 1857 passed, 44 skipped, 4 xfailed in 115.79s (0:01:55) |
| 3.11 | 1857 passed, 44 skipped, 4 xfailed in 101.58s (0:01:41) |
| 3.12 | 1857 passed, 44 skipped, 4 xfailed in 99.34s (0:01:39) |

YAML parsing and `git diff --check` passed; the restored test command and dependency-install block match the pre-diagnostic workflow exactly. Local validation covers 3.10-3.12; 3.13/3.14 remain covered by the GitHub matrix. Local source is the PR branch; synthetic GitHub merge runs included one additional test from upstream.

## Evidence and limitations

The decisive run uploaded `CI diagnostics (Python 3.10)` containing `comparison.json`, `cpu.txt`, both probe `output.log` and `iterations.jsonl` files, original/probe result files and timeout stacks. Artifacts have a 14-day retention period; the measurements and checkpoint values above preserve the central findings after expiry. The diagnostic implementation can be recovered from commit 06f8af98.

The two probes ran in fixed order; cache/load effects and observer overhead remain limitations. No claim is made that all SkylakeX workloads fail, that Haswell universally fixes vector fitting, or that the configurable-frequency-unit change caused the numerical issue. Further work should isolate lower-level numerical differences and review algorithm termination independently. This workflow override should be reconsidered when an upstream fix is available.
