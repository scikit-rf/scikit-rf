# Temporary diagnostics for PR #1437

The original pytest command remains the regression test. On Python 3.10/3.11 its supervisor limits runtime to ten minutes and captures external stacks before killing the process group. Failure status is preserved. No fitting algorithm or test assertions are changed.

Afterward, Python 3.10/3.11/3.12 each run two standalone auto_fit probes on the same runner: automatic BLAS selection and forced Haswell. Each uses a fresh process, identical inherited thread settings and a 90-second limit. Diagnostics are uploaded even on failure. The standalone probes are numerical experiments, not replacements for the original test assertions.

## Read the artifacts

- cpu.txt: actual CPU and exposed instruction capabilities.
- vectorfit-{default,haswell}/output.log: Python/NumPy/SciPy, actual BLAS library/version/kernel/thread count, source and input hashes, QR progress and final result.
- iterations.jsonl in each probe directory: every outer-loop condition check, including the exit check. Fields include the condition's error_peak, recomputed current_error_peak, recent error history, delta_eps/alpha, model_order/limit, poles, condition number, rank deficiency, QR counts and relocation counts. n_skim/n_add at a checkpoint describe the preceding iteration, when present. Writes are line-buffered so completed checkpoints survive termination. Final cleanup/skimming after the outer loop can change final model order.
- comparison.json: actual kernel contrast, matching input/source/thread counts, first exact numerical difference, first recorded decision/state difference, and the last checkpoint from each side. A numerical difference is not automatically significant. Unequal trace lengths are retained, not treated as equal convergence.
- result.txt and snapshot/timeout directories: exit status, timing, external stacks and processes.

If both processes select Haswell, the job is a control; it cannot establish SkylakeX behavior. Unknown observer data, source/input differences and changed thread counts must be resolved before attribution. If automatic SkylakeX times out while Haswell completes, this supports a kernel-sensitive trigger, not necessarily an OpenBLAS bug. The two probes run in a fixed order, and tracing adds overhead. Check the original untraced test outcome as well.

The observer locates auto_fit's outer while condition from its AST and traces only that Python frame. It records locals without modifying them; the numerical function and QR wrapper are restored on normal exit or exceptions. Unsupported source layouts fail explicitly instead of silently producing misleading logs.

## Validation

Run `python -m unittest discover -s .github/scripts/tests -v` for missing artifacts, timeout/truncated output, first-divergence detection, thread mismatch, same-kernel controls and unsupported source layouts. The workflow runs these checks before the probes. Local integration also checks exact QR counts/final numerical results against previous untraced runs, genuine timeout/child cleanup, and the full original pytest suite on Python 3.10-3.12.
