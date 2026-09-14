"""Temporary numerical probe for the input used by test_dc_enforcement."""

import ast
import hashlib
import inspect
import json
import os
import sys
import textwrap
import time
from pathlib import Path


def emit(event, **values):
    sys.stdout.write(json.dumps({"event": event, **values}) + "\n")
    sys.stdout.flush()


def loop_location(function):
    """Find the actual outer stopping condition, without fixed source lines."""
    lines, start = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    matches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.While)
        and {"error_peak", "model_order", "delta_eps"}.issubset(
            {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}
        )
    ]
    if len(matches) != 1:
        raise RuntimeError("Cannot identify auto_fit outer loop; update diagnostic observer")
    return start + matches[0].lineno - 1


def main():
    # Import the observer from an isolated directory, outside the test venv.
    observer = os.environ.get("CI_OBSERVER_PATH")
    if observer:
        sys.path.insert(0, observer)
    import numpy as np
    import scipy

    import skrf
    from skrf.vectorFitting import VectorFitting

    emit(
        "environment",
        python=sys.version,
        numpy=np.__version__,
        scipy=scipy.__version__,
        skrf=skrf.__file__,
        affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_CORETYPE",
            )
        },
    )
    try:
        from threadpoolctl import threadpool_info

        emit("threadpools", libraries=threadpool_info())
    except ImportError as error:
        emit("observer_unavailable", reason=str(error))

    data = Path("skrf/tests/cst_example_4ports.s4p")
    emit("input", sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    network = skrf.Network(data)
    fitting = VectorFitting(network)
    original = np.linalg.qr
    count = 0
    qr_seconds = 0.0
    next_progress = time.perf_counter() + 5

    def observed(matrix, *args, **kwargs):
        nonlocal count, qr_seconds, next_progress
        count += 1
        if count == 1:
            emit("qr_start", call=count, shape=list(matrix.shape), dtype=str(matrix.dtype))
        started = time.perf_counter()
        result = original(matrix, *args, **kwargs)
        elapsed = time.perf_counter() - started
        qr_seconds += elapsed
        # Keep exact counters, but avoid hundreds of thousands of log lines.
        now = time.perf_counter()
        if now >= next_progress:
            emit(
                "qr_progress",
                completed_calls=count,
                shape=list(matrix.shape),
                completed_qr_seconds=qr_seconds,
                relocations=len(fitting.d_res_history),
            )
            next_progress = now + 5
        return result

    # Trace only auto_fit's Python frame. Never change its locals or stopping rule.
    loop_line = loop_location(VectorFitting.auto_fit)
    source = Path(inspect.getsourcefile(VectorFitting.auto_fit))
    emit("source", sha256=hashlib.sha256(source.read_bytes()).hexdigest(), loop_line=loop_line)
    emit(
        "parsed_input",
        frequency_sha256=hashlib.sha256(network.f.tobytes()).hexdigest(),
        s_sha256=hashlib.sha256(network.s.tobytes()).hexdigest(),
    )
    iteration = 0
    trace_path = Path(os.environ.get("CI_ITERATIONS_PATH", "iterations.jsonl"))
    previous_trace = sys.gettrace()
    with trace_path.open("w", buffering=1) as trace_log:
        def trace(frame, event, arg):
            nonlocal iteration
            if frame.f_code is not VectorFitting.auto_fit.__code__:
                return None
            if event == "line" and frame.f_lineno == loop_line:
                state = frame.f_locals
                conditions = {
                    "error_above_target": bool(state["error_peak"] > state["target_error"]),
                    "order_below_limit": bool(state["model_order"] < state["model_order_max"]),
                    "change_above_alpha": bool(state["delta_eps"] > state["alpha"]),
                }
                record = {
                    "iteration": iteration,
                    "seconds": time.perf_counter() - started,
                    "qr_calls": count,
                    "relocations": len(fitting.d_res_history),
                    "conditions": conditions,
                    "continue_loop": all(conditions.values()),
                    "poles": [[float(p.real), float(p.imag)] for p in state["poles"]],
                    "error_history_tail": [float(x) for x in state["error_peak_history"][-5:]],
                }
                for key in ("error_peak", "target_error", "model_order", "model_order_max",
                            "delta_eps", "alpha", "n_skim", "n_add", "cond", "rank_deficiency"):
                    if key in state:
                        record[key] = float(state[key])
                record["current_error_peak"] = float(np.max(state["delta"]))
                record["finite"] = {
                    "poles": bool(np.all(np.isfinite(state["poles"]))),
                    "error": bool(np.all(np.isfinite(state["delta"]))),
                    "delta_eps": bool(np.isfinite(state["delta_eps"])),
                }
                if "spurious" in state:
                    record["previous_spurious_mask"] = [bool(x) for x in state["spurious"]]
                if "idx_freqs_max" in state:
                    record["previous_error_band_indices"] = [int(x) for x in state["idx_freqs_max"]]
                trace_log.write(json.dumps(record) + "\n")
                iteration += 1
            return trace

        np.linalg.qr = observed
        started = time.perf_counter()
        emit("fit_start", nports=network.nports, samples=len(network.f))
        try:
            sys.settrace(trace)
            fitting.auto_fit()
        finally:
            sys.settrace(previous_trace)
            np.linalg.qr = original
            emit(
                "fit_stopped",
                seconds=time.perf_counter() - started,
                qr_calls=count,
                completed_qr_seconds=qr_seconds,
                relocations=len(fitting.d_res_history),
                loop_checkpoints=iteration,
            )
    emit(
        "fit_result",
        model_order=int(fitting.get_model_order(fitting.poles)),
        rms_error=float(fitting.get_rms_error()),
    )


if __name__ == "__main__":
    main()
