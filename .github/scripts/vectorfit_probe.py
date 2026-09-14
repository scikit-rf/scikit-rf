"""Temporary numerical probe for the input used by test_dc_enforcement."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def emit(event, **values):
    sys.stdout.write(json.dumps({"event": event, **values}) + "\n")
    sys.stdout.flush()


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

    def observed(matrix, *args, **kwargs):
        nonlocal count, qr_seconds
        count += 1
        emit("qr_start", call=count, shape=list(matrix.shape), dtype=str(matrix.dtype))
        started = time.perf_counter()
        result = original(matrix, *args, **kwargs)
        elapsed = time.perf_counter() - started
        qr_seconds += elapsed
        emit("qr_end", call=count, seconds=elapsed)
        return result

    np.linalg.qr = observed
    started = time.perf_counter()
    emit("fit_start", nports=network.nports, samples=len(network.f))
    try:
        fitting.auto_fit()
    finally:
        np.linalg.qr = original
        emit(
            "fit_stopped",
            seconds=time.perf_counter() - started,
            qr_calls=count,
            completed_qr_seconds=qr_seconds,
            relocations=len(fitting.d_res_history),
        )
    emit(
        "fit_result",
        model_order=int(fitting.get_model_order(fitting.poles)),
        rms_error=float(fitting.get_rms_error()),
    )


if __name__ == "__main__":
    main()
