"""Temporary Linux CI diagnostics; run the original command before any probes."""
# ruff: noqa: T201 -- this command-line diagnostic intentionally prints CI logs.

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def capture(command, destination, timeout=20):
    with destination.open("w") as log:
        try:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            log.write(f"\nDiagnostic exit code: {result.returncode}\n")
        except (OSError, subprocess.TimeoutExpired) as error:
            log.write(f"\nDiagnostic unavailable: {error}\n")


def snapshot(pid, directory):
    directory.mkdir(parents=True, exist_ok=True)
    capture(
        ["ps", "-eo", "pid,ppid,pgid,stat,pcpu,etime,wchan:32,args", "--forest"],
        directory / "processes.txt",
    )
    # Inspect the live parent and xdist workers without a timer inside pytest.
    capture(
        [
            "sudo",
            "-n",
            shutil.which("py-spy") or "py-spy",
            "dump",
            "--pid",
            str(pid),
            "--subprocesses",
            "--nonblocking",
        ],
        directory / "stacks.txt",
    )


def stop_group(process):
    # Also clean up workers if the parent has already exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            time.sleep(1)
    process.wait()


def monitored(command, directory, limit, interval, env=None):
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Starting: {shlex.join(command)}", flush=True)
    started = time.monotonic()
    next_snapshot = interval
    timed_out = False
    with (directory / "output.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= limit:
                    timed_out = True
                    snapshot(process.pid, directory / "timeout")
                    break
                if elapsed >= next_snapshot:
                    print(f"Still running: {elapsed:.0f}s; taking snapshot", flush=True)
                    snapshot(process.pid, directory / f"snapshot-{int(elapsed)}")
                    next_snapshot = time.monotonic() - started + interval
                time.sleep(0.5)
        finally:
            stop_group(process)
    code = 124 if timed_out else process.returncode
    (directory / "result.txt").write_text(
        f"command={shlex.join(command)}\nexit_code={code}\n"
        f"diagnostic_timeout={timed_out}\nelapsed={time.monotonic() - started:.2f}s\n"
    )
    print((directory / "output.log").read_text(errors="replace"), flush=True)
    print(f"Finished: exit={code}, diagnostic_timeout={timed_out}", flush=True)
    return code, timed_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    root = Path("ci-diagnostics")
    root.mkdir(exist_ok=True)
    if args.probe_only:
        capture(["lscpu"], root / "cpu.txt")
        command = [sys.executable, str(Path(__file__).with_name("vectorfit_probe.py"))]
        codes = []
        # Fresh processes on the same runner: only kernel selection changes
        # within each pair. Never force AVX-512 on unsupported hardware.
        for name in ("default", "haswell"):
            env = os.environ.copy()
            env.pop("OPENBLAS_CORETYPE", None)
            if name.startswith("haswell"):
                env["OPENBLAS_CORETYPE"] = "Haswell"
            env["CI_ITERATIONS_PATH"] = str(root / f"vectorfit-{name}" / "iterations.jsonl")
            code, _ = monitored(command, root / f"vectorfit-{name}", 90, 30, env=env)
            codes.append(code)
        import json

        from compare_vectorfit import compare

        comparison = compare(root)
        (root / "comparison.json").write_text(json.dumps(comparison, indent=2))
        print(json.dumps(comparison, indent=2), flush=True)
        return int(any(code != 0 for code in codes))
    (root / "environment.txt").write_text(
        f"Python: {sys.version}\nExecutable: {sys.executable}\n"
        f"CPU count: {os.cpu_count()}\n"
        + "".join(
            f"{key}={os.environ.get(key, '')}\n" for key in ("ImageOS", "ImageVersion", "RUNNER_OS", "RUNNER_ARCH")
        )
    )
    capture(["uname", "-a"], root / "kernel.txt")
    capture(["uv", "pip", "freeze"], root / "dependencies.txt")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "auto",
        "--timeout=60",
        "-v",
        f"--junitxml=test-results/junit-{args.python_version}.xml",
        f"--junit-prefix={args.python_version}",
        "--dist",
        "loadscope",
    ]
    code, _ = monitored(command, root / "original", 600, 120)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())
