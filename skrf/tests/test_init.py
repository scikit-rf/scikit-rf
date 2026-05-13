"""
Tests that importing skrf does not eagerly load heavy optional modules
that are not required for core functionality.
"""
import subprocess
import sys


def _modules_loaded_by(import_statement: str) -> set[str]:
    """Return the set of module names present in sys.modules after running *import_statement*."""
    code = f"import sys; {import_statement}; print('\\n'.join(sys.modules.keys()))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.splitlines())


# Explicitly named modules that must never be loaded as a side effect of
# `import skrf` (optional / heavy dependencies not required for core use).
# scipy is not included here because it loads submodules lazily. So we need
# to check for any new scipy submodules dynamically instead of maintaining a static list.
UNWANTED_MODULE_PREFIXES = [
    "matplotlib",
    "pandas",
    "IPython",
    "bokeh",
    "PIL",       # Pillow — image processing
    "cv2",       # OpenCV
    "tensorflow",
    "torch",
]


def test_no_heavy_modules_on_import():
    """Importing skrf must not load heavy optional modules.

    For explicitly listed third-party libraries the check is a simple prefix
    match.  For scipy the check is dynamic: any submodule loaded by
    ``import skrf`` that is *not* already loaded by a bare ``import scipy``
    is considered a violation, so the list does not need to be maintained
    manually as new scipy submodules are added.
    """
    scipy_baseline = _modules_loaded_by("import scipy")
    skrf_modules = _modules_loaded_by("import skrf")

    # Scipy submodules present after `import skrf` but not after `import scipy`.
    # Collapse to first-level submodule name only (e.g. scipy.fft._basic -> scipy.fft).
    extra_scipy = {
        ".".join(m.split(".")[:2])
        for m in skrf_modules
        if m.startswith("scipy.") and m not in scipy_baseline
    }

    # Explicitly unwanted third-party modules — collapse to the matched prefix.
    explicit_violations = {
        next(p for p in UNWANTED_MODULE_PREFIXES if m == p or m.startswith(p + "."))
        for m in skrf_modules
        for p in UNWANTED_MODULE_PREFIXES
        if m == p or m.startswith(p + ".")
    }

    violations = sorted(extra_scipy | explicit_violations)

    assert violations == [], (
        "The following heavy/optional modules were unexpectedly loaded by "
        "'import skrf'. Please use lazy-loading techniques to load them only when needed:\n"
        + "\n".join(f"  {m}" for m in violations)
    )
