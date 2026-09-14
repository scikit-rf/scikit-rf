"""Summarize same-runner probe evidence without claiming a root cause."""

import json
from pathlib import Path


def records(path):
    if not path.exists():
        return
    with path.open() as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except ValueError:
                # A killed process can leave a partial final line.
                continue


def read_probe(directory):
    events = {event["event"]: event for event in records(directory / "output.log")}
    result = {}
    if (directory / "result.txt").exists():
        result = dict(line.split("=", 1) for line in (directory / "result.txt").read_text().splitlines()
                      if "=" in line)
    pools = events.get("threadpools", {}).get("libraries", [])
    return {
        "result": result,
        "environment": events.get("environment"),
        "parsed_input": events.get("parsed_input"),
        "source": events.get("source"),
        "kernels": [pool.get("architecture") for pool in pools],
        "threads": [pool.get("num_threads") for pool in pools],
        "libraries": pools,
        "fit": events.get("fit_stopped"),
        "fit_result": events.get("fit_result"),
    }


def compare(root):
    left = read_probe(root / "vectorfit-default")
    right = read_probe(root / "vectorfit-haswell")
    verified = bool(left["kernels"] and right["kernels"])
    contrast = verified and left["kernels"] != right["kernels"]
    output = {
        "default": left,
        "haswell": right,
        "kernel_contrast_observed": bool(contrast),
        "requested_haswell_verified": bool(right["kernels"]) and all(k == "Haswell" for k in right["kernels"]),
        "same_parsed_input": left["parsed_input"] is not None and left["parsed_input"] == right["parsed_input"],
        "same_source": left["source"] is not None and left["source"] == right["source"],
        "same_thread_counts": verified and left["threads"] == right["threads"],
        "first_numeric_difference": None,
        "first_decision_difference": None,
        "last_checkpoints": {},
        "note": "Observational comparison; differences do not establish an OpenBLAS bug.",
    }
    numeric = ("error_peak", "current_error_peak", "delta_eps", "poles", "cond")
    decisions = (
        "conditions", "n_skim", "n_add", "model_order", "previous_spurious_mask", "previous_error_band_indices",
    )
    paths = [root / f"vectorfit-{name}" / "iterations.jsonl" for name in ("default", "haswell")]
    for a, b in zip(records(paths[0]), records(paths[1])):
        for category, keys in (("numeric", numeric), ("decision", decisions)):
            field = f"first_{category}_difference"
            changed = [key for key in keys if a.get(key) != b.get(key)]
            if output[field] is None and changed:
                output[field] = {"iteration": a["iteration"], "fields": changed,
                                 "default": {key: a.get(key) for key in changed},
                                 "haswell": {key: b.get(key) for key in changed}}
    for name, path in zip(("default", "haswell"), paths):
        last = None
        count = 0
        for record in records(path):
            last = record
            count += 1
        output["last_checkpoints"][name] = {"count": count, "last": last}
    return output


if __name__ == "__main__":
    root = Path("ci-diagnostics")
    (root / "comparison.json").write_text(json.dumps(compare(root), indent=2))
