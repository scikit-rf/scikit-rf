"""Tests for diagnostic interpretation, including incomplete timeout artifacts."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_vectorfit import compare
from vectorfit_probe import loop_location


class DiagnosticComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_probe(self, name, kernel, checkpoints, code=0, threads=4):
        directory = self.root / f"vectorfit-{name}"
        directory.mkdir()
        events = [
            {"event": "threadpools", "libraries": [{"architecture": kernel, "num_threads": threads}]},
            {"event": "parsed_input", "frequency_sha256": "f", "s_sha256": "s"},
            {"event": "source", "sha256": "source"},
        ]
        (directory / "output.log").write_text("\n".join(map(json.dumps, events)))
        (directory / "iterations.jsonl").write_text("\n".join(map(json.dumps, checkpoints)))
        (directory / "result.txt").write_text(f"exit_code={code}\n")
        return directory

    def test_missing_artifacts_do_not_claim_contrast(self):
        result = compare(self.root)
        self.assertFalse(result["kernel_contrast_observed"])
        self.assertFalse(result["same_parsed_input"])

    def test_haswell_on_both_sides_is_only_a_control(self):
        for name in ("default", "haswell"):
            self.write_probe(name, "Haswell", [{"iteration": 0, "model_order": 10}])
        result = compare(self.root)
        self.assertFalse(result["kernel_contrast_observed"])
        self.assertTrue(result["same_parsed_input"])
        self.assertIsNone(result["first_decision_difference"])

    def test_numeric_difference_precedes_decision_difference(self):
        a = [{"iteration": 0, "current_error_peak": 0.1, "model_order": 10},
             {"iteration": 1, "current_error_peak": 0.2, "model_order": 12}]
        b = [{"iteration": 0, "current_error_peak": 0.10000000001, "model_order": 10},
             {"iteration": 1, "current_error_peak": 0.3, "model_order": 14}]
        self.write_probe("default", "SkylakeX", a, code=124)
        self.write_probe("haswell", "Haswell", b)
        result = compare(self.root)
        self.assertTrue(result["kernel_contrast_observed"])
        self.assertEqual(result["first_numeric_difference"]["iteration"], 0)
        self.assertEqual(result["first_decision_difference"]["iteration"], 1)
        self.assertEqual(result["default"]["result"]["exit_code"], "124")

    def test_partial_timeout_record_keeps_last_complete_checkpoint(self):
        d = self.write_probe("default", "SkylakeX", [{"iteration": 0}, {"iteration": 1}], code=124)
        with (d / "iterations.jsonl").open("a") as stream:
            stream.write('\n{"iteration":')
        self.write_probe("haswell", "Haswell", [{"iteration": 0}])
        result = compare(self.root)
        self.assertEqual(result["last_checkpoints"]["default"]["count"], 2)
        self.assertEqual(result["last_checkpoints"]["default"]["last"]["iteration"], 1)

    def test_changed_threads_are_exposed(self):
        self.write_probe("default", "SkylakeX", [], threads=4)
        self.write_probe("haswell", "Haswell", [], threads=1)
        self.assertFalse(compare(self.root)["same_thread_counts"])

    def test_unrecognized_loop_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot identify"):
            loop_location(self.setUp)


if __name__ == "__main__":
    unittest.main()
