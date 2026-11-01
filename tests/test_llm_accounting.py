"""Verify real retry paths preserve reproducibility without logging credentials."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent.imitation_based.llm_integration import GeneralAPIInterface, reset_llm_cache_sequence, run_llm


class LLMAccountingTests(unittest.TestCase):
    def test_independent_samples_replay_without_population_collapse(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"LLM_CACHE_DIR": directory, "LLM_CACHE_MODE": "readwrite",
                   "LLM_CACHE_KEY_MODE": "call_sequence", "LLM_BACKEND": "",
                   "LLM_USAGE_LOG": "", "LLM_TRACE_DIR": "", "LLM_EXTRA_HEADERS": ""}
            def generate():
                return run_llm("identical prompt", api_endpoint="example.invalid",
                               api_key="fake-test-key", model="test-model", temperature=.8)
            with patch.dict(os.environ, env), patch.object(
                GeneralAPIInterface, "get_response", side_effect=["sample A", "sample B", "other source"]
            ) as api:
                reset_llm_cache_sequence("source 1")
                self.assertEqual([generate(), generate()], ["sample A", "sample B"])
                reset_llm_cache_sequence("source 2")
                self.assertEqual(generate(), "other source")
                os.environ["LLM_CACHE_MODE"] = "readonly"
                reset_llm_cache_sequence("source 1")
                self.assertEqual([generate(), generate()], ["sample A", "sample B"])
                self.assertEqual(api.call_count, 3)
                self.assertEqual(len(list(Path(directory).glob('*.json'))), 3)

    def test_retry_archives_full_responses_usage_and_redacts_key(self):
        key = "fake-private-key-for-test"
        bad = Mock(status=429)
        bad.read.return_value = json.dumps({"error": {"message": "denied " + key}}).encode()
        answer = {"choices": [{"message": {"content": "def compute_utility(): pass"}}],
                  "usage": {"total_tokens": 12}, "model": "test-model", "cost": 0.01}
        good = Mock(status=200)
        good.read.return_value = json.dumps(answer).encode()
        connections = [Mock(), Mock()]
        for conn, response in zip(connections, (bad, good)):
            conn.getresponse.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {"LLM_TRACE_DIR": str(root / "traces"), "LLM_USAGE_LOG": str(root / "usage.jsonl")}
            with patch.dict(os.environ, env), patch(
                "agent.imitation_based.llm_integration._open_http_connection", side_effect=connections
            ), contextlib.redirect_stdout(io.StringIO()) as log:
                result = GeneralAPIInterface("example.invalid", key, "test-model", debug_mode=True).get_response("source prompt")
            self.assertEqual(result, answer["choices"][0]["message"]["content"])
            traces = [json.loads(p.read_text()) for p in sorted((root / "traces").glob("*.json"))]
            self.assertEqual([t["status"] for t in traces], ["error", "success"])
            self.assertEqual(traces[0]["request"]["messages"][0]["content"], "source prompt")
            self.assertEqual(json.loads(traces[1]["response_text"]), answer)
            events = [json.loads(line) for line in (root / "usage.jsonl").read_text().splitlines()]
            self.assertEqual(events[1]["usage"]["total_tokens"], 12)
            self.assertEqual(events[1]["request_id"], traces[1]["request_id"])
            self.assertNotIn(key, log.getvalue() + ''.join(p.read_text() for p in root.rglob("*") if p.is_file()))
            for conn in connections:
                conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
