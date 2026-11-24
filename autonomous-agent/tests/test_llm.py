import unittest

from agent.llm import LLMError, Router, extract_json
from tests.helpers import ScriptedLLM


class ExtractJsonTests(unittest.TestCase):
    def test_plain_fenced_and_chatty(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('Sure!\n```json\n{"a": 2}\n```\nDone.'), {"a": 2})
        self.assertEqual(extract_json('Here you go: {"a": {"b": [1, 2]}} hope it helps'), {"a": {"b": [1, 2]}})

    def test_garbage_raises(self):
        for bad in ("", "no json here", "[1, 2, 3]", '{"unclosed": '):
            with self.assertRaises(ValueError, msg=bad):
                extract_json(bad)


class RouterTests(unittest.TestCase):
    def providers(self):
        self.local = ScriptedLLM({"PLAN": [{"who": "local"}], "RESOLVE_ARGS": [{"who": "local"}]})
        self.cloud = ScriptedLLM({"PLAN": [{"who": "cloud"}], "RESOLVE_ARGS": [{"who": "cloud"}]})
        self.local.name, self.cloud.name = "local", "cloud"

    def ask(self, router, role_tag="PLAN", role="plan"):
        return router.complete_json(role, f"ROLE: {role_tag}", "go")["who"]

    def test_hybrid_sends_heavy_work_to_cloud_and_light_work_to_local(self):
        self.providers()
        router = Router(self.local, self.cloud, "hybrid")
        self.assertEqual(self.ask(router, "PLAN", "plan"), "cloud")
        self.assertEqual(self.ask(router, "RESOLVE_ARGS", "step"), "local")

    def test_pinned_modes(self):
        self.providers()
        self.assertEqual(self.ask(Router(self.local, self.cloud, "local")), "local")
        self.providers()
        self.assertEqual(self.ask(Router(self.local, self.cloud, "cloud")), "cloud")

    def test_local_only_never_escalates_to_cloud(self):
        self.providers()
        router = Router(self.local, self.cloud, "hybrid")
        result = router.complete_json("plan", "ROLE: PLAN", "screen state", local_only=True)
        self.assertEqual(result["who"], "local")
        self.assertEqual(self.cloud.calls, [])

    def test_local_only_reports_missing_ollama_instead_of_using_cloud(self):
        self.providers()
        router = Router(None, self.cloud, "hybrid")
        with self.assertRaisesRegex(LLMError, "local Ollama"):
            router.complete_json("plan", "ROLE: PLAN", "screen state", local_only=True)
        self.assertEqual(self.cloud.calls, [])

    def test_invalid_json_twice_escalates_to_the_other_provider(self):
        self.providers()
        self.cloud.script["PLAN"] = ["not json", "still not json"]
        router = Router(self.local, self.cloud, "hybrid")
        self.assertEqual(self.ask(router), "local")
        self.assertEqual(router.last_provider, "local")

    def test_second_attempt_can_recover_with_a_nudge(self):
        self.providers()
        self.local.script["PLAN"] = ["oops", {"who": "local"}]
        router = Router(self.local, None, "local")
        self.assertEqual(self.ask(router), "local")
        self.assertIn("ONLY the JSON", self.local.calls[1][1])

    def test_provider_outage_escalates_without_retrying_it(self):
        self.providers()
        self.cloud.script["PLAN"] = [LLMError("connection refused")]
        router = Router(self.local, self.cloud, "hybrid")
        self.assertEqual(self.ask(router), "local")
        self.assertEqual(len(self.cloud.calls), 1)

    def test_required_fields_are_enforced(self):
        self.providers()
        router = Router(self.local, None, "local")
        with self.assertRaises(LLMError):
            router.complete_json("plan", "ROLE: PLAN", "go", required=("steps",))

    def test_no_provider_available(self):
        router = Router(None, None, "hybrid")
        self.assertFalse(router.available())
        with self.assertRaises(LLMError):
            router.complete_json("plan", "ROLE: PLAN", "go")


if __name__ == "__main__":
    unittest.main()
