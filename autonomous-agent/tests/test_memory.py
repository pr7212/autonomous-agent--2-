import shutil
import tempfile
import unittest
from pathlib import Path

from agent.memory import Checkpoints, Database, TaskHistory
from agent.models import Step, StepStatus, Task, TaskStatus


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db = Database(self.tmp / "t.db")
        self.addCleanup(self.db.close)
        self.history = TaskHistory(self.db)
        self.checkpoints = Checkpoints(self.db)

    def task(self, tid="aaaa1111", **kw):
        return Task(id=tid, objective="do a thing", steps=[Step(1, "write", "write_file", {"path": "a"})], **kw)

    def test_task_roundtrip_keeps_enums_and_steps(self):
        t = self.task()
        t.steps[0].status = StepStatus.VERIFIED
        t.status = TaskStatus.COMPLETED
        self.history.save(t)
        back = self.history.get("aaaa1111")
        self.assertEqual(back.status, TaskStatus.COMPLETED)
        self.assertEqual(back.steps[0].status, StepStatus.VERIFIED)
        self.assertEqual(back.steps[0].args, {"path": "a"})

    def test_save_is_an_upsert(self):
        t = self.task()
        self.history.save(t)
        t.status = TaskStatus.FAILED
        self.history.save(t)
        self.assertEqual(len(self.history.list()), 1)
        self.assertEqual(self.history.get(t.id).status, TaskStatus.FAILED)

    def test_prefix_lookup_and_ambiguity(self):
        self.history.save(self.task("abc11111"))
        self.history.save(self.task("abc22222"))
        self.assertEqual(self.history.find("abc1").id, "abc11111")
        with self.assertRaises(ValueError):
            self.history.find("abc")
        self.assertIsNone(self.history.find("zzz"))

    def test_events_are_ordered(self):
        self.history.log("t1", "a", "first", {"k": 1})
        self.history.log("t1", "b", "second")
        self.history.log("t2", "c", "other task")
        events = self.history.events("t1")
        self.assertEqual([e["kind"] for e in events], ["a", "b"])
        self.assertEqual(events[0]["data"], {"k": 1})

    def test_context_only_includes_completed_work(self):
        done = self.task("done1111", status=TaskStatus.COMPLETED, summary="built it")
        failed = self.task("fail1111", status=TaskStatus.FAILED, summary="nope")
        self.history.save(done)
        self.history.save(failed)
        context = self.history.context_for()
        self.assertIn("built it", context)
        self.assertNotIn("nope", context)
        self.assertEqual(self.history.context_for(exclude="done1111"), "")

    def test_checkpoints_return_latest_and_prune(self):
        t = self.task()
        for i in range(Checkpoints.KEEP + 5):
            t.actions = i
            self.checkpoints.save(t, f"n{i}")
        self.assertEqual(self.checkpoints.count(t.id), Checkpoints.KEEP)
        self.assertEqual(self.checkpoints.latest(t.id).actions, Checkpoints.KEEP + 4)
        self.assertIsNone(self.checkpoints.latest("nope"))


if __name__ == "__main__":
    unittest.main()
