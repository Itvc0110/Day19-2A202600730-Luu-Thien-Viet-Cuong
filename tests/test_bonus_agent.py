import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class HybridMemoryAgentTests(unittest.TestCase):
    def test_remember_and_recall_include_profile_activity_and_memories(self) -> None:
        from bonus.agent import HybridMemoryAgent

        agent = HybridMemoryAgent(use_external_stores=False)
        agent.remember(
            "Tôi đã đọc tài liệu Kubernetes về autoscaling, pod lifecycle, và cloud security.",
            user_id="u_001",
        )
        agent.remember(
            "Ghi chú: tôi thích ví dụ tiếng Việt ngắn, ưu tiên cloud và AI.",
            user_id="u_001",
        )

        context = agent.recall("summary cloud security", user_id="u_001")

        self.assertIn("User profile", context)
        self.assertIn("Recent activity", context)
        self.assertIn("Top memories", context)
        self.assertIn("cloud", context.lower())
        self.assertIn("Kubernetes", context)

    def test_user_memories_are_filtered_by_user_id(self) -> None:
        from bonus.agent import HybridMemoryAgent

        agent = HybridMemoryAgent(use_external_stores=False)
        agent.remember("Ngọc đọc về Kubernetes và cloud cost.", user_id="u_001")
        agent.remember("Another user read about Flutter mobile offline cache.", user_id="u_002")

        context = agent.recall("Kubernetes cloud", user_id="u_001")

        self.assertIn("Kubernetes", context)
        self.assertNotIn("Flutter", context)

    def test_demo_prints_five_required_queries(self) -> None:
        result = subprocess.run(
            [sys.executable, "bonus/demo.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.count("Query "), 5)
        self.assertIn("What have I read about Kubernetes?", result.stdout)
        self.assertIn("Recommend what to read next", result.stdout)
        self.assertIn("What am I focused on lately?", result.stdout)
        self.assertIn("Documents about scaling infrastructure?", result.stdout)
        self.assertIn("Give me a cloud security summary", result.stdout)


if __name__ == "__main__":
    unittest.main()
