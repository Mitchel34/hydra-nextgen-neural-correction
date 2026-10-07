import unittest

from hydra_correctors.demo import run_demo


class DemoTests(unittest.TestCase):
    def test_replay_and_scoring_are_explicitly_synthetic(self):
        first, second = run_demo(), run_demo()
        self.assertEqual(first, second)
        self.assertFalse(first["scientific_skill_claim"])
        self.assertTrue(first["observation_contract"]["future_value_rejected"])
        score = first["illustrative_fixed_support_score"]
        self.assertEqual(score["primary"], 0.5)
        self.assertEqual(score["registered_rows"], 3)
        self.assertEqual(score["scored_rows"], 3)


if __name__ == "__main__":
    unittest.main()
