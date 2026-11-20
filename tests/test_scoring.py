"""Exercise the registered terminal-answer rule and paired cost accounting."""

import unittest
from pathlib import Path

from slm_decomposition.common import load_config
from slm_decomposition.score import summarize


class ScoringTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "configs/study.yaml"
        self.config = load_config(path)["evaluation"]
        self.config["bootstrap_draws"] = 16
        self.case = {
            "id": "synthetic",
            "document": "Anna has 9 hats.\nBen has 2 hats.",
            "question": "How many hats does Anna have more than Ben?",
        }
        self.key = {
            "synthetic": {
                "answer": 7,
                "direction": "first_minus_second",
                "children": [
                    {"name": "Anna", "unit": "hats", "value": 9},
                    {"name": "Ben", "unit": "hats", "value": 2},
                ],
            }
        }
        self.row = {"id": "synthetic"}
        for stage, cap in self.config["caps"].items():
            self.row[stage] = {
                "id": "synthetic",
                "stage": stage,
                "cap": cap,
                "text": "[missing generated item]",
                "stop": "missing",
                "eos": False,
                "batch_id": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "amortized_seconds": 0.0,
            }

    def test_terminal_format_caps_and_missing_costs(self):
        direct, review = self.row["direct"], self.row["review"]
        # A complete final answer at the cap is still parsed. EOS alone cannot
        # rescue text whose last nonempty line fails the signed-integer contract.
        direct.update(
            text="Calculation\nAnswer: +7\n",
            stop="cap",
            batch_id="direct",
            input_tokens=50,
            output_tokens=direct["cap"],
            amortized_seconds=0.5,
        )
        review.update(
            text="Answer: 7\nDone.",
            stop="eos",
            eos=True,
            batch_id="review",
            input_tokens=100,
            output_tokens=9,
            amortized_seconds=0.25,
        )
        scores = summarize([self.case], [self.row], self.key, self.config)
        arms = scores["arms"]
        self.assertEqual((arms["direct"]["correct"], arms["direct"]["terminal_cap_hits"]), (1, 1))
        self.assertEqual((arms["review"]["valid"], arms["review"]["correct"]), (0, 0))
        self.assertEqual((arms["review"]["cap_hits"], arms["review"]["terminal_cap_hits"]), (1, 0))
        self.assertEqual((arms["review"]["calls"], arms["review"]["input_tokens"]), (2, 150))
        self.assertEqual(arms["review"]["output_tokens"], direct["cap"] + 9)
        self.assertEqual(arms["review"]["batch_seconds"], 0.75)
        self.assertEqual(
            (arms["full"]["correct"], arms["full"]["valid"], arms["full"]["calls"]), (0, 0, 0)
        )
        paired = scores["paired"]["full_vs_direct"]
        self.assertEqual((paired["gains"], paired["losses"], paired["delta_pp"]), (0, 1, -100))

    def test_unrecorded_stage_is_rejected(self):
        del self.row["left_last"]
        with self.assertRaisesRegex(ValueError, "stages"):
            summarize([self.case], [self.row], self.key, self.config)
