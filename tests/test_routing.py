"""A generated carry may be wrong; a fabricated citation must not gain support."""

import itertools
import unittest
from unittest.mock import patch

from slm_decomposition.routing import combined, make_prompt, original_trace
from slm_decomposition.source_check import inspect_work


def chain(names):
    document = [f"{names[0]} has 10 rugs."]
    working = [f"S1: {names[0]} = 10 rugs"]
    amount = 10
    for index, (previous, owner) in enumerate(itertools.pairwise(names), 2):
        sign, relation, delta = ("+", "more", 2) if index % 2 == 0 else ("-", "fewer", -2)
        document.append(f"{owner} has 2 rugs {relation} than {previous}.")
        working.append(f"S{index}: {owner} = {previous} {amount} {sign} 2 = {amount + delta} rugs")
        amount += delta
    first = working[:5] + [f"Value: {names[4]} = 10 rugs"]
    last = [f"S1: {names[4]} = 10 rugs"]
    last += [f"S{i}: " + step.split(": ", 1)[1] for i, step in enumerate(working[5:], 2)]
    last += [f"Value: {names[-1]} = {amount} rugs"]
    return document, {"text": "\n".join(first)}, {"text": "\n".join(last)}


class RoutingTest(unittest.TestCase):
    def setUp(self):
        left, left_first, left_last = chain(
            ["Ada", "Beth", "Chad", "Dale", "Evan", "Finn", "Glen", "Hugh", "Iris"]
        )
        right, right_first, right_last = chain(
            ["Judy", "Kane", "Lara", "Mina", "Noah", "Opal", "Paul", "Quin", "Rita"]
        )
        self.case = {
            "id": "source-scope",
            "document": "\n".join(reversed(left + right)),
            "question": "How many rugs does Iris have more than Rita?",
        }
        self.previous = {
            "plan": {"text": "Q1: Iris | rugs\nQ2: Rita | rugs"},
            "direct": {"text": "Source check: a sentence from the model\nAnswer: 0"},
            "left_first": left_first,
            "right_first": right_first,
            "left_last": left_last,
            "right_last": right_last,
        }

    def check_left(self):
        trace = combined(self.case, self.previous, "left")
        original = original_trace(self.case, trace, "child1", self.previous["plan"]["text"])
        return inspect_work(self.case, original, "Iris", "rugs")

    def test_stage_local_citations(self):
        self.assertEqual(self.check_left()["verified_steps"], 9)
        self.assertEqual(self.check_left()["supported_value"], 10)
        # S6 was unavailable to this first chunk, even though it exists in the full chain.
        self.previous["left_first"]["text"] = self.previous["left_first"]["text"].replace(
            "S5:", "S6:"
        )
        check = self.check_left()
        self.assertGreater(check["invalid_steps"], 0)
        self.assertIsNone(check["supported_value"])

    def test_carry_is_generated_and_typed(self):
        self.previous["left_first"]["text"] = "Value: Evan = 73 rugs"
        self.assertIn("S1: Evan has 73 rugs.", make_prompt("left_last", self.case, self.previous))
        self.previous["left_first"]["text"] = "Value: Evan = 73 clocks"
        self.assertIsNone(make_prompt("left_last", self.case, self.previous))

    def test_unchecked_policies_preserve_model_text(self):
        for policy in ("review", "full"):
            native = make_prompt(policy, self.case, self.previous)
            with patch("slm_decomposition.prompts.inspect_work", side_effect=AssertionError("checker invoked")):
                unchecked = make_prompt(policy + "_no_check", self.case, self.previous)
            retained = [
                line
                for line in native.splitlines()
                if not line.startswith("Source check: ")
                or line == "Source check: a sentence from the model"
            ]
            self.assertEqual(unchecked.splitlines(), retained)
