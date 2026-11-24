"""prime.lumos: multiple-choice drug question over a knowledge graph from getGraph.

Uses the fixture tests/specs/fixtures/primekg.csv (edge-list CSV). A run is
defined when the sampled drug d has a "treat" and a "contraindicate"
neighbour and at least three other drugs remain as distractors; with this
fixture that holds for d1, d2, d3 and d6 out of seven drugs (getEdge on
non-adjacent pairs gives null, which filters those drugs out instead of
failing). The four options are shuffled by sampling without replacement.

Mock world: the model always answers "A)", so it is right iff d was
shuffled to position 0: P = 1/4.
"""

import csv
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from lumos.testing import Recorder, assert_discards, assert_distribution, assert_rate

PROGRAM = "nl2lumos/spec_progs_claude/prime.lumos"
BASE_DIR = "fixtures"
FIXTURE = Path(__file__).parent / "fixtures" / "primekg.csv"
PROMPT = re.compile(
    r"(CTX-d\d)(CTX-x\d)(CTX-x\d)Which drug treats (.*) and is contraindicated for (.*)\?"
    r"A\) .* B\) .* C\) .* D\) .*"
)


def load_fixture():
    attributes, relations = {}, {}
    with open(FIXTURE, newline="") as handle:
        for row in csv.DictReader(handle):
            attributes[row["src_id"]] = [row["src_type"], row["src_name"], row["src_context"]]
            if row["dst_id"]:
                attributes[row["dst_id"]] = [row["dst_type"], row["dst_name"], row["dst_context"]]
                relations[frozenset((row["src_id"], row["dst_id"]))] = row["relation"]
    return attributes, relations


def reference():
    """P(defined) and, given defined, the distribution of (d, dis1, dis2)."""
    attributes, relations = load_fixture()
    drugs = [n for n, a in attributes.items() if a[0] == "drug"]
    rel = lambda a, b: relations.get(frozenset((a, b)))
    defined = {}
    for d in drugs:
        treats = [x for x in attributes if rel(d, x) == "treat"]
        contra = [x for x in attributes if rel(d, x) == "contraindicate"]
        for dis1 in treats:
            for dis2 in contra:
                correct = {v for v in drugs if rel(v, dis1) == "treat" and rel(v, dis2) == "contraindicate"}
                if len(set(drugs) - correct) >= 3:
                    p = Fraction(1, len(drugs)) / (len(treats) * len(contra))
                    defined[(d, dis1, dis2)] = defined.get((d, dis1, dis2), 0) + p
    p_defined = sum(defined.values())
    return attributes, p_defined, {k: v / p_defined for k, v in defined.items()}


def externals():
    return {"L": Recorder(lambda prompt: "A)"), "J_QA": Recorder(lambda response, gt: 1 if response == gt else 0)}


def check(t, results, ext):
    (result,) = results
    attributes, p_defined, scenarios = reference()
    t.assertEqual(p_defined, Fraction(4, 7))
    t.assertEqual(result.trials, 1000)
    prompts, judged = ext["L"].calls, ext["J_QA"].calls
    t.assertEqual((len(prompts), len(judged)), (result.trials, result.trials))  # ⊥ always strikes before L

    seen, substituted = Counter(), Counter()
    for prompt in prompts:
        match = PROMPT.fullmatch(prompt)
        t.assertIsNotNone(match, prompt)
        d, dis1, dis2 = (m.removeprefix("CTX-") for m in match.groups()[:3])
        seen[(d, dis1, dis2)] += 1
        # format(...) replaced {dis1} and {dis2} with one of their attributes.
        t.assertIn(match.group(4), attributes[dis1])
        t.assertIn(match.group(5), attributes[dis2])
        substituted[attributes[dis1].index(match.group(4))] += 1

    assert_distribution(t, seen, scenarios, result.trials, "(d, dis1, dis2)")
    assert_distribution(t, substituted, {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}, result.trials, "format attribute")
    assert_distribution(t, Counter(gt for _, gt in judged), {"A)": 0.25, "B)": 0.25, "C)": 0.25, "D)": 0.25},
                        result.trials, "gt_label")
    assert_rate(t, result.successes, result.trials, 0.25, "P(correct)")
    assert_discards(t, result, p_defined)
