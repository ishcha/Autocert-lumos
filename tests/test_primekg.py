"""prime.lumos on the full PrimeKG (skipped unless data/primekg/primekg.csv exists).

    python scripts/primekg_to_lumos.py data/primekg     # creates the CSV
    python -m unittest tests/test_primekg.py            # about three minutes

The reference model reads the CSV directly (not through the interpreter) to
compute the probability that a sampled scenario is defined.
"""

import csv
import sys
import unittest
from collections import Counter, defaultdict
from fractions import Fraction

from lumos_test_utils import REPO
from lumos.cli import execute
from lumos.testing import Recorder, assert_discards, assert_distribution, assert_rate

DATA = REPO / "data/primekg/primekg.csv"


def reference_p_defined():
    """P(d has a treat and a contraindicate neighbour and >= 3 distractors remain)."""
    csv.field_size_limit(sys.maxsize)
    drugs, treats, contra = set(), defaultdict(set), defaultdict(set)
    with open(DATA, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for side in ("src", "dst"):
                if row[f"{side}_type"] == "drug":
                    drugs.add(row[f"{side}_id"])
            for drug, other in ((row["src_id"], row["dst_id"]), (row["dst_id"], row["src_id"])):
                if row["relation"] == "treat":
                    treats[drug].add(other)
                elif row["relation"] == "contraindicate":
                    contra[drug].add(other)
    total = Fraction(0)
    for d in drugs:
        t = [x for x in treats[d] if x not in drugs]
        c = [x for x in contra[d] if x not in drugs]
        for dis1 in t:
            for dis2 in c:
                correct = sum(1 for v in drugs if dis1 in treats[v] and dis2 in contra[v])
                if len(drugs) - correct >= 3:
                    total += Fraction(1, len(drugs) * len(t) * len(c))
    return total


@unittest.skipUnless(DATA.exists(), "run scripts/primekg_to_lumos.py data/primekg first")
class FullPrimeKGTest(unittest.TestCase):
    def test_prime_on_full_primekg(self):
        p_defined = reference_p_defined()
        judge = Recorder(lambda reply, label: 1 if reply == label else 0)
        (result,) = execute(REPO / "nl2lumos/spec_progs_claude/prime.lumos", trials=200, seed=1,
                            world=REPO / "worlds/prime_primekg.py",
                            externals={"L": lambda prompt: "A)", "J_QA": judge})
        self.assertEqual(result.trials, 200)
        assert_discards(self, result, p_defined)
        # The model always answers "A)": right iff the shuffle put the drug first.
        assert_rate(self, result.successes, result.trials, 0.25, "P(correct)")
        assert_distribution(self, Counter(label for _, label in judge.calls),
                            {"A)": 0.25, "B)": 0.25, "C)": 0.25, "D)": 0.25}, result.trials, "gt_label")


if __name__ == "__main__":
    unittest.main()
