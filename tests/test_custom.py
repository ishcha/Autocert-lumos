"""Run a Lumos file of your choice; change SPEC (and WORLD) to try another.

    python -m unittest tests/test_custom.py

WORLD is a Python file whose externals() binds L, judges and inputs; the
mock worlds in tests/specs/ work offline.
"""

import unittest

from lumos_test_utils import REPO
from lumos.cli import execute

SPEC = REPO / "nl2lumos/spec_progs_claude/cat_risk_adaptive.lumos"
WORLD = REPO / "tests/specs/cat_risk_adaptive.py"


class CustomSpecTest(unittest.TestCase):
    def test_custom_spec(self):
        results = execute(SPEC, world=WORLD, trials=50)
        self.assertEqual(results[0].trials, 50)


if __name__ == "__main__":
    unittest.main()
