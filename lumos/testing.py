"""Helpers for writing worlds and spec checks (tests/specs/*.py, tests/heldout/*.py)."""

import math
from collections import defaultdict
from fractions import Fraction


class Recorder:
    """Wraps a mock external and records every call, shared by all trials."""

    __lumos_shared_callable__ = True  # do not deep-copy per trial

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args if len(args) > 1 else args[0])
        return self.fn(*args)


def assert_rate(test, successes, trials, p, what="rate", sigmas=4.0):
    """successes/trials is within `sigmas` binomial standard deviations of p."""
    p = float(p)
    tolerance = sigmas * math.sqrt(p * (1 - p) / trials) + 1e-9
    test.assertLessEqual(
        abs(successes / trials - p), tolerance,
        f"{what}: {successes}/{trials} = {successes / trials:.4f}, expected {p:.4f} ± {tolerance:.4f}",
    )


def assert_discards(test, result, p_defined, sigmas=4.0):
    """Discarded runs before result.trials defined ones match P(defined) (negative binomial)."""
    p = float(p_defined)
    mean = result.trials * (1 - p) / p
    sd = math.sqrt(result.trials * (1 - p)) / p
    test.assertLessEqual(
        abs(result.discarded - mean), sigmas * sd + 1e-9,
        f"discarded {result.discarded} runs, expected {mean:.1f} ± {sigmas * sd:.1f}",
    )


def assert_distribution(test, counts, expected, total, what="distribution", sigmas=4.0):
    """Every outcome's count is within `sigmas` binomial sds of expected[outcome] * total."""
    unexpected = set(counts) - set(expected)
    test.assertFalse(unexpected, f"{what}: unexpected outcomes {sorted(unexpected)[:5]}")
    for outcome, p in expected.items():
        assert_rate(test, counts.get(outcome, 0), total, p, f"{what}[{outcome}]", sigmas)


class Distribution:
    """Exact finite distribution, with mass for ⊥ (undefined scenarios)."""

    def __init__(self):
        self.mass = defaultdict(Fraction)
        self.undefined = Fraction(0)

    def add(self, outcome, p):
        self.mass[outcome] += p

    @property
    def defined(self):
        return 1 - self.undefined

    def conditional(self, predicate):
        """P(predicate(outcome) | defined)."""
        return sum((p for o, p in self.mass.items() if predicate(o)), Fraction(0)) / self.defined

    def conditional_marginal(self, key):
        result = defaultdict(Fraction)
        for outcome, p in self.mass.items():
            result[key(outcome)] += p / self.defined
        return dict(result)
