"""03_arith_if: arithmetic on sampled numbers, if/then (block body, no else) and if/then/else.

score = b + (a * 3) + bonus, where bonus = 10 iff a = 1. Enumerating the 12
equally likely (a, b) pairs gives P(level = "high") = 5/6.
Wrong precedence, (b + a) * 3 + bonus, would give 11/12; skipping the
else-less if would give 1/2; always applying the bonus would give 1.
"""

from fractions import Fraction

from lumos.testing import assert_rate


def reference():
    high = Fraction(0)
    for a in (1, 2, 3):
        for b in (0, 1, 2, 3):
            bonus = 10 if a == 1 else 0
            score = b + a * 3 + bonus
            if 7 < score:
                high += Fraction(1, 12)
    return high


def externals():
    return {}


def check(t, results, ext):
    (result,) = results
    p = reference()
    t.assertEqual(p, Fraction(5, 6))
    t.assertEqual((result.trials, result.discarded), (1200, 0))
    assert_rate(t, result.successes, result.trials, p, 'P(level = "high")')
