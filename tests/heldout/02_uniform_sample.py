"""02_uniform_sample: sample(meas(S)) over a literal set is uniform by default.

budget is uniform on {10, 20, 30, 40, 50}, so P(budget < 25) = 2/5.
Always taking the first element would give 1, the last 0.
"""

from fractions import Fraction

from lumos.testing import assert_rate

BUDGETS = [10, 20, 30, 40, 50]


def externals():
    return {}


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded), (1000, 0))
    t.assertEqual(len(result.observations), 1000)
    t.assertTrue(all(o in (True, False) for o in result.observations))
    t.assertEqual(result.successes, sum(1 for o in result.observations if o))
    p = Fraction(sum(1 for b in BUDGETS if b < 25), len(BUDGETS))
    assert_rate(t, result.successes, result.trials, p, "P(budget < 25)")
