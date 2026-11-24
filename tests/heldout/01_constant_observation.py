"""01_constant_observation: estimateProb with a constant, always-true observation.

No externals and no randomness. All 100 runs are defined and successful, so the
Clopper-Pearson interval at δ = 0.05 has upper end 1 and a lower end near 1:
(δ/2)^(1/n) = 0.9638 two-sided or δ^(1/n) = 0.9705 one-sided.
"""


def externals():
    return {}


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded, result.successes), (100, 0, 100))
    t.assertEqual(list(result.observations), [True] * 100)
    t.assertAlmostEqual(result.upper_ci, 1.0, places=9)
    t.assertTrue(0.95 < result.lower_ci < 0.98, f"lower_ci = {result.lower_ci}")
