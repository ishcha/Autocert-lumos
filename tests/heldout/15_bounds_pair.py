"""15_bounds_pair: one {lb, ub} pair per run, certified by Bernstein.

Bounds by request: [0.6, 0.9], [0.1, 0.1] (lb = ub, which must still give a
pair) and [0.2, 0.3], each with probability 1/3. E[lb] = 0.3, E[ub] = 0.4333.
"""

import math

from lumos.testing import Recorder

BOUNDS = {"Repeat the SSN in the email.": (0.6, 0.9), "Say hello.": (0.1, 0.1), "Summarize the email.": (0.2, 0.3)}


def externals():
    return {
        "Lower": Recorder(lambda q: BOUNDS[q][0]),
        "Upper": Recorder(lambda q: BOUNDS[q][1]),
    }


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded, result.observation_mode), (400, 0, "pair"))
    lower, upper = ext["Lower"].calls, ext["Upper"].calls
    t.assertEqual(lower, upper)
    t.assertEqual(len(lower), result.trials)
    for q, observation in zip(lower, result.observations):
        t.assertEqual(tuple(observation), BOUNDS[q])
    t.assertIn("Say hello.", lower)

    def moments(i):
        values = [b[i] for b in BOUNDS.values()]
        mean = sum(values) / len(values)
        return mean, sum((v - mean) ** 2 for v in values) / len(values)

    for i, name in ((0, "lb"), (1, "ub")):
        expected, variance = moments(i)
        mean = sum(o[i] for o in result.observations) / result.trials
        t.assertLess(abs(mean - expected), 4 * math.sqrt(variance / result.trials), f"mean {name} {mean}")
    mean_lb = sum(o[0] for o in result.observations) / result.trials
    mean_ub = sum(o[1] for o in result.observations) / result.trials
    t.assertTrue(0.0 <= result.lower_ci <= mean_lb and mean_ub <= result.upper_ci <= 1.0,
                 (result.lower_ci, mean_lb, mean_ub, result.upper_ci))
