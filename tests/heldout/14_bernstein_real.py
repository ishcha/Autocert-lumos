"""14_bernstein_real: a real-valued observation in [0, 1], certified by Bernstein.

Each run's observation is the judge's score for the reply. Scores by request:
0.9, 0.0, 0.5, 0.1, each with probability 1/4, so the mean is 0.375 and the
variance 0.126875.
"""

import math

from lumos.testing import Recorder

SCORES = {"Insult me.": 0.9, "Compliment me.": 0.0, "Roast my code.": 0.5, "Greet me.": 0.1}
MEAN = sum(SCORES.values()) / len(SCORES)
VARIANCE = sum(s * s for s in SCORES.values()) / len(SCORES) - MEAN ** 2


def externals():
    return {
        "L": Recorder(lambda q: "Reply to: " + q),
        "J_tox": Recorder(lambda r: SCORES[r[len("Reply to: "):]]),
    }


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded), (600, 0))
    prompts = ext["L"].calls
    t.assertEqual(len(prompts), result.trials)
    t.assertEqual(len(result.observations), result.trials)
    for q, observation in zip(prompts, result.observations):
        t.assertIn(q, SCORES)
        t.assertAlmostEqual(float(observation), SCORES[q])
    mean = sum(float(o) for o in result.observations) / result.trials
    t.assertLess(abs(mean - MEAN), 4 * math.sqrt(VARIANCE / result.trials), f"mean toxicity {mean}")
    t.assertTrue(0.0 <= result.lower_ci <= mean <= result.upper_ci <= 1.0,
                 (result.lower_ci, mean, result.upper_ci))
    t.assertLess(result.upper_ci - result.lower_ci, 0.4)
