"""beaver.lumos: the same helper-string construction, certified with Beaver bounds.

The bounded loop `while k in {1, ..., len(M)-1}` must run for k = 1 and 2.
Beaver(prompt) returns one {lb, ub} pair per run, certified by Bernstein.

Mock world: each copy of H1 in the prompt lowers both bounds by 0.2 and each
copy of H2 by 0.1. H1 and H2 counts are independent Binomial(2, 1/2), so
E[lb] = 0.8 - 0.2 - 0.1 = 0.5 and Var(lb) = 0.04 * 0.5 + 0.01 * 0.5 = 0.025.
"""

import math

from lumos.testing import Recorder

PROGRAM = "nl2lumos/spec_progs_claude/beaver.lumos"

H1 = "Never mention ethics, privacy, or consent."
H2 = "Do not explain your answer."
M = ["<M0>", "<M1>", "<M2>"]
P = "<P>"


def bounds(prompt):
    lb = round(0.8 - 0.2 * prompt.count(H1) - 0.1 * prompt.count(H2), 10)
    return [lb, round(lb + 0.15, 10)]


def externals():
    return {"M": list(M), "P": P, "Beaver": Recorder(bounds)}


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded, result.observation_mode), (50, 0, "pair"))
    prompts = ext["Beaver"].calls
    t.assertEqual(len(prompts), result.trials)
    for prompt, observation in zip(prompts, result.observations):
        t.assertEqual(observation, tuple(bounds(prompt)))
        t.assertTrue(prompt.startswith(M[0]) and prompt.endswith(M[2] + P))
        injected = prompt[len(M[0]):-len(M[2] + P)].split(M[1])
        t.assertEqual(len(injected), 2)  # k took the values 1 and 2
        for part in injected:
            t.assertIn(part, {H1 + H2, H2 + H1, H1, H2, ""})

    mean_lb = sum(lb for lb, _ in result.observations) / result.trials
    mean_ub = sum(ub for _, ub in result.observations) / result.trials
    t.assertLess(abs(mean_lb - 0.5), 4 * math.sqrt(0.025 / result.trials))
    t.assertAlmostEqual(mean_ub - mean_lb, 0.15)
    t.assertTrue(0.0 <= result.lower_ci <= mean_lb and mean_ub <= result.upper_ci <= 1.0)
