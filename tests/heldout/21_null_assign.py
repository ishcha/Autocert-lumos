"""21_null_assign: null can be assigned; e = e is false for null; passing null is ⊥.

- flu: the edge exists, e = e is true (edges compare by content), rel = "treats", gt = "yes".
- ulcer: the edge exists (declared in the other order), rel = "worsens", gt = "no".
- gout: no edge. e and rel are null, e = e is false ("unknown"), and gt = "no".
  L is called, then Log(rel) passes null to a function, which is ⊥, so the run
  is discarded and Log never receives it. P(defined) = 2/3.
Mock world: the model always answers "yes", so P(correct | defined) = 1/2.
"""

from collections import Counter

from lumos.testing import Recorder, assert_discards, assert_distribution, assert_rate

PROMPTS = {
    "Does aspirin treat flu? (known)": "flu",
    "Does aspirin treat ulcer? (known)": "ulcer",
    "Does aspirin treat gout? (unknown)": "gout",
}
RELATION = {"flu": "treats", "ulcer": "worsens"}
GT = {"flu": "yes", "ulcer": "no"}


def externals():
    return {
        "L": Recorder(lambda prompt: "yes"),
        "Log": Recorder(lambda x: 1),
        "J_QA": Recorder(lambda response, gt: 1 if response == gt else 0),
    }


def check(t, results, ext):
    (result,) = results
    t.assertEqual(result.trials, 900)
    prompts = ext["L"].calls
    for p in prompts:
        t.assertIn(p, PROMPTS)
    conditions = [PROMPTS[p] for p in prompts]
    t.assertEqual(len(prompts), result.trials + result.discarded)
    t.assertEqual(conditions.count("gout"), result.discarded, "runs that passed null to Log were not discarded")
    accepted = [c for c in conditions if c != "gout"]
    t.assertEqual(len(accepted), result.trials)
    t.assertEqual(ext["Log"].calls, [RELATION[c] for c in accepted], "Log received a null")
    t.assertEqual(ext["J_QA"].calls, [("yes", GT[c]) for c in accepted])
    for c, observation in zip(accepted, result.observations):
        t.assertEqual(observation, GT[c] == "yes")
    assert_distribution(t, Counter(accepted), {"flu": 0.5, "ulcer": 0.5}, result.trials, "condition | defined")
    assert_rate(t, result.successes, result.trials, 0.5, "P(correct | defined)")
    assert_discards(t, result, 2 / 3)
