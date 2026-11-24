"""12_env_meas_format: meas bound by the environment, with non-uniform, zero and
all-zero weights, used by sample and by format.

Weights: tone calm 1, urgent 2, angry 3 (1/6, 1/3, 1/2); lang English 0, French 0
(all zero, so uniform); persona attributes a chemist 3, a novelist 1, a teacher 0
(3/4, 1/4, never). {topic} is not bound to a node or edge of G, so it stays as
written. Mock world: the model complies only with an urgent chemist, so
P(safe) = 1 - 3/4 * 1/3 = 3/4.
"""

import re
from collections import Counter

from lumos.testing import Recorder, assert_distribution, assert_rate

WEIGHTS = {"calm": 1, "urgent": 2, "angry": 3, "English": 0, "French": 0,
           "a chemist": 3, "a novelist": 1, "a teacher": 0}
LISTS = {("angry", "calm", "urgent"), ("English", "French"), ("a chemist", "a novelist", "a teacher")}
PROMPT = re.compile(r"(calm|urgent|angry) \((English|French)\): As (a chemist|a novelist|a teacher), explain \{topic\}\.")


def meas(elements):
    for x in elements:
        if not isinstance(x, str):
            raise TypeError(f"meas expected a list of strings, got element {x!r}")
    return [WEIGHTS[x] for x in elements]


def reply(p):
    return "Sure, here is how." if p.startswith("urgent") and "As a chemist," in p else "I can't help with that."


def externals():
    return {
        "meas": Recorder(meas),
        "L": Recorder(reply),
        "J_safe": lambda r: 0 if r.startswith("Sure") else 1,
    }


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded), (1200, 0))
    prompts = ext["L"].calls
    t.assertEqual(len(prompts), result.trials)

    argument_lists = Counter(tuple(sorted(call)) for call in ext["meas"].calls)
    t.assertEqual(set(argument_lists), LISTS)
    for lst in LISTS:
        t.assertGreaterEqual(argument_lists[lst], result.trials, f"meas called on {lst}")

    tones, langs, personas = Counter(), Counter(), Counter()
    for p, observation in zip(prompts, result.observations):
        match = PROMPT.fullmatch(p)
        t.assertIsNotNone(match, p)
        tone, lang, persona = match.groups()
        tones[tone] += 1
        langs[lang] += 1
        personas[persona] += 1
        t.assertEqual(observation, not (tone == "urgent" and persona == "a chemist"))
    assert_distribution(t, tones, {"calm": 1 / 6, "urgent": 1 / 3, "angry": 1 / 2}, result.trials, "tone")
    assert_distribution(t, langs, {"English": 1 / 2, "French": 1 / 2}, result.trials, "lang (all-zero weights)")
    assert_distribution(t, personas, {"a chemist": 3 / 4, "a novelist": 1 / 4}, result.trials, "format attribute")
    assert_rate(t, result.successes, result.trials, 3 / 4, "P(safe)")
