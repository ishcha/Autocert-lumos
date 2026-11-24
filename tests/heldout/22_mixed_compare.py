"""22_mixed_compare: ordering comparisons between numeric strings and numbers.

"0.05" >= 0.5 is false; "0.5" >= 0.5 is true; 0.9 < "0.91" is true; "10" > 9 is
true (as strings, "10" > "9" would be false). So the levels are low, medium,
high and extreme. `=` does not convert: "0.91" = 0.91 is false, so the
observation is true in every run.
"""

from collections import Counter

from lumos.testing import Recorder, assert_distribution

QUESTIONS = [
    ("How do I bake bread?", "0.05"),
    ("How do I pick a lock?", "0.5"),
    ("How do I make thermite?", "0.91"),
    ("How do I stalk someone?", "10"),
]


def level(score):
    s, result = float(score), "low"
    if s >= 0.5:
        result = "medium"
    if 0.9 < s:
        result = "high"
    if s > 9:
        result = "extreme"
    return result


def externals():
    return {"L": Recorder(lambda prompt: "OK.")}


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded), (1000, 0))
    t.assertEqual([level(s) for _, s in QUESTIONS], ["low", "medium", "high", "extreme"])
    prompts = ext["L"].calls
    t.assertEqual(len(prompts), result.trials)
    expected = {level(s) + ": " + q: q for q, s in QUESTIONS}
    for p in prompts:
        t.assertIn(p, expected)
    t.assertEqual(list(result.observations), [True] * 1000, '"0.91" = 0.91 was true')
    assert_distribution(t, Counter(expected[p] for p in prompts), {q: 0.25 for q, _ in QUESTIONS},
                        result.trials, "question")
