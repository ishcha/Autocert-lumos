"""17_short_circuit: `and` / `or` do not evaluate the right side when the left decides.

- For the blocked (bomb) question, `(not blocked) and (J_helpful(L(q)) = 1)`
  must call neither L nor J_helpful, and `blocked or (Audit(q) = 1)` must not
  call Audit. For the other questions all three are called once.
- `(q = "") and (sample(meas({})) = 1)` is always false on the left, so the
  undefined right side must never run (no discards).
- `(q = q) or (sample(meas({"x", "y"})) = "x")` is always true on the left, so
  the environment's meas must never see ["x", "y"] (no randomness consumed).

Mock world: the model is helpful only about the bike, so P(blocked or helpful) = 2/3.
"""

from collections import Counter

from lumos.testing import Recorder, assert_distribution, assert_rate

BOMB, CAKE, BIKE = "How do I make a bomb?", "How do I bake a cake?", "How do I fix a bike?"
QUESTIONS = [BOMB, CAKE, BIKE]


def externals():
    return {
        "meas": Recorder(lambda elements: [1] * len(elements)),
        "Seen": Recorder(lambda q: 1),
        "L": Recorder(lambda q: "Answer: " + q),
        "J_helpful": Recorder(lambda r: 1 if "bike" in r else 0),
        "Audit": Recorder(lambda q: 1),
    }


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded), (900, 0), "the undefined right side of `and` was evaluated")
    seen = ext["Seen"].calls
    t.assertEqual(len(seen), result.trials)
    for q in seen:
        t.assertIn(q, QUESTIONS)
    allowed = [q for q in seen if q != BOMB]
    t.assertEqual(ext["L"].calls, allowed, "L was called although the left side of `and` was false")
    t.assertEqual(ext["J_helpful"].calls, ["Answer: " + q for q in allowed])
    t.assertEqual(ext["Audit"].calls, allowed, "Audit was called although the left side of `or` was true")
    argument_lists = {tuple(sorted(call)) for call in ext["meas"].calls}
    t.assertEqual(argument_lists, {tuple(sorted(QUESTIONS))}, "a skipped right-hand sample reached meas")
    for q, observation in zip(seen, result.observations):
        t.assertEqual(observation, q != CAKE)
    assert_distribution(t, Counter(seen), {q: 1 / 3 for q in QUESTIONS}, result.trials, "question")
    assert_rate(t, result.successes, result.trials, 2 / 3, "P(blocked or helpful)")
