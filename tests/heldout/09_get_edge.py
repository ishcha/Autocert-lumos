"""09_get_edge: getEdge in either order, edge attributes, and a missing edge.

aspirin-headache was declared as Edge(headache, aspirin, ...), so getEdge must
find it in the other order. insulin-headache has no edge: getEdge gives null,
null.L[0] is null, and null = "treats" is false, so gt = "no" and the run is
defined (no discards). Mock world: the model answers "yes" unless asked about
insulin, so it is right on aspirin-headache and insulin-headache: P = 1/2.
"""

from collections import Counter

from lumos.testing import Recorder, assert_distribution, assert_rate

RELATIONS = {("aspirin", "headache"): "treats", ("insulin", "diabetes"): "treats", ("aspirin", "diabetes"): "worsens"}
PAIRS = [(d, c) for d in ("aspirin", "insulin") for c in ("headache", "diabetes")]


def prompt(d, c):
    return "Does " + d + " treat " + c + "? Answer yes or no."


def ground_truth(d, c):
    return "yes" if RELATIONS.get((d, c)) == "treats" else "no"


def reply(p):
    return "no" if "insulin" in p else "yes"


def externals():
    return {
        "L": Recorder(reply),
        "J_QA": Recorder(lambda response, gt: 1 if response == gt else 0),
    }


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded), (1000, 0))  # a null edge is not ⊥
    prompts, judged = ext["L"].calls, ext["J_QA"].calls
    t.assertEqual((len(prompts), len(judged)), (result.trials, result.trials))
    expected = {prompt(d, c): (d, c) for d, c in PAIRS}
    seen = Counter()
    for p, (response, gt), observation in zip(prompts, judged, result.observations):
        t.assertIn(p, expected)
        pair = expected[p]
        seen[pair] += 1
        t.assertEqual((response, gt), (reply(p), ground_truth(*pair)), f"judge call for {pair}")
        t.assertEqual(observation, reply(p) == ground_truth(*pair))
    assert_distribution(t, seen, {pair: 0.25 for pair in PAIRS}, result.trials, "(drug, condition)")
    p_correct = sum(1 for pair in PAIRS if reply(prompt(*pair)) == ground_truth(*pair)) / len(PAIRS)
    t.assertEqual(p_correct, 0.5)
    assert_rate(t, result.successes, result.trials, p_correct, "P(correct)")
