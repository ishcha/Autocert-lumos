"""20_node_identity: Node(...) values with identical attributes are distinct.

{d1, d2, harmful} has three elements, so q is each with probability 1/3, and
NB(harmful) = {d1, d2}, so n is each with probability 1/2. The tag records
which node was drawn, by identity: six tags, each with probability 1/6.
d1 = d2 is false in every run.
"""

from collections import Counter

from lumos.testing import Recorder, assert_distribution

TAGS = [q + "/" + n for q in ("d1", "d2", "harmful") for n in ("d1", "d2")]


def externals():
    return {"L": Recorder(lambda tag: "OK.")}


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded), (1200, 0))
    t.assertEqual(list(result.observations), [True] * 1200, "d1 = d2 was true")
    tags = ext["L"].calls
    t.assertEqual(len(tags), result.trials)
    assert_distribution(t, Counter(tags), {tag: 1 / 6 for tag in TAGS}, result.trials, "(q, n) by identity")
