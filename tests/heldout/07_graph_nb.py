"""07_graph_nb: Node, Edge, Graph and NB(v) on an undirected graph.

Edges a-hub, hub-b, c-b (declared in mixed directions). NB(hub) = {a, b},
NB(a) = {hub}, NB(b) = {hub, c}. So P(n2 = hub) = 1/2 + 1/2 * 1/2 = 3/4.
All four nodes have no attributes: they are distinct nodes that differ only
by identity (README: a node is equal only to itself).
"""

from fractions import Fraction

from lumos.testing import assert_rate

NODES = ["hub", "a", "b", "c"]
EDGES = [("a", "hub"), ("hub", "b"), ("c", "b")]


def neighbours(v):
    return [u for u in NODES if (u, v) in EDGES or (v, u) in EDGES]


def reference():
    p = Fraction(0)
    for n1 in neighbours("hub"):
        for n2 in neighbours(n1):
            if n2 == "hub":
                p += Fraction(1, len(neighbours("hub"))) / len(neighbours(n1))
    return p


def externals():
    return {}


def check(t, results, ext):
    (result,) = results
    p = reference()
    t.assertEqual(p, Fraction(3, 4))
    t.assertEqual((result.trials, result.discarded), (1000, 0))
    assert_rate(t, result.successes, result.trials, p, "P(n2 = hub)")
