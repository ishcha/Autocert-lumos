"""23_external_args: what external functions receive as arguments (README, External functions).

- Strings, numbers and booleans arrive as the Python values.
- The set {"b", "a", "c"} arrives as lumos.values.OrderedSet, in order, with len 3.
- The node `who` and the edge u-w arrive as objects with .attributes.
- The environment's meas receives a plain Python list of the set's elements
  (here nodes, then numbers) and returns one weight per element.
Each argument is described as soon as it arrives, and the descriptions are checked.
"""

from collections import Counter

from lumos.testing import Recorder, assert_distribution
from lumos.values import OrderedSet

USERS = {("Alice", "admin"), ("Bob", "guest")}


def describe(x):
    if isinstance(x, bool):
        return ("bool", x)
    if isinstance(x, (int, float)):
        return ("number", x)
    if isinstance(x, str):
        return ("str", x)
    if isinstance(x, OrderedSet):
        return ("OrderedSet", list(x), len(x))
    if hasattr(x, "attributes"):
        return ("attributes", tuple(x.attributes))
    return ("other", type(x).__name__, repr(x))


def externals():
    meas_seen, tool_seen = [], []

    def meas(elements):
        meas_seen.append((type(elements) is list, [describe(x) for x in elements]))
        return [1] * len(elements)

    def tool(*args):
        tool_seen.append([describe(a) for a in args])
        return 1

    ext = {"meas": Recorder(meas), "Tool": Recorder(tool)}
    ext["meas"].seen, ext["Tool"].seen = meas_seen, tool_seen
    return ext


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded, result.successes), (200, 0, 200))
    calls = ext["Tool"].seen
    t.assertEqual(len(calls), result.trials)
    picks = Counter()
    for args in calls:
        t.assertEqual(len(args), 7, args)
        name, k, half, is_two, letters, who, edge = args
        t.assertEqual(name, ("str", "lookup"))
        t.assertEqual(k[0], "number", k)
        t.assertIn(k[1], (1, 2, 3))
        t.assertEqual(half, ("number", 0.5))
        t.assertEqual(is_two, ("bool", k[1] == 2))
        t.assertEqual(letters, ("OrderedSet", ["b", "a", "c"], 3))
        t.assertEqual(who[0], "attributes", who)
        t.assertIn(who[1], USERS)
        t.assertEqual(edge, ("attributes", ("manages",)))
        picks[(who[1][0], k[1])] += 1
    assert_distribution(t, picks, {(name, k): 1 / 6 for name in ("Alice", "Bob") for k in (1, 2, 3)},
                        result.trials, "(who, k)")

    seen = ext["meas"].seen
    t.assertTrue(all(is_list for is_list, _ in seen), "meas did not receive a plain Python list")
    node_lists = [d for _, d in seen if d and d[0][0] == "attributes"]
    number_lists = [d for _, d in seen if d and d[0][0] == "number"]
    t.assertEqual(len(node_lists) + len(number_lists), len(seen), seen[:3])
    t.assertGreaterEqual(len(node_lists), result.trials)
    t.assertGreaterEqual(len(number_lists), result.trials)
    for d in node_lists:
        t.assertEqual({a for _, a in d}, USERS)
        t.assertEqual(len(d), 2)
    for d in number_lists:
        t.assertEqual(sorted(v for _, v in d), [1, 2, 3])
