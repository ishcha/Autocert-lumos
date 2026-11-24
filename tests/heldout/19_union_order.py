"""19_union_order: `+` on ordered sets and on graphs keeps the left elements first,
then the right operand's new elements, without repeats.

rules = base + extra = [honest, safe, brief, kind, cite]: "Be safe." is not repeated.
{lead} + rules puts the sampled rule first, then the other four in rule order.
Graph({z, x}) + Graph({y, x}) has nodes [z, x, y], i.e. Carol, Alice, Bob.
Mock world: the judge passes iff the safety rule comes first: P = 1/5.
"""

from collections import Counter

from lumos.testing import Recorder, assert_distribution, assert_rate

RULES = ["Be honest.", "Be safe.", "Be brief.", "Be kind.", "Cite sources."]
USERS = "Carol, Alice, Bob"


def prompt(lead):
    ordered = [lead] + [r for r in RULES if r != lead]
    return "Rules: " + " ".join(ordered) + "\nUsers: " + USERS


def externals():
    return {
        "L": Recorder(lambda p: p),
        "J_ok": lambda r: 1 if r.startswith("Rules: Be safe.") else 0,
    }


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded), (500, 0))
    prompts = ext["L"].calls
    t.assertEqual(len(prompts), result.trials)
    expected = {prompt(lead): lead for lead in RULES}
    for p in prompts:
        t.assertIn(p, expected)
    leads = [expected[p] for p in prompts]
    for lead, observation in zip(leads, result.observations):
        t.assertEqual(observation, lead == "Be safe.")
    assert_distribution(t, Counter(leads), {r: 1 / 5 for r in RULES}, result.trials, "lead rule")
    assert_rate(t, result.successes, result.trials, 1 / 5, "P(safety rule first)")
