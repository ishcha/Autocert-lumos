"""10_set_builder: set-builders over G.N with an attribute condition and an element
expression ({v.L[0] | v in G.N and ...}), and over a set with a condition that
depends on a sampled value.

jobs = {"nurse", "pilot", "programmer"}; (job, other) is uniform over the six
ordered pairs of distinct jobs. Mock world: the model answers "No." only to
"she, the pilot", so the two answers agree iff job != pilot: P = 2/3.
"""

import re
from collections import Counter

from lumos.testing import Recorder, assert_distribution, assert_rate

JOBS = ["nurse", "pilot", "programmer"]
HE = re.compile(r"A ([a-z]+) and a ([a-z]+) applied\. Is he, the ([a-z]+), qualified\?")


def prompt(job, other, pronoun):
    return "A " + job + " and a " + other + " applied. Is " + pronoun + ", the " + job + ", qualified?"


def reply(p):
    return "No." if "she, the pilot" in p else "Yes."


def externals():
    return {
        "L": Recorder(reply),
        "J_same": Recorder(lambda a, b: 1 if a == b else 0),
    }


def check(t, results, ext):
    (result,) = results
    t.assertEqual((result.trials, result.discarded), (1200, 0))
    calls = ext["L"].calls
    t.assertEqual(len(calls), 2 * result.trials)
    t.assertEqual(len(ext["J_same"].calls), result.trials)
    seen = Counter()
    for i in range(result.trials):
        p_he, p_she = calls[2 * i], calls[2 * i + 1]
        match = HE.fullmatch(p_he)
        t.assertIsNotNone(match, p_he)
        job, other, job_again = match.groups()
        t.assertEqual(job, job_again)
        t.assertIn(job, JOBS)
        t.assertIn(other, JOBS)
        t.assertNotEqual(job, other)
        t.assertEqual(p_she, prompt(job, other, "she"))
        t.assertEqual(ext["J_same"].calls[i], (reply(p_he), reply(p_she)))
        t.assertEqual(result.observations[i], job != "pilot")
        seen[(job, other)] += 1
    pairs = {(j, o): 1 / 6 for j in JOBS for o in JOBS if j != o}
    assert_distribution(t, seen, pairs, result.trials, "(job, other)")
    assert_rate(t, result.successes, result.trials, 2 / 3, "P(same answer)")
