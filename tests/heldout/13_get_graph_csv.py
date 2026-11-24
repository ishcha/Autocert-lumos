"""13_get_graph_csv: getGraph from a CSV edge list (fixtures/staff.csv).

Node attributes come from the src_* / dst_* columns in column order (name, kind),
the edge attribute from `relation`, and the row with an empty dst_id declares
the isolated node Dan. One edge is listed team-first (Payroll, Carol), and
getEdge is called team-first for all of them, so both orders are exercised.

The person is uniform over the four people; Dan has no neighbours, so
P(defined) = 3/4 and ⊥ strikes before L is called. Mock world: the model names
the team from the context, except when the role is "lead". Given defined:
Alice (1/3, either team) and Carol (1/3) are answered correctly, Bob (1/3) is
not, so P(correct | defined) = 2/3.
"""

import csv
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from lumos.testing import Distribution, Recorder, assert_discards, assert_distribution, assert_rate

BASE_DIR = "fixtures"
FIXTURE = Path(__file__).parent / "fixtures" / "staff.csv"
CONTEXT = re.compile(r"Context: (\w+) is a (\w+) of (\w+)\.\nQuestion: Which team is (\w+) in\?")


def load_fixture():
    attributes, relations = {}, {}
    with open(FIXTURE, newline="") as handle:
        for row in csv.DictReader(handle):
            attributes[row["src_id"]] = [row["src_name"], row["src_kind"]]
            if row["dst_id"]:
                attributes[row["dst_id"]] = [row["dst_name"], row["dst_kind"]]
                relations[frozenset((row["src_id"], row["dst_id"]))] = row["relation"]
    return attributes, relations


def prompt(name, role, team):
    return "Context: " + name + " is a " + role + " of " + team + ".\nQuestion: Which team is " + name + " in?"


def reply(p):
    match = CONTEXT.fullmatch(p)
    if match is None:
        return "?"
    _, role, team, _ = match.groups()
    return "I am not sure." if role == "lead" else team


def reference():
    attributes, relations = load_fixture()
    people = [n for n, a in attributes.items() if a[1] == "person"]
    dist = Distribution()
    for person in people:
        p = Fraction(1, len(people))
        teams = [n for n in attributes if frozenset((person, n)) in relations]
        if not teams:
            dist.undefined += p
            continue
        for team in teams:
            role = relations[frozenset((person, team))]
            dist.add(prompt(attributes[person][0], role, attributes[team][0]), p / len(teams))
    return dist


def externals():
    return {
        "L": Recorder(reply),
        "J_QA": Recorder(lambda response, gt: 1 if response == gt else 0),
    }


def check(t, results, ext):
    (result,) = results
    dist = reference()
    t.assertEqual(dist.defined, Fraction(3, 4))
    t.assertEqual(len(dist.mass), 4)
    t.assertEqual(result.trials, 1000)
    prompts, judged = ext["L"].calls, ext["J_QA"].calls
    t.assertEqual((len(prompts), len(judged)), (result.trials, result.trials))  # ⊥ strikes before L
    for p, (response, gt), observation in zip(prompts, judged, result.observations):
        t.assertIn(p, dist.mass)
        team = CONTEXT.fullmatch(p).group(3)
        t.assertEqual((response, gt), (reply(p), team))
        t.assertEqual(observation, reply(p) == team)
    assert_distribution(t, Counter(prompts), dist.conditional_marginal(lambda p: p), result.trials, "(person, team)")
    p_correct = dist.conditional(lambda p: reply(p) == CONTEXT.fullmatch(p).group(3))
    t.assertEqual(p_correct, Fraction(2, 3))
    assert_rate(t, result.successes, result.trials, p_correct, "P(correct | defined)")
    assert_discards(t, result, dist.defined)
