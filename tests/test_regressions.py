"""Regression tests for correctness bugs found in review (each failed before its fix)."""

import random
import tempfile
import textwrap
import unittest
from pathlib import Path

from lumos_test_utils import env_after, run, value_of
from lumos.certify import Certifier
from lumos.cli import execute, load_world
from lumos.errors import RuntimeErrorLumos, UndefinedValue
from lumos.interpreter import NULL, Interpreter
from lumos_test_utils import parse

GRAPH = '''
a := Node(L{"A", "alpha"})
b := Node(L{"B"})
c := Node(L{"C"})
d := Node(L{"D"})
ab := Edge(a, b, L{"ab-rel"})
bc := Edge(b, c, L{"bc-rel"})
G := Graph({a, b, c, d}, {ab, bc})
'''


def names(nodes):
    return [n.name for n in nodes]


def certify(source, cache=True, seed=0, **externals):
    interp = Interpreter(seed=seed)
    for name, value in externals.items():
        interp.env.set(name, value)
    return Certifier(cache_invariants=cache).run_program(parse(source), interp)


class InvariantCacheSoundnessTests(unittest.TestCase):
    def test_per_run_bindings_are_not_treated_as_invariant(self):
        class PerRun(list):
            def lumos_clone_for_trial(self, seed):
                return PerRun([random.Random(seed).choice(["p0", "p1", "p2", "p3"])])

        source = 'estimateProb 0.05 200 "CP" :\n    S := {m | m in M}\ntoEstimate(S[0] = M[0])'
        for cache in (True, False):
            with self.subTest(cache=cache):
                (result,) = certify(source, cache=cache, M=PerRun(["p0"]))
                self.assertEqual(result.successes, 200)

    def test_trial_index_is_not_invariant(self):
        source = ('estimateProb 0.05 6 "CP" :\n    S := {i | i in {0, ..., 5} and i < __lumos_trial_index__}\n'
                  'toEstimate(len(S) = __lumos_trial_index__)')
        (result,) = certify(source)
        self.assertEqual(result.successes, 6)


class ObservationCheckTests(unittest.TestCase):
    def test_bad_observation_stops_at_the_first_run(self):
        calls = []
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            certify('estimateProb 0.05 1000 "CP" :\n    r := L("p")\ntoEstimate(J(r))',
                    L=lambda p: calls.append(p) or "reply", J=lambda r: 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("estimateProb run 0, seed", str(ctx.exception))
        self.assertEqual(ctx.exception.line, 3)
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            certify('estimateProb 0.05 200 "Bernstein" : x := 2; toEstimate(x)')
        self.assertIn("run 0", str(ctx.exception))


class ObservationListTests(unittest.TestCase):
    def test_pair_literal_keeps_equal_bounds(self):
        for bounds in ([1.0, 1.0], [0, 0.0], [0.2, 0.4]):
            with self.subTest(bounds=bounds):
                (result,) = certify('estimateProb 0.05 20 "Bernstein" :\n    b := B("p")\ntoEstimate({b[0], b[1]})',
                                    B=lambda p, v=bounds: v)
                self.assertEqual(result.observations[0], (float(bounds[0]), float(bounds[1])))


class NodeIdentityTests(unittest.TestCase):
    def test_redeclared_node_is_a_new_node(self):
        env = env_after('v := Node(L{"a"})\nw := v\nv := Node(L{"a"})\nsame := w = v\nalso := w = w')
        self.assertEqual((env["same"], env["also"]), (False, True))

    def test_edges_keep_the_node_they_were_declared_with(self):
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            env_after('v_1 := Node(L{"one"})\nv_2 := Node(L{"two"})\nv_e := Node(L{""})\ne1 := Edge(v_1, v_e)\n'
                      'v_e := Node(L{""})\ne2 := Edge(v_2, v_e)\nG := Graph({v_1, v_2, v_e}, {e1, e2})')
        self.assertIn("not a node of the graph", str(ctx.exception))

    def test_nodes_with_the_same_name_can_share_a_graph(self):
        env = env_after('v := Node(L{"a"})\nw := v\nv := Node(L{"b"})\nG := Graph({w, v}, {})\nn := len(G.N)')
        self.assertEqual(env["n"], 2)


class GraphValueTests(unittest.TestCase):
    def test_graph_equality_is_structural(self):
        self.assertTrue(value_of("(a + b) = (a + b)", GRAPH))
        self.assertTrue(value_of("(a + b) = (b + a)", GRAPH))
        self.assertFalse(value_of("(a + b) = (a + c)", GRAPH))
        self.assertFalse(value_of("(a + b) = (a + b + ab)", GRAPH))
        self.assertEqual(value_of("len({a + b, a + b})", GRAPH), 1)

    def test_union_puts_the_left_operand_first(self):
        self.assertEqual(names(value_of("(c + G).N", GRAPH)), ["c", "a", "b", "d"])
        self.assertEqual(names(value_of("NB(b, c + G)", GRAPH)), ["c", "a"])
        self.assertEqual(names(value_of("(G + c).N", GRAPH)), ["a", "b", "c", "d"])

    def test_nb_with_one_graph_under_two_names(self):
        self.assertEqual(names(value_of("NB(a)", GRAPH + "H := G")), ["b"])
        self.assertEqual(names(value_of("NB(a)", GRAPH + "G2 := G + a")), ["b"])


class NullMemberTests(unittest.TestCase):
    def test_only_attribute_access_on_null_gives_null(self):
        self.assertIs(value_of("getEdge(a, d, G).L", GRAPH), NULL)
        self.assertIs(value_of("getEdge(a, d, G).L[0]", GRAPH), NULL)
        for member in ("N", "E", "foo"):
            with self.subTest(member=member):
                with self.assertRaises(UndefinedValue):
                    value_of(f"getEdge(a, d, G).{member}", GRAPH)


class ExternalErrorTests(unittest.TestCase):
    def test_bad_measures_are_lumos_errors(self):
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            certify('estimateProb 0.05 5 "CP" : x := sample(meas({1, 2})); toEstimate(true)',
                    meas=lambda xs: 0.5)
        self.assertIn("list of weights", str(ctx.exception))

    def test_huge_weights_are_rescaled(self):
        (result,) = certify('estimateProb 0.05 400 "CP" : x := sample(meas({1, 2})); toEstimate(x = 1)',
                            meas=lambda xs: [1e308, 1e308])
        self.assertLess(abs(result.successes / 400 - 0.5), 0.1)

    def test_only_meas_can_come_from_the_environment(self):
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            value_of("sample(len({1, 2}))")
        self.assertIn("Unknown measure 'len'", str(ctx.exception))

    def test_failing_clone_hook_is_reported(self):
        class Broken:
            def lumos_clone_for_trial(self, seed):
                raise ValueError("no copies")

            def __call__(self, x):
                return 1

        with self.assertRaises(RuntimeErrorLumos) as ctx:
            certify('estimateProb 0.05 5 "CP" : r := J(1); toEstimate(r = 1)', J=Broken())
        self.assertIn("'J'", str(ctx.exception))


class ListSetConsistencyTests(unittest.TestCase):
    def test_membership_matches_equality(self):
        self.assertTrue(value_of('a.L = {"A", "alpha"}', GRAPH))
        self.assertTrue(value_of('a.L in {{"A", "alpha"}}', GRAPH))
        self.assertEqual(value_of('len({a.L, {"A", "alpha"}})', GRAPH), 1)


class CsvLoaderTests(unittest.TestCase):
    def load(self, text, encoding="utf-8"):
        directory = tempfile.mkdtemp()
        (Path(directory) / "kg.csv").write_text(text, encoding=encoding)
        interp = Interpreter(base_dir=directory)
        interp.eval_program(parse('G := getGraph("kg.csv")'))
        return interp.env.get("G")

    def test_byte_order_mark(self):
        self.assertEqual(len(self.load("src_id,dst_id\na,b\n", encoding="utf-8-sig").nodes), 2)

    def test_empty_node_id_is_an_error(self):
        with self.assertRaises(RuntimeErrorLumos):
            self.load("src_id,dst_id\n,b\n")

    def test_long_fields(self):
        g = self.load("src_id,src_ctx,dst_id\na," + "x" * 200_000 + ",b\n")
        self.assertEqual(len(g.node_named("a").attributes[0]), 200_000)

    def test_errors_report_physical_lines(self):
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            self.load('src_id,src_ctx,dst_id\na,"multi\nline",b\na,other,c\n')
        self.assertIn("line 4", str(ctx.exception))


class WorldLoadingTests(unittest.TestCase):
    def write(self, name, text):
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(textwrap.dedent(text))
        return path

    def test_world_with_dataclasses_and_future_annotations(self):
        world = self.write("w.py", """
            from __future__ import annotations
            from dataclasses import dataclass

            @dataclass
            class Judge:
                threshold: int = 1
                def __call__(self, x) -> int:
                    return 1

            def externals():
                return {"J": Judge()}
        """)
        self.assertIn("J", load_world(world).externals())

    def test_world_paths(self):
        world = self.write("world_without_suffix", "def externals():\n    return {}\n")
        self.assertEqual(load_world(world).externals(), {})
        with self.assertRaises(RuntimeErrorLumos):
            load_world(Path(tempfile.mkdtemp()) / "missing.py")


class MinorTests(unittest.TestCase):
    def test_long_operator_chains(self):
        self.assertEqual(value_of(" + ".join(['"x"'] * 2000)), "x" * 2000)
        self.assertEqual(value_of(" and ".join(["true"] * 2000)), True)

    def test_tracing_stops_after_estimate_prob(self):
        directory = tempfile.mkdtemp()
        spec = Path(directory) / "s.lumos"
        spec.write_text('estimateProb 0.05 2 "CP" : r := L("in"); toEstimate(true)\nafter := L("after")\n')
        lines = []
        execute(spec, trace=5, externals={"L": lambda p: p}, out=lines.append)
        self.assertFalse(any('"after"' in line for line in lines))

    def test_execute_keeps_per_run_data_bindings(self):
        class PerRun(list):
            def lumos_clone_for_trial(self, seed):
                return PerRun([random.Random(seed).random()])

        directory = tempfile.mkdtemp()
        spec = Path(directory) / "s.lumos"
        spec.write_text('estimateProb 0.05 40 "CP" : r := J(M[0]); toEstimate(true)\n')
        seen = []
        execute(spec, externals={"M": PerRun([0.5]), "J": lambda x: seen.append(x) or 1})
        self.assertEqual(len(set(seen)), 40)


if __name__ == "__main__":
    unittest.main()
