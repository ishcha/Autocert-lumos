"""Interpreter semantics, one feature at a time (expected values derived by hand)."""

import collections
import tempfile
import unittest
from pathlib import Path

from lumos_test_utils import env_after, run, value_of
from lumos.errors import RuntimeErrorLumos, UndefinedValue
from lumos.graph import AttributedEdge, AttributedNode, Graph
from lumos.interpreter import NULL
from lumos.ordered_set import OrderedSet

# A small undirected graph used by several tests:  a - b - c,  a - c,  d isolated.
GRAPH = '''
a := Node(L{"A", "alpha"})
b := Node(L{"B"})
c := Node(L{"C", "gamma", "ctx"})
d := Node(L{"D"})
ab := Edge(a, b, L{"ab-rel"})
bc := Edge(b, c, L{"bc-rel"})
ac := Edge(a, c, L{"ac-rel"})
G := Graph({a, b, c, d}, {ab, bc, ac})
'''


def names(nodes):
    return [n.name for n in nodes]


class AssignmentAndScopeTests(unittest.TestCase):
    def test_rebinding(self):
        self.assertEqual(env_after("x := 1\nx := x + 1")["x"], 2)

    def test_single_flat_store(self):
        # Variables assigned inside if/while bodies stay visible afterwards (paper: one state σ).
        env = env_after("i := 0\nwhile i < 3 do\n    i := i + 1\n    last := i\nif true then y := 5")
        self.assertEqual((env["i"], env["last"], env["y"]), (3, 3, 5))

    def test_bounded_loop_runs_once_per_element_inclusive(self):
        env = env_after("total := 0\nseen := {}\nwhile k in {1, ..., 4}\n    total := total + k\n    seen := seen + {k}")
        self.assertEqual(env["total"], 10)
        self.assertEqual(env["seen"].to_list(), [1, 2, 3, 4])
        self.assertEqual(env["k"], 4)

    def test_bounded_loop_over_empty_range_and_over_a_set(self):
        env = env_after("n := 0\nwhile k in {3, ..., 1} n := n + 1\nwhile s in {\"a\", \"b\"} n := n + 10")
        self.assertEqual(env["n"], 20)

    def test_undefined_variable(self):
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            env_after("x := 1\n\ny := z")
        self.assertIn("Undefined variable 'z'", str(ctx.exception))
        self.assertEqual(ctx.exception.line, 3)


class ArithmeticTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(value_of("10 - 3 - 2"), 5)
        self.assertIsInstance(value_of("10 - 3 - 2"), int)
        self.assertEqual(value_of("2 + 3 * 4"), 14)
        self.assertEqual(value_of("7 / 2"), 3.5)
        self.assertEqual(value_of("-2 * 3"), -6)
        self.assertEqual(value_of("0.5 * 4"), 2.0)

    def test_division_by_zero_is_undefined(self):
        with self.assertRaises(UndefinedValue):
            value_of("1 / 0")

    def test_booleans_are_not_numbers(self):
        for source in ["true + 1", "true * 5", "-true", "true < 2"]:
            with self.subTest(source=source):
                with self.assertRaises(RuntimeErrorLumos):
                    value_of(source)


class ComparisonTests(unittest.TestCase):
    def test_numeric_comparisons(self):
        self.assertTrue(value_of("1 < 2"))
        self.assertFalse(value_of("2 < 2"))
        self.assertTrue(value_of("2 <= 2"))
        self.assertTrue(value_of("3 > 2.5"))
        self.assertTrue(value_of("3 >= 3"))

    def test_numeric_strings_compare_as_numbers(self):
        # Numerically, not lexicographically ("10" < "9" as text).
        self.assertTrue(value_of('"10" > "9"'))
        self.assertTrue(value_of('"0.78" < "0.91"'))
        self.assertTrue(value_of('"0.5" < 1'))
        self.assertTrue(value_of('" 1e-3 " < "0.01"'))

    def test_non_numeric_strings_do_not_compare(self):
        with self.assertRaises(RuntimeErrorLumos):
            value_of('"abc" < "abd"')

    def test_equality(self):
        self.assertTrue(value_of('"a" = "a"'))
        self.assertTrue(value_of("1 = 1.0"))
        self.assertFalse(value_of("1 = true"))
        self.assertFalse(value_of("0 = false"))
        self.assertFalse(value_of('"0.91" = 0.91'))  # '=' does not convert strings
        self.assertTrue(value_of("{1, 2} = {1, 2}"))
        self.assertFalse(value_of("{1, 2} = {2, 1}"))  # ordered sets are ordered

    def test_nodes_are_distinct_declarations(self):
        env = env_after('e1 := Node(L{""})\ne2 := Node(L{""})\nsame := e1 = e1\ndiff := e1 = e2')
        self.assertTrue(env["same"])
        self.assertFalse(env["diff"])

    def test_edges_are_undirected(self):
        env = env_after('a := Node(L{"a"})\nb := Node(L{"b"})\nx := Edge(a, b)\ny := Edge(b, a)\n'
                        'z := Edge(a, b, L{"r"})\nxy := x = y\nxz := x = z')
        self.assertTrue(env["xy"])
        self.assertFalse(env["xz"])


class PlusTests(unittest.TestCase):
    def test_strings(self):
        self.assertEqual(value_of('"Human: " + "hi" + "\\n"'), "Human: hi\n")
        self.assertEqual(value_of('"n=" + 1'), "n=1")
        self.assertEqual(value_of('"x" + true'), "xtrue")

    def test_ordered_set_union(self):
        # First set's order, then the new elements of the second.
        self.assertEqual(value_of("{3, 1} + {1, 2, 3, 4}").to_list(), [3, 1, 2, 4])
        self.assertEqual(value_of("{} + {1}").to_list(), [1])

    def test_graph_union_lifts_nodes_and_edges(self):
        env = env_after(GRAPH + "g := a + c + ac\nh := b + bc")
        g, h = env["g"], env["h"]
        self.assertEqual(names(g.nodes), ["a", "c"])
        self.assertEqual(len(g.edges), 1)
        # Lifting the bare edge bc brings its endpoints with their real attributes.
        self.assertEqual(h.node_named("c").attributes, ["C", "gamma", "ctx"])

    def test_graph_union_with_contained_elements_is_the_graph(self):
        env = env_after(GRAPH + "g := G + a + bc")
        self.assertIs(env["g"], env["G"])

    def test_mixed_kinds_are_errors(self):
        for expression in ['"q: " + a', "{a} + a", "G + 1", '"x" + {1}']:
            with self.subTest(expression=expression):
                with self.assertRaises(RuntimeErrorLumos):
                    value_of(expression, GRAPH)


class LogicTests(unittest.TestCase):
    def test_truth_tables(self):
        self.assertFalse(value_of("true and false"))
        self.assertTrue(value_of("false or true"))
        self.assertTrue(value_of("not false"))

    def test_and_or_short_circuit(self):
        self.assertFalse(value_of("false and (1 / 0 = 1)"))
        self.assertTrue(value_of("true or (1 / 0 = 1)"))
        calls = []
        env_after("x := false and (J(1) = 1)", J=lambda v: calls.append(v) or 1)
        self.assertEqual(calls, [])

    def test_operands_must_be_booleans(self):
        # The right operand is only type-checked when short-circuiting lets it run.
        self.assertTrue(value_of("true or 0"))
        for source in ["1 and true", "false or 0", "not 1"]:
            with self.subTest(source=source):
                with self.assertRaises(RuntimeErrorLumos):
                    value_of(source)

    def test_conditions_must_be_booleans(self):
        with self.assertRaises(RuntimeErrorLumos):
            env_after("if 1 then x := 1")
        with self.assertRaises(RuntimeErrorLumos):
            env_after("while 0 do x := 1")

    def test_membership(self):
        self.assertTrue(value_of("2 in {1, 2}"))
        self.assertFalse(value_of("true in {1}"))
        self.assertTrue(value_of("a in G.N", GRAPH))
        self.assertTrue(value_of('"B" in b.L', GRAPH))
        with self.assertRaises(RuntimeErrorLumos):
            value_of("a in G", GRAPH)


class SetTests(unittest.TestCase):
    def test_set_literals_deduplicate(self):
        self.assertEqual(value_of("{1, 2, 1, 1.0}").to_list(), [1, 2])

    def test_range(self):
        self.assertEqual(value_of("{0, ..., 3}").to_list(), [0, 1, 2, 3])
        self.assertEqual(value_of("{2, ..., 1}").to_list(), [])
        with self.assertRaises(RuntimeErrorLumos):
            value_of("{0, ..., 1.5}")

    def test_set_builder_filters_and_maps(self):
        self.assertEqual(value_of("{v | v in {1, ..., 6} and 3 < v}").to_list(), [4, 5, 6])
        self.assertEqual(value_of("{v * 10 | v in {1, 2}}").to_list(), [10, 20])
        self.assertEqual(value_of("{v.L[0] | v in G.N and not (v = b)}", GRAPH).to_list(), ["A", "C", "D"])

    def test_set_builder_variable_is_local(self):
        env = env_after("v := 99\nS := {v | v in {1, 2}}\nT := {w | w in {1}}")
        self.assertEqual(env["v"], 99)
        self.assertNotIn("w", env)

    def test_indexing(self):
        self.assertEqual(value_of("{5, 6, 7}[0]"), 5)
        self.assertEqual(value_of("{5, 6, 7}[-1]"), 7)  # [-i] = (n - i)-th element
        self.assertEqual(value_of("c.L[-1]", GRAPH), "ctx")
        self.assertEqual(value_of('"abc"[1]'), "b")
        with self.assertRaises(UndefinedValue):
            value_of("{5}[1]")
        with self.assertRaises(UndefinedValue):
            value_of("d.L[1]", GRAPH)
        with self.assertRaises(RuntimeErrorLumos):
            value_of("{5}[0.5]")

    def test_concat_and_len(self):
        self.assertEqual(value_of('concat({"a", "b", "c"})'), "abc")
        self.assertEqual(value_of("concat({v.L[0] | v in G.N})", GRAPH), "ABCD")
        self.assertEqual(value_of("len(M)", M=["x", "y"]), 2)
        self.assertEqual(value_of("len(G.N)", GRAPH), 4)


class GraphTests(unittest.TestCase):
    def test_members(self):
        env = env_after(GRAPH + "N := G.N\nE := G.E\nattrs := a.L\nrel := ab.L")
        self.assertEqual(names(env["N"]), ["a", "b", "c", "d"])
        self.assertEqual(len(env["E"]), 3)
        self.assertEqual(env["attrs"], ["A", "alpha"])
        self.assertEqual(env["rel"], ["ab-rel"])
        with self.assertRaises(RuntimeErrorLumos):
            value_of("a.N", GRAPH)

    def test_neighbours_are_undirected_and_in_node_order(self):
        env = env_after(GRAPH + "na := NB(a)\nnb := NB(b)\nnc := NB(c)\nnd := NB(d)")
        self.assertEqual(names(env["na"]), ["b", "c"])
        self.assertEqual(names(env["nb"]), ["a", "c"])
        self.assertEqual(names(env["nc"]), ["a", "b"])
        self.assertEqual(names(env["nd"]), [])

    def test_self_loop_is_its_own_neighbour(self):
        env = env_after('v := Node(L{"v"})\nl := Edge(v, v)\nG := Graph({v}, {l})\nn := NB(v)')
        self.assertEqual(names(env["n"]), ["v"])

    def test_nb_uses_the_program_graph_and_explicit_graphs(self):
        env = env_after(GRAPH + "H := Graph({a, b}, {ab})\nn := NB(c, G)\nm := NB(a, H)")
        self.assertEqual(names(env["n"]), ["a", "b"])
        self.assertEqual(names(env["m"]), ["b"])
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            env_after(GRAPH + "H := Graph({a}, {})\nn := NB(a)")
        self.assertIn("ambiguous", str(ctx.exception))
        with self.assertRaises(RuntimeErrorLumos):
            env_after('v := Node(L{})\nn := NB(v)')

    def test_graph_construction_rules(self):
        with self.assertRaises(RuntimeErrorLumos):  # endpoint missing from the node set
            env_after(GRAPH + "H := Graph({a}, {ab})")
        with self.assertRaises(RuntimeErrorLumos):  # two edges between one pair
            env_after(GRAPH + 'ab2 := Edge(b, a, L{"other"})\nH := Graph({a, b}, {ab, ab2})')
        with self.assertRaises(RuntimeErrorLumos):  # not a node
            env_after(GRAPH + 'H := Graph({a, "b"}, {})')
        with self.assertRaises(RuntimeErrorLumos):
            env_after('x := 1\ne := Edge(x, x)')

    def test_graph_binding_names_the_node_and_edge_sets(self):
        # g := Graph(N, E) := e  (paper section 4): g is the graph e, N its nodes, E its edges.
        env = env_after(GRAPH + "H := Graph(N, E) := G")
        self.assertIs(env["H"], env["G"])
        self.assertEqual(names(env["N"]), ["a", "b", "c", "d"])
        self.assertEqual(list(env["E"]), [env["ab"], env["bc"], env["ac"]])
        self.assertIsInstance(env["N"], OrderedSet)
        # e is any graph expression; a union, and a bare edge or node lifted to a graph.
        env = env_after(GRAPH + "H := Graph(N, E) := ab + d")
        self.assertIsInstance(env["H"], Graph)
        self.assertEqual(names(env["N"]), ["a", "b", "d"])
        self.assertEqual(list(env["E"]), [env["ab"]])
        env = env_after(GRAPH + "H := Graph(N, E) := d")
        self.assertEqual((names(env["N"]), list(env["E"])), (["d"], []))
        # Rebinding replaces all three names.
        env = env_after(GRAPH + "H := Graph(N, E) := G\nH := Graph(N, E) := ab")
        self.assertEqual((names(env["H"].nodes), names(env["N"]), len(env["E"])), (["a", "b"], ["a", "b"], 1))

    def test_graph_binding_needs_a_graph(self):
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            env_after(GRAPH + "H := Graph(N, E) := 3")
        self.assertNotIsInstance(ctx.exception, UndefinedValue)
        self.assertIn("needs e to be a graph", str(ctx.exception))
        with self.assertRaises(UndefinedValue):  # a missing edge is null, and using null is ⊥
            env_after(GRAPH + "H := Graph(N, E) := getEdge(a, d, G)")

    def test_graph_binding_declares_a_program_graph(self):
        only = 'a := Node(L{"A"})\nb := Node(L{"B"})\nab := Edge(a, b)\nG := Graph(N, E) := ab\nn := NB(a)'
        self.assertEqual(names(env_after(only)["n"]), ["b"])
        # Naming the sets of the same graph does not make NB ambiguous; a second graph does.
        self.assertEqual(names(env_after(GRAPH + "H := Graph(N, E) := G\nn := NB(a)")["n"]), ["b", "c"])
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            env_after(GRAPH + "H := Graph(N, E) := ab + d\nn := NB(a)")
        self.assertIn("ambiguous", str(ctx.exception))

    def test_graph_binding_sets_can_be_sampled(self):
        _, results = run(GRAPH + 'H := Graph(N, E) := G\nestimateProb 0.05 60 "CP" :\n'
                                 '    v := sample(meas(N))\n    e := sample(meas(E))\n'
                                 'toEstimate(v in G.N and e in G.E and len(N) = 4 and len(E) = 3)')
        self.assertEqual(results[0].successes, 60)

    def test_get_edge_either_order(self):
        env = env_after(GRAPH + "e := getEdge(c, a, G)\nr := getEdge(a, c, G).L[0]")
        self.assertEqual(env["e"], env["ac"])
        self.assertEqual(env["r"], "ac-rel")


class NullTests(unittest.TestCase):
    """getEdge with no edge gives null; comparisons with null are false; other uses are ⊥."""

    def test_missing_edge_is_null(self):
        env = env_after(GRAPH + "e := getEdge(a, d, G)\nl := e.L\nfirst := e.L[0]")
        self.assertIs(env["e"], NULL)
        self.assertIs(env["l"], NULL)
        self.assertIs(env["first"], NULL)

    def test_comparisons_with_null_are_false(self):
        env = env_after(GRAPH + 'x := getEdge(a, d, G).L[0] = "ab-rel"\ny := not (getEdge(a, d, G).L[0] = "z")\n'
                        'z := getEdge(a, d, G) in G.E\nw := getEdge(a, d, G) = getEdge(a, d, G)')
        self.assertEqual((env["x"], env["y"], env["z"], env["w"]), (False, True, False, False))

    def test_attribute_of_an_edge_without_attributes_is_undefined(self):
        with self.assertRaises(UndefinedValue):
            value_of("getEdge(x, y, H).L[0]", 'x := Node(L{"x"})\ny := Node(L{"y"})\n'
                     'xy := Edge(x, y)\nH := Graph({x, y}, {xy})')

    def test_null_filters_out_of_set_builders(self):
        self.assertEqual(names(value_of('{v | v in G.N and getEdge(v, c, G).L[0] = "ac-rel"}', GRAPH)), ["a"])

    def test_other_uses_of_null_are_undefined(self):
        for expression in ['"p" + getEdge(a, d, G).L[0]', "{getEdge(a, d, G)}", "J(getEdge(a, d, G))",
                           "not getEdge(a, d, G)", "format(\"x\", getEdge(a, d, G))"]:
            with self.subTest(expression=expression):
                with self.assertRaises(UndefinedValue):
                    value_of(expression, GRAPH, J=lambda e: 1)
        with self.assertRaises(UndefinedValue):
            env_after(GRAPH + "if getEdge(a, d, G) = ab or getEdge(a, d, G) then x := 1")


class FormatTests(unittest.TestCase):
    def test_placeholders_of_elements_in_the_graph(self):
        env = env_after(GRAPH + 'g := b + d + ab\ns := format("{b} and {d} via {ab}", g)')
        self.assertEqual(env["s"], "B and D via ab-rel")

    def test_other_placeholders_are_left_as_written(self):
        env = env_after(GRAPH + 'name := "x"\ng := b + d\ns := format("{a} {name} {nope} {d}", g)')
        self.assertEqual(env["s"], "{a} {name} {nope} D")  # a is not in g

    def test_attribute_is_sampled_with_meas(self):
        interp, _ = run(GRAPH + "\n".join(f's{i} := format("{{a}}", G)' for i in range(400)))
        seen = collections.Counter(interp.env.bindings[f"s{i}"] for i in range(400))
        self.assertEqual(set(seen), {"A", "alpha"})
        self.assertGreater(min(seen.values()), 150)  # uniform over the two attributes
        # A program-supplied meas decides the attribute (here: always the first one).
        first_only = lambda elements: [1] + [0] * (len(elements) - 1)
        self.assertEqual(value_of('format("{c}", G)', GRAPH, meas=first_only), "C")

    def test_empty_attribute_list_is_undefined(self):
        with self.assertRaises(UndefinedValue):
            value_of('format("{v}", v + w)', 'v := Node(L{})\nw := Node(L{"w"})')

    def test_needs_a_graph(self):
        with self.assertRaises(RuntimeErrorLumos):
            value_of('format("{a}", a)', GRAPH)


class SamplingTests(unittest.TestCase):
    def draws(self, source, n=3000, **externals):
        program = "\n".join(f"x{i} := {source}" for i in range(n))
        interp, _ = run(program, seed=7, **externals)
        return collections.Counter(interp.env.bindings[f"x{i}"] for i in range(n))

    def test_default_meas_is_uniform(self):
        counts = self.draws("sample(meas({1, 2, 3}))")
        for k in (1, 2, 3):
            self.assertLess(abs(counts[k] / 3000 - 1 / 3), 0.04)

    def test_program_supplied_meas(self):
        counts = self.draws("sample(meas({1, 2, 3}))", meas=lambda xs: [0, 3, 1])
        self.assertEqual(counts[1], 0)
        self.assertLess(abs(counts[2] / 3000 - 0.75), 0.04)

    def test_all_zero_weights_are_uniform(self):
        counts = self.draws("sample(meas({1, 2}))", meas=lambda xs: [0, 0])
        self.assertLess(abs(counts[1] / 3000 - 0.5), 0.04)

    def test_bad_weights(self):
        for weights in ([-1, 2], [1], [float("nan"), 1], ["1", 1]):
            with self.subTest(weights=weights):
                with self.assertRaises(RuntimeErrorLumos):
                    value_of("sample(meas({1, 2}))", meas=lambda xs, w=weights: w)

    def test_registered_measures_and_unknown_names(self):
        self.assertEqual(value_of("sample(one_hot({7, 8}))"), 7)
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            value_of("sample(mesa({1}))")
        self.assertIn("Unknown measure 'mesa'", str(ctx.exception))

    def test_sampling_the_empty_set_is_undefined(self):
        with self.assertRaises(UndefinedValue):
            value_of("sample(meas({}))")

    def test_sampling_nodes_gives_nodes(self):
        self.assertIsInstance(value_of("sample(meas(G.N))", GRAPH), AttributedNode)

    def test_same_seed_same_draws(self):
        program = "\n".join(f"x{i} := sample(meas({{1, ..., 100}}))" for i in range(20))
        first = env_after(program, seed=3)
        second = env_after(program, seed=3)
        other = env_after(program, seed=4)
        draws = lambda env: [env[f"x{i}"] for i in range(20)]
        self.assertEqual(draws(first), draws(second))
        self.assertNotEqual(draws(first), draws(other))


class ExternalCallTests(unittest.TestCase):
    def test_calls_and_tool_calls(self):
        prompts = []
        env = env_after('r := L("hi")\nimg := tool-call(Scenic, "cmd")\nj := J(r, img)',
                        L=lambda p: prompts.append(p) or "resp:" + p,
                        Scenic=lambda c: "img:" + c,
                        J=lambda r, i: 1 if r == "resp:hi" and i == "img:cmd" else 0)
        self.assertEqual(prompts, ["hi"])
        self.assertEqual(env["j"], 1)

    def test_errors_inside_externals_are_reported_with_the_name(self):
        def broken(prompt):
            raise ValueError("boom")
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            env_after("x := 1\nr := L(\"p\")", L=broken)
        self.assertIn("L(...) failed: ValueError: boom", str(ctx.exception))
        self.assertEqual(ctx.exception.line, 2)

    def test_not_callable(self):
        with self.assertRaises(RuntimeErrorLumos):
            env_after("x := 1\ny := x(2)")
        with self.assertRaises(RuntimeErrorLumos):
            env_after("t := 1\ny := tool-call(t, \"p\")")

    def test_while_loop_guard(self):
        from lumos.interpreter import Interpreter
        old = Interpreter.max_loop_iterations
        Interpreter.max_loop_iterations = 50
        try:
            with self.assertRaises(RuntimeErrorLumos):
                env_after("while true do x := 1")
        finally:
            Interpreter.max_loop_iterations = old


class GetGraphTests(unittest.TestCase):
    def write(self, text):
        directory = tempfile.mkdtemp()
        path = Path(directory) / "kg.csv"
        path.write_text(text)
        return directory

    def load(self, text):
        from lumos.interpreter import Interpreter
        from lumos_test_utils import parse
        interp = Interpreter(base_dir=self.write(text))
        interp.eval_program(parse('G := getGraph("kg.csv")'))
        return interp.env.get("G")

    def test_edge_list(self):
        g = self.load(
            "src_id,src_type,src_name,dst_id,dst_type,dst_name,relation,source\n"
            "d1,drug,Aspirin,x1,disease,Flu,treat,db\n"
            "d1,drug,Aspirin,x2,disease,Cold,contraindicate,db\n"
            "d2,drug,Ibuprofen,,,,,\n"
        )
        self.assertEqual(names(g.nodes), ["d1", "x1", "x2", "d2"])
        self.assertEqual(g.node_named("d1").attributes, ["drug", "Aspirin"])
        self.assertEqual(g.node_named("x2").attributes, ["disease", "Cold"])
        self.assertEqual(g.get_edge(g.node_named("x1"), g.node_named("d1")).attributes, ["treat", "db"])
        self.assertEqual(g.neighbors(g.node_named("d2")), [])

    def test_errors(self):
        for text in ["a,b\n1,2\n",
                     "src_id,src_t,dst_id,dst_t\nd1,drug,x1,dis\nd1,DRUG,x2,dis\n",
                     "src_id,dst_id,rel\nd1,x1,a\nx1,d1,b\n",
                     "src_id,dst_id\nd1\n"]:
            with self.subTest(text=text):
                with self.assertRaises(RuntimeErrorLumos):
                    self.load(text)

    def test_missing_file(self):
        with self.assertRaises(RuntimeErrorLumos):
            env_after('G := getGraph("/nonexistent/kg.csv")')


class RuntimeValueTests(unittest.TestCase):
    def test_ordered_set_keeps_booleans_and_numbers_apart(self):
        s = OrderedSet([1, True, 0, False])
        self.assertEqual(len(s), 4)
        self.assertIn(True, s)
        self.assertNotIn(True, OrderedSet([1]))

    def test_ordered_set_with_unhashable_elements(self):
        s = OrderedSet([[1], [1], [2]])
        self.assertEqual(s.to_list(), [[1], [2]])
        self.assertIn([2], s)

    def test_graph_values_are_shared_between_trials(self):
        import copy
        g = Graph()
        n = g.add_node(AttributedNode("n", ["x"]))
        self.assertIs(copy.deepcopy(g), g)
        self.assertIs(copy.deepcopy(n), n)
        self.assertIs(copy.deepcopy(AttributedEdge(n, n, [])).src_node, n)


if __name__ == "__main__":
    unittest.main()
