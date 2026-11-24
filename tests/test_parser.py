"""Parser: statements, layout of blocks, estimateProb, expressions and errors."""

import unittest

from lumos_test_utils import parse
from lumos.ast import *
from lumos.errors import ParseError


def expr(source):
    """Parse a single expression."""
    return parse(f"x := {source}").statements[0].value


def estimate(source):
    return [s for s in parse(source).statements if isinstance(s, EstimateProbStatement)][0]


class StatementTests(unittest.TestCase):
    def test_assignment_and_line(self):
        stmt = parse("\n\nx := 1").statements[0]
        self.assertEqual(stmt, AssignStatement("x", NumberLiteral(1)))
        self.assertEqual(stmt.line, 3)

    def test_semicolons(self):
        self.assertEqual(len(parse("a := 1; b := 2; c := 3;").statements), 3)

    def test_node_edge_graph_declarations(self):
        prog = parse('v1 := Node(L{"q", "0.9"})\nv2 := Node(L{})\n'
                     'e := Edge(v1, v2, L{"ahead"})\nf := Edge(v1, v2)\nG := Graph({v1, v2}, {e})')
        v1, v2, e, f, g = [s.value for s in prog.statements]
        self.assertEqual(v1, NodeConstructor("v1", ["q", "0.9"]))
        self.assertEqual(v2, NodeConstructor("v2", []))
        self.assertEqual(e, EdgeConstructor(Identifier("v1"), Identifier("v2"), ["ahead"]))
        self.assertEqual(f.attributes, [])
        self.assertIsInstance(g, GraphConstructor)

    def test_graph_binding_names_the_node_and_edge_sets(self):
        stmt = parse('\nH := Graph(N, E) := G + v').statements[0]
        self.assertIsInstance(stmt, GraphBindStatement)
        self.assertEqual((stmt.name, stmt.nodes_name, stmt.edges_name, stmt.line), ("H", "N", "E", 2))
        self.assertEqual(stmt.value, BinaryOperation(Identifier("G"), "PLUS", Identifier("v")))
        # Without the second ':=' it is the ordinary constructor.
        plain = parse('H := Graph(N, E)').statements[0]
        self.assertIsInstance(plain, AssignStatement)
        self.assertIsInstance(plain.value, GraphConstructor)

    def test_graph_binding_errors(self):
        for source, fragment in [
            ('H := Graph({a}, E) := G', "variable names"),
            ('H := Graph(N, G.E) := G', "variable names"),
            ('H := Graph(H, E) := G', "three different names"),
            ('H := Graph(N, N) := G', "three different names"),
            ('x := 1 := 2', "':='"),              # only Graph(N, E) can be chained
            ('x := y := Graph(N, E)', "':='"),
        ]:
            with self.subTest(source=source):
                with self.assertRaises(ParseError) as ctx:
                    parse(source)
                self.assertIn(fragment, str(ctx.exception))

    def test_inline_node_is_rejected(self):
        with self.assertRaises(ParseError) as ctx:
            parse('G := Graph({Node(L{"a"})}, {})')
        self.assertIn("own variable", str(ctx.exception))

    def test_attribute_list_rules(self):
        for bad in ['v := Node({"a"})', 'v := Node(L{1})', 'x := L{"a"}', 'v := Node(L{"a"}, L{"b"})']:
            with self.subTest(bad=bad):
                with self.assertRaises(ParseError):
                    parse(bad)

    def test_assignment_with_equals_gets_hint(self):
        with self.assertRaises(ParseError) as ctx:
            parse("x = 1")
        self.assertIn("':='", str(ctx.exception))

    def test_old_dialect_and_bare_expressions_are_rejected(self):
        for bad in ["let x = 1", "L(p)", "x := 1 y := 2", "if (a) { x := 1 }"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ParseError):
                    parse(bad)


class IfWhileTests(unittest.TestCase):
    def test_inline_if_else(self):
        stmt = parse('if a = b then g := "yes" else g := "no"').statements[0]
        self.assertIsInstance(stmt, IfStatement)
        self.assertEqual(stmt.then_block, [AssignStatement("g", StringLiteral("yes"))])
        self.assertEqual(stmt.else_block, [AssignStatement("g", StringLiteral("no"))])

    def test_if_without_else(self):
        stmt = parse('if a then g := 1\nh := 2').statements[0]
        self.assertIsNone(stmt.else_block)

    def test_block_branches(self):
        prog = parse("if a then\n    x := 1\n    y := 2\nelse\n    x := 3\nz := 4")
        stmt = prog.statements[0]
        self.assertEqual(len(stmt.then_block), 2)
        self.assertEqual(len(stmt.else_block), 1)
        self.assertEqual(prog.statements[1], AssignStatement("z", NumberLiteral(4)))

    def test_inline_then_with_else_on_next_line(self):
        stmt = parse("if a then x := 1\nelse x := 2").statements[0]
        self.assertEqual(stmt.else_block, [AssignStatement("x", NumberLiteral(2))])

    def test_dangling_else_binds_to_nearest_if(self):
        outer = parse("if a then if b then x := 1 else x := 2").statements[0]
        self.assertIsNone(outer.else_block)
        self.assertIsNotNone(outer.then_block[0].else_block)

    def test_if_then_on_next_line(self):
        self.assertIsInstance(parse("if a and\n   b\nthen x := 1").statements[0], IfStatement)

    def test_while_do(self):
        stmt = parse("while k < 3 do\n    k := k + 1").statements[0]
        self.assertIsInstance(stmt, WhileStatement)
        self.assertEqual(stmt.condition.operator, 'LT')
        self.assertEqual(parse("while k < 3 do k := k + 1").statements[0].body,
                         [AssignStatement("k", BinaryOperation(Identifier("k"), 'PLUS', NumberLiteral(1)))])

    def test_bounded_loop(self):
        stmt = parse("while i in {0, ..., q}\n    x := x + i").statements[0]
        self.assertIsInstance(stmt, ForEachStatement)
        self.assertEqual(stmt.variable, "i")
        self.assertEqual(stmt.iterable, RangeExpression(NumberLiteral(0), Identifier("q")))

    def test_membership_while_with_do_is_an_ordinary_loop(self):
        self.assertIsInstance(parse("while i in S do i := i + 1").statements[0], WhileStatement)

    def test_while_without_do(self):
        with self.assertRaises(ParseError) as ctx:
            parse("while k < 3\n    k := k + 1")
        self.assertIn("'do'", str(ctx.exception))

    def test_unexpected_indentation(self):
        with self.assertRaises(ParseError) as ctx:
            parse("x := 1\n    y := 2")
        self.assertIn("indentation", str(ctx.exception))


class EstimateProbTests(unittest.TestCase):
    def test_to_estimate_after_block(self):
        stmt = estimate('estimateProb 0.05 1000 "Clopper-Pearson" :\n    x := 1\ntoEstimate(x = 1)')
        self.assertEqual((stmt.delta, stmt.trials, stmt.method), (0.05, 1000, "Clopper-Pearson"))
        self.assertEqual(len(stmt.body), 1)
        self.assertEqual(stmt.to_estimate_line, 3)

    def test_to_estimate_as_last_line_of_block(self):
        stmt = estimate('estimateProb 0.05 10 "CP":\n\n    x := 1\n\n    toEstimate(x = 1)\ny := 2')
        self.assertEqual(len(stmt.body), 1)
        self.assertEqual(stmt.to_estimate_line, 5)

    def test_to_estimate_after_semicolon_in_block(self):
        stmt = estimate('estimateProb 0.05 10 "CP":\n    x := 1; toEstimate(x = 1)')
        self.assertEqual(len(stmt.body), 1)

    def test_inline_body(self):
        stmt = estimate('estimateProb 0.05 10 "CP" : x := 1; y := 2; toEstimate(x = y)')
        self.assertEqual(len(stmt.body), 2)
        stmt = estimate('estimateProb 0.05 10 "CP" : x := 1\ntoEstimate(x = 1)')
        self.assertEqual(len(stmt.body), 1)

    def test_statements_after_estimate_prob(self):
        prog = parse('estimateProb 0.05 10 "CP":\n    x := 1\ntoEstimate(x = 1)\ny := 2')
        self.assertEqual(prog.statements[-1], AssignStatement("y", NumberLiteral(2)))

    def test_errors(self):
        cases = {
            'estimateProb 0.05 10 "CP":\n    x := 1\ny := 2': "toEstimate",
            'estimateProb 0.05 10 "CP":\n    toEstimate(true)\n    x := 1': "last line",
            'estimateProb 0.05 10.5 "CP":\n    x := 1\ntoEstimate(true)': "integer",
            'estimateProb 0.05 10 "CP"\n    x := 1\ntoEstimate(true)': "':'",
            'estimateProb 0.05 10 CP:\n    x := 1\ntoEstimate(true)': "certifier",
            'if a then\n    estimateProb 0.05 10 "CP":\n        x := 1\n    toEstimate(true)': "top level",
            'x := 1\ntoEstimate(x = 1)': "toEstimate",
            'estimateProb 0.05 10 "CP":\n    x := 1\ntoEstimate x = 1': "'('",
            'estimateProb 0.05 10 "CP":\n    x := 1\ntoEstimate(x) and (y)': "end of line",
        }
        for source, fragment in cases.items():
            with self.subTest(source=source):
                with self.assertRaises(ParseError) as ctx:
                    parse(source)
                self.assertIn(fragment, str(ctx.exception))


class ExpressionTests(unittest.TestCase):
    def test_boolean_precedence_not_and_or(self):
        self.assertEqual(
            expr("not a and b or c"),
            BinaryOperation(
                BinaryOperation(UnaryOperation('NOT', Identifier('a')), 'AND', Identifier('b')),
                'OR', Identifier('c')),
        )

    def test_not_applies_to_a_whole_comparison(self):
        self.assertEqual(expr("not u = v"),
                         UnaryOperation('NOT', BinaryOperation(Identifier('u'), 'EQ', Identifier('v'))))

    def test_arithmetic_precedence_and_associativity(self):
        self.assertEqual(expr("1 + 2 * 3"),
                         BinaryOperation(NumberLiteral(1), 'PLUS',
                                         BinaryOperation(NumberLiteral(2), 'MULTIPLY', NumberLiteral(3))))
        self.assertEqual(expr("10 - 3 - 2"),
                         BinaryOperation(BinaryOperation(NumberLiteral(10), 'MINUS', NumberLiteral(3)),
                                         'MINUS', NumberLiteral(2)))
        self.assertEqual(expr("-2 * 3"),
                         BinaryOperation(UnaryOperation('MINUS', NumberLiteral(2)), 'MULTIPLY', NumberLiteral(3)))

    def test_in_binds_looser_than_plus(self):
        self.assertEqual(expr("x in S + T"),
                         BinaryOperation(Identifier('x'), 'IN',
                                         BinaryOperation(Identifier('S'), 'PLUS', Identifier('T'))))

    def test_comparisons_do_not_chain(self):
        for source in ["1 < 2 < 3", "a = b = c", "a in S = true"]:
            with self.subTest(source=source):
                with self.assertRaises(ParseError) as ctx:
                    expr(source)
                self.assertIn("chained", str(ctx.exception))

    def test_number_literals(self):
        self.assertEqual(expr("7"), NumberLiteral(7))
        self.assertIsInstance(expr("7").value, int)
        self.assertEqual(expr("0.5"), NumberLiteral(0.5))

    def test_postfix_chains(self):
        self.assertEqual(
            expr("getEdge(d, v, G).L[0]"),
            IndexExpression(MemberExpression(GetEdgeExpression(Identifier('d'), Identifier('v'), Identifier('G')), 'L'),
                            NumberLiteral(0)),
        )
        self.assertEqual(expr("options[0].L[-1]"),
                         IndexExpression(MemberExpression(IndexExpression(Identifier('options'), NumberLiteral(0)), 'L'),
                                         UnaryOperation('MINUS', NumberLiteral(1))))
        self.assertEqual(expr("J(a, b)"), CallExpression(Identifier('J'), [Identifier('a'), Identifier('b')]))
        self.assertEqual(expr("G.N"), MemberExpression(Identifier('G'), 'N'))

    def test_sets(self):
        self.assertEqual(expr("{}"), OrderedSetLiteral([]))
        self.assertEqual(expr("{a, b}"), OrderedSetLiteral([Identifier('a'), Identifier('b')]))
        self.assertEqual(expr("{1, ..., n - 1}"),
                         RangeExpression(NumberLiteral(1), BinaryOperation(Identifier('n'), 'MINUS', NumberLiteral(1))))

    def test_set_builder(self):
        builder = expr("{v.L[0] | v in G.N and v.L[0] = \"drug\"}")
        self.assertIsInstance(builder, SetBuilder)
        self.assertEqual(builder.variable, "v")
        self.assertEqual(builder.source, MemberExpression(Identifier('G'), 'N'))
        self.assertEqual(builder.condition.operator, 'EQ')
        self.assertIsNone(expr("{v | v in S}").condition)

    def test_set_builder_predicate_may_use_or(self):
        # Paper Alg. 2: {v | v in NB(n) and (C1) or (C2)} keeps the binder and filters on C1 or C2.
        builder = expr("{v | v in NB(n) and (c1) or (c2)}")
        self.assertIsInstance(builder.source, NBExpression)
        self.assertEqual(builder.condition.operator, 'OR')

    def test_set_errors(self):
        for bad in ["{a, b,}", "{a, b, ..., c}", "{1, ...}", "{v | in S}", "{v | v S}"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ParseError):
                    expr(bad)

    def test_keyword_forms(self):
        self.assertEqual(expr("sample(meas(S))"), SampleExpression("meas", Identifier('S')))
        self.assertEqual(expr("NB(v)"), NBExpression(Identifier('v'), None))
        self.assertEqual(expr("NB(v, G)"), NBExpression(Identifier('v'), Identifier('G')))
        self.assertEqual(expr("format(\"{x}\", g)"), FormatExpression(StringLiteral("{x}"), Identifier('g')))
        self.assertEqual(expr("tool-call(Scenic, p)"), ToolCallExpression(Identifier('Scenic'), Identifier('p')))
        self.assertEqual(expr("Graph(N, E)"), GraphConstructor(Identifier('N'), Identifier('E')))

    def test_keyword_form_errors(self):
        for bad in ["sample(S)", "sample(meas(S), 1)", "getEdge(a, b)", "format(s)", "NB()", "NB(a, b, c)",
                    "tool-call(p)"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ParseError):
                    expr(bad)


if __name__ == "__main__":
    unittest.main()
