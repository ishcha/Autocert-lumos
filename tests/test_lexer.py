"""Lexer: tokens, strings, comments and the indentation layout."""

import unittest

from lumos_test_utils import token_types
from lumos.errors import ParseError
from lumos.lexer import Lexer


def values(source):
    return [(t.type, t.value) for t in Lexer(source).tokenize() if t.type not in ('NEWLINE', 'EOF')]


class TokenTests(unittest.TestCase):
    def test_operators_longest_match(self):
        self.assertEqual(
            [t for t, _ in values("x := a <= b >= c < d > e = f : ... . | + - * /")],
            ['IDENTIFIER', 'DEFINE', 'IDENTIFIER', 'LE', 'IDENTIFIER', 'GE', 'IDENTIFIER', 'LT',
             'IDENTIFIER', 'GT', 'IDENTIFIER', 'EQ', 'IDENTIFIER', 'COLON', 'ELLIPSIS', 'DOT',
             'PIPE', 'PLUS', 'MINUS', 'MULTIPLY', 'DIVIDE'],
        )

    def test_keywords_and_identifiers(self):
        self.assertEqual(
            values("in inside not note L meas NB getEdge estimateProb toEstimate Node Edge Graph"),
            [('IN', 'in'), ('IDENTIFIER', 'inside'), ('NOT', 'not'), ('IDENTIFIER', 'note'),
             ('IDENTIFIER', 'L'), ('IDENTIFIER', 'meas'), ('NB', 'NB'), ('GETEDGE', 'getEdge'),
             ('ESTIMATEPROB', 'estimateProb'), ('TOESTIMATE', 'toEstimate'), ('NODE', 'Node'),
             ('EDGE', 'Edge'), ('GRAPH', 'Graph')],
        )

    def test_tool_call_is_one_token_but_subtraction_is_not(self):
        self.assertEqual(values("tool-call(t, p)")[0], ('TOOLCALL', 'tool-call'))
        self.assertEqual([t for t, _ in values("tool - call")], ['IDENTIFIER', 'MINUS', 'IDENTIFIER'])
        self.assertEqual([t for t, _ in values("tool-caller")], ['IDENTIFIER', 'MINUS', 'IDENTIFIER'])

    def test_numbers(self):
        self.assertEqual(values("0.05 1000 7"), [('NUMBER', '0.05'), ('NUMBER', '1000'), ('NUMBER', '7')])
        # '1.' is a number followed by a dot; the parser rejects it.
        self.assertEqual([t for t, _ in values("1.")], ['NUMBER', 'DOT'])

    def test_member_access_after_index(self):
        self.assertEqual(
            [t for t, _ in values("options[0].L[1]")],
            ['IDENTIFIER', 'LBRACKET', 'NUMBER', 'RBRACKET', 'DOT', 'IDENTIFIER', 'LBRACKET',
             'NUMBER', 'RBRACKET'],
        )


class StringTests(unittest.TestCase):
    def string(self, source):
        return values(source)[0][1]

    def test_escapes(self):
        self.assertEqual(self.string(r'"a\nb"'), "a\nb")
        self.assertEqual(self.string(r'"tab\there"'), "tab\there")
        self.assertEqual(self.string(r'"say \"hi\""'), 'say "hi"')

    def test_escaped_backslash_before_n_is_not_a_newline(self):
        self.assertEqual(self.string(r'"x\\ny"'), "x\\ny")
        self.assertEqual(self.string(r'"\\"'), "\\")

    def test_unknown_escape_is_kept(self):
        self.assertEqual(self.string(r'"a\qb"'), "a\\qb")

    def test_hash_inside_string_is_not_a_comment(self):
        self.assertEqual(self.string('"# not a comment"'), "# not a comment")

    def test_unterminated_string(self):
        with self.assertRaises(ParseError) as ctx:
            Lexer('x := "abc\ny := 1').tokenize()
        self.assertEqual(ctx.exception.line, 1)
        self.assertIn("Unterminated", str(ctx.exception))

    def test_triple_quoted_wrapper_is_rejected(self):
        with self.assertRaises(ParseError):
            Lexer('"""\nx := 1\n"""').tokenize()


class ErrorTests(unittest.TestCase):
    def test_foreign_operators_get_hints(self):
        for op, hint in [('==', "'='"), ('!=', "not (a = b)"), ('&&', "'and'"),
                         ('||', "'or'"), ('!', "'not'"), ('//', "'#'")]:
            with self.subTest(op=op):
                with self.assertRaises(ParseError) as ctx:
                    Lexer(f"x := a {op} b").tokenize()
                self.assertIn(hint, str(ctx.exception))

    def test_unknown_character_reports_line(self):
        with self.assertRaises(ParseError) as ctx:
            Lexer("x := 1\n\n# c\ny := @").tokenize()
        self.assertEqual(ctx.exception.line, 4)


class LayoutTests(unittest.TestCase):
    def test_lines_are_separated_by_newline(self):
        self.assertEqual(
            token_types("a := 1\nb := 2"),
            ['IDENTIFIER', 'DEFINE', 'NUMBER', 'NEWLINE', 'IDENTIFIER', 'DEFINE', 'NUMBER', 'NEWLINE', 'EOF'],
        )

    def test_blank_and_comment_lines_are_ignored(self):
        self.assertEqual(token_types("a := 1\n\n   # comment\n\nb := 2  # trailing"),
                         token_types("a := 1\nb := 2"))

    def test_indent_and_nested_dedent(self):
        source = "while x do\n    if y then\n        z := 1\nw := 2"
        self.assertEqual(
            token_types(source),
            ['WHILE', 'IDENTIFIER', 'DO', 'NEWLINE', 'INDENT', 'IF', 'IDENTIFIER', 'THEN', 'NEWLINE',
             'INDENT', 'IDENTIFIER', 'DEFINE', 'NUMBER', 'NEWLINE', 'DEDENT', 'DEDENT',
             'IDENTIFIER', 'DEFINE', 'NUMBER', 'NEWLINE', 'EOF'],
        )

    def test_blocks_are_closed_at_end_of_file(self):
        self.assertEqual(token_types("while x do\n    y := 1")[-3:], ['NEWLINE', 'DEDENT', 'EOF'])

    def test_inconsistent_dedent(self):
        with self.assertRaises(ParseError) as ctx:
            Lexer("while x do\n    a := 1\n  b := 2").tokenize()
        self.assertEqual(ctx.exception.line, 3)

    def test_open_brackets_continue_lines(self):
        self.assertEqual(token_types("S := {a,\n        b}\nT := 1").count('NEWLINE'), 2)
        self.assertNotIn('INDENT', token_types("S := {a,\n        b}"))

    def test_leading_operator_continues_line(self):
        # As in cat_risk_adaptive.lumos: a line starting with '+'.
        types = token_types("x := {a}\n     + {b}\ny := 1")
        self.assertNotIn('INDENT', types)
        self.assertEqual(types.count('NEWLINE'), 2)

    def test_trailing_operator_continues_line(self):
        types = token_types("x := a and\n    b\ny := 1")
        self.assertNotIn('INDENT', types)
        self.assertEqual(types.count('NEWLINE'), 2)

    def test_then_and_do_on_next_line_continue(self):
        self.assertNotIn('NEWLINE', token_types("if a\nthen x := 1")[:-2])
        self.assertNotIn('NEWLINE', token_types("while a\ndo x := 1")[:-2])

    def test_else_line_closes_then_block(self):
        types = token_types("if a then\n    x := 1\nelse\n    x := 2")
        index = types.index('ELSE')
        self.assertEqual(types[index - 2:index], ['NEWLINE', 'DEDENT'])

    def test_crlf_and_tabs(self):
        self.assertEqual(token_types("while a do\r\n\tx := 1\r\ny := 2\r\n"),
                         token_types("while a do\n        x := 1\ny := 2\n"))

    def test_line_numbers(self):
        tokens = Lexer("a := 1\n\n# c\nb := 2").tokenize()
        b = [t for t in tokens if t.value == 'b'][0]
        self.assertEqual(b.line, 4)

    def test_byte_order_mark_is_ignored(self):
        self.assertEqual(token_types("﻿a := 1"), token_types("a := 1"))


if __name__ == "__main__":
    unittest.main()
