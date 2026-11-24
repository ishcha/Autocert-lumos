"""Lexer for Lumos.

Besides ordinary tokens, the lexer emits Python-style layout tokens:

- NEWLINE ends a logical line,
- INDENT / DEDENT open and close an indented block.

A physical line continues the previous logical line (no NEWLINE is emitted)
when a bracket is still open, when the previous line ends with an operator or
a comma, or when the line itself starts with an operator such as a leading
'+'. Blank lines and comment-only lines are ignored. Comments start with '#'.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .errors import ParseError


@dataclass
class Token:
    """A token with the line it starts on."""
    type: str
    value: str
    line: int


KEYWORDS = {
    'if': 'IF', 'then': 'THEN', 'else': 'ELSE',
    'while': 'WHILE', 'do': 'DO',
    'in': 'IN', 'and': 'AND', 'or': 'OR', 'not': 'NOT',
    'true': 'TRUE', 'false': 'FALSE',
    'estimateProb': 'ESTIMATEPROB', 'toEstimate': 'TOESTIMATE',
    'sample': 'SAMPLE', 'format': 'FORMAT',
    'NB': 'NB', 'getEdge': 'GETEDGE',
    'Node': 'NODE', 'Edge': 'EDGE', 'Graph': 'GRAPH',
}

# Order matters: the first alternative that matches wins.
_TOKEN_SPEC = [
    ('NUMBER', r'\d+(?:\.\d+)?'),
    ('STRING', r'"(?:[^"\\\n]|\\[^\n])*"'),
    ('TOOLCALL', r'tool-call(?![A-Za-z0-9_])'),
    ('IDENTIFIER', r'[A-Za-z_][A-Za-z0-9_]*'),
    ('ELLIPSIS', r'\.\.\.'),
    ('DEFINE', r':='),
    ('LE', r'<='),
    ('GE', r'>='),
    ('FOREIGN', r'==|!=|&&|\|\||//|!'),
    ('LT', r'<'),
    ('GT', r'>'),
    ('EQ', r'='),
    ('PLUS', r'\+'),
    ('MINUS', r'-'),
    ('MULTIPLY', r'\*'),
    ('DIVIDE', r'/'),
    ('PIPE', r'\|'),
    ('DOT', r'\.'),
    ('COLON', r':'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('LBRACE', r'\{'),
    ('RBRACE', r'\}'),
    ('LBRACKET', r'\['),
    ('RBRACKET', r'\]'),
    ('COMMA', r','),
    ('SEMICOLON', r';'),
]
_MASTER = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in _TOKEN_SPEC))

# Operators from other languages, rejected with a hint.
_FOREIGN_HINTS = {
    '==': "use '=' for equality",
    '!=': "use 'not (a = b)'",
    '&&': "use 'and'",
    '||': "use 'or'",
    '!': "use 'not'",
    '//': "comments start with '#'",
}

_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\'}

# A line ending in one of these continues on the next physical line.
_CONTINUES_AFTER = {
    'PLUS', 'MINUS', 'MULTIPLY', 'DIVIDE', 'AND', 'OR', 'NOT',
    'EQ', 'LT', 'LE', 'GT', 'GE', 'IN', 'PIPE', 'COMMA', 'DEFINE', 'DOT',
}
# A line starting with one of these continues the previous line. ELSE is
# deliberately absent: an `else` line may close an indented `then` block.
_CONTINUES_BEFORE = {
    'PLUS', 'MINUS', 'MULTIPLY', 'DIVIDE', 'AND', 'OR',
    'EQ', 'LT', 'LE', 'GT', 'GE', 'IN', 'PIPE', 'COMMA', 'DOT', 'ELLIPSIS',
    'RPAREN', 'RBRACE', 'RBRACKET', 'THEN', 'DO',
}
_OPENERS = {'LPAREN', 'LBRACE', 'LBRACKET'}
_CLOSERS = {'RPAREN', 'RBRACE', 'RBRACKET'}


def _unescape(body: str) -> str:
    """Process escapes in one left-to-right pass; unknown escapes are kept."""
    return re.sub(r'\\(.)', lambda m: _ESCAPES.get(m.group(1), m.group(0)), body)


class Lexer:
    """Tokenizes Lumos source code."""

    def __init__(self, source: str):
        self.source = source[1:] if source.startswith('﻿') else source

    def tokenize(self) -> List[Token]:
        tokens, indents = self._scan()
        return self._layout(tokens, indents)

    def _scan(self) -> Tuple[List[Token], Dict[int, int]]:
        """Produce raw tokens plus the indentation width of every line."""
        src = self.source
        n = len(src)
        pos, line = 0, 1
        at_line_start = True
        tokens: List[Token] = []
        indents: Dict[int, int] = {}
        while pos < n:
            if at_line_start:
                width = 0
                while pos < n and src[pos] in ' \t':
                    width = (width // 8 + 1) * 8 if src[pos] == '\t' else width + 1
                    pos += 1
                indents[line] = width
                at_line_start = False
                continue
            ch = src[pos]
            if ch == '\n':
                line += 1
                pos += 1
                at_line_start = True
                continue
            if ch in ' \t\r\f':
                pos += 1
                continue
            if ch == '#':
                while pos < n and src[pos] != '\n':
                    pos += 1
                continue
            match = _MASTER.match(src, pos)
            if match is None:
                if ch == '"':
                    raise ParseError("Unterminated string literal", line)
                raise ParseError(f"Unexpected character '{ch}'", line)
            kind, text = match.lastgroup, match.group()
            if kind == 'FOREIGN':
                raise ParseError(f"'{text}' is not Lumos syntax: {_FOREIGN_HINTS[text]}", line)
            if kind == 'IDENTIFIER':
                kind = KEYWORDS.get(text, 'IDENTIFIER')
            value = _unescape(text[1:-1]) if kind == 'STRING' else text
            tokens.append(Token(kind, value, line))
            pos = match.end()
        return tokens, indents

    @staticmethod
    def _layout(tokens: List[Token], indents: Dict[int, int]) -> List[Token]:
        """Insert NEWLINE / INDENT / DEDENT tokens."""
        out: List[Token] = []
        stack = [0]
        depth = 0
        prev: Token | None = None
        for tok in tokens:
            if prev is None or tok.line != prev.line:
                continuation = prev is not None and (
                    depth > 0
                    or prev.type in _CONTINUES_AFTER
                    or tok.type in _CONTINUES_BEFORE
                )
                if not continuation:
                    if prev is not None:
                        out.append(Token('NEWLINE', '', prev.line))
                    width = indents.get(tok.line, 0)
                    if width > stack[-1]:
                        stack.append(width)
                        out.append(Token('INDENT', '', tok.line))
                    else:
                        while width < stack[-1]:
                            stack.pop()
                            out.append(Token('DEDENT', '', tok.line))
                        if width != stack[-1]:
                            raise ParseError(
                                "Indentation does not match any enclosing block", tok.line
                            )
            if tok.type in _OPENERS:
                depth += 1
            elif tok.type in _CLOSERS:
                depth = max(0, depth - 1)
            out.append(tok)
            prev = tok
        last_line = prev.line if prev is not None else 1
        if prev is not None:
            out.append(Token('NEWLINE', '', last_line))
        while len(stack) > 1:
            stack.pop()
            out.append(Token('DEDENT', '', last_line))
        out.append(Token('EOF', '', last_line))
        return out
