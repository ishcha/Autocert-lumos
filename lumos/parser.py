"""Recursive-descent parser for Lumos.

Statements (paper section 4 and the nl2lumos prompt), laid out with
indentation as in the example programs:

  line       := statement (';' statement)* [';'] NEWLINE
  statement  := IDENT ':=' rhs
              | IDENT ':=' 'Graph' '(' IDENT ',' IDENT ')' ':=' expr    names the node and edge sets
              | 'if' expr 'then' suite ['else' suite]
              | 'while' expr 'do' suite
              | 'while' IDENT 'in' expr suite                  bounded loop
              | 'estimateProb' NUMBER NUMBER STRING ':' body 'toEstimate' '(' expr ')'
  suite      := NEWLINE INDENT line+ DEDENT  |  statement       (inline)
  rhs        := 'Node' '(' attrs ')'  |  'Edge' '(' expr ',' expr [',' attrs] ')'  |  expr
  attrs      := 'L' '{' [STRING (',' STRING)*] '}'

`toEstimate(obs)` may be the last line inside the estimateProb block or the
first line after it. estimateProb may only appear at the top level.

Expressions, loosest binding first:

  or  <  and  <  not  <  comparison (= < > <= >= in, non-associative)
      <  + -  <  * /  <  unary -  <  postfix: call (...), index [...], member .x

The paper's grammar has only '<' and '+'/'*'; '>', '<=', '>=', '-' and '/'
are accepted as harmless extensions.
"""

from typing import List, Tuple

from .ast import *
from .errors import ParseError
from .lexer import Token

_COMPARISON_OPS = ('EQ', 'LT', 'GT', 'LE', 'GE', 'IN')
_TOKEN_NAMES = {
    'NEWLINE': 'end of line',
    'INDENT': 'indentation',
    'DEDENT': 'end of block',
    'EOF': 'end of file',
}


class Parser:
    """Parses a token stream from `Lexer.tokenize()` into a Program."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> Program:
        statements: List[Statement] = []
        while not self._check('EOF'):
            if self._check('DEDENT'):  # cannot happen at top level; be defensive
                raise ParseError("Unexpected end of block", self._peek().line)
            statements.extend(self._parse_line(top_level=True))
        return Program(statements)

    # --- Statements ---------------------------------------------------------

    def _parse_line(self, top_level: bool, in_estimate: bool = False) -> List[Statement]:
        """Parse `s1; s2; ...` up to the end of the logical line.

        Inside an estimateProb body, stops in front of a `toEstimate` that
        follows a ';' so the caller can parse it.
        """
        statements = [self._parse_statement(top_level)]
        while not self._ended_with_block() and self._match('SEMICOLON'):
            if self._check('NEWLINE'):
                break
            if in_estimate and self._check('TOESTIMATE'):
                return statements
            statements.append(self._parse_statement(top_level))
        if not self._ended_with_block():
            self._end_of_line()
        return statements

    def _parse_statement(self, top_level: bool) -> Statement:
        tok = self._peek()
        if tok.type == 'IDENTIFIER' and self._peek(1).type == 'DEFINE':
            return self._parse_assignment()
        if tok.type == 'IF':
            return self._parse_if()
        if tok.type == 'WHILE':
            return self._parse_while()
        if tok.type == 'ESTIMATEPROB':
            if not top_level:
                raise ParseError("estimateProb can only appear at the top level of a program", tok.line)
            return self._parse_estimate_prob()
        if tok.type == 'TOESTIMATE':
            raise ParseError("toEstimate(...) can only close an estimateProb block", tok.line)
        if tok.type == 'IDENTIFIER' and self._peek(1).type == 'EQ':
            raise ParseError(f"Use ':=' to assign to '{tok.value}' ('=' is equality)", tok.line)
        if tok.type == 'INDENT':
            raise ParseError("Unexpected indentation", tok.line)
        raise ParseError(
            f"Expected a statement (x := e, if, while or estimateProb), got {self._describe(tok)}",
            tok.line,
        )

    def _parse_assignment(self) -> Statement:
        name_tok = self._advance()
        self._advance()  # ':='
        if self._check('NODE'):
            value = self._parse_node_constructor(name_tok.value)
        elif self._check('EDGE'):
            value = self._parse_edge_constructor()
        else:
            value = self._parse_expression()
            if self._check('DEFINE') and isinstance(value, GraphConstructor):
                return self._parse_graph_binding(name_tok, value)
        return AssignStatement(name_tok.value, value, line=name_tok.line)

    def _parse_graph_binding(self, name_tok: Token, pattern: GraphConstructor) -> GraphBindStatement:
        """g := Graph(N, E) := e  (paper section 4): N and E name g's node and edge sets."""
        line = self._advance().line  # the second ':='
        names = [name_tok.value]
        for part in (pattern.nodes, pattern.edges):
            if not isinstance(part, Identifier):
                raise ParseError(
                    "In g := Graph(N, E) := e, N and E must be variable names; "
                    "they are bound to the graph's node set and edge set",
                    line,
                )
            names.append(part.name)
        if len(set(names)) != 3:
            raise ParseError(f"g := Graph(N, E) := e needs three different names, got {', '.join(names)}", line)
        value = self._parse_expression()
        return GraphBindStatement(names[0], names[1], names[2], value, line=name_tok.line)

    def _parse_node_constructor(self, name: str) -> NodeConstructor:
        self._advance()  # 'Node'
        self._consume('LPAREN', "Expected '(' after 'Node'")
        attributes = self._parse_attribute_list('Node')
        self._consume('RPAREN', "Expected ')' to close Node(...)")
        return NodeConstructor(name, attributes)

    def _parse_edge_constructor(self) -> EdgeConstructor:
        self._advance()  # 'Edge'
        self._consume('LPAREN', "Expected '(' after 'Edge'")
        src = self._parse_expression()
        self._consume('COMMA', "Edge needs two endpoints, e.g. Edge(v1, v2)")
        dst = self._parse_expression()
        attributes: List[str] = []
        if self._match('COMMA'):
            attributes = self._parse_attribute_list('Edge')
        self._consume('RPAREN', "Expected ')' to close Edge(...)")
        return EdgeConstructor(src, dst, attributes)

    def _parse_attribute_list(self, owner: str) -> List[str]:
        tok = self._peek()
        if not (tok.type == 'IDENTIFIER' and tok.value == 'L' and self._peek(1).type == 'LBRACE'):
            raise ParseError(f'{owner} attributes are written L{{"a", "b", ...}}', tok.line)
        self._advance()
        self._advance()
        attributes: List[str] = []
        if not self._check('RBRACE'):
            attributes.append(self._consume('STRING', "Attributes must be string literals").value)
            while self._match('COMMA'):
                attributes.append(self._consume('STRING', "Attributes must be string literals").value)
        self._consume('RBRACE', "Expected '}' to close the attribute list")
        return attributes

    def _parse_if(self) -> IfStatement:
        tok = self._advance()
        condition = self._parse_expression()
        self._consume('THEN', "Expected 'then' after the if condition")
        then_block = self._parse_suite()
        else_block = None
        if self._match('ELSE'):
            else_block = self._parse_suite()
        elif (
            self._check('NEWLINE')
            and self._peek(1).type == 'ELSE'
            and not self._ended_with_block()
        ):
            # Inline `then` branch with `else` on the following line.
            self._advance()
            self._advance()
            else_block = self._parse_suite()
        return IfStatement(condition, then_block, else_block, line=tok.line)

    def _parse_while(self) -> Statement:
        tok = self._advance()
        condition = self._parse_expression()
        if self._match('DO'):
            return WhileStatement(condition, self._parse_suite(), line=tok.line)
        if (
            isinstance(condition, BinaryOperation)
            and condition.operator == 'IN'
            and isinstance(condition.left, Identifier)
        ):
            body = self._parse_suite()
            return ForEachStatement(condition.left.name, condition.right, body, line=tok.line)
        raise ParseError(
            "Expected 'do' after the while condition "
            "(a bounded loop is written: while i in {0, ..., q} ...)",
            self._peek().line,
        )

    def _parse_suite(self) -> List[Statement]:
        """An indented block, or a single statement on the same line."""
        if self._match('NEWLINE'):
            self._consume('INDENT', "Expected an indented block on the next line")
            statements: List[Statement] = []
            while not self._check('DEDENT'):
                statements.extend(self._parse_line(top_level=False))
            self._advance()  # DEDENT
            return statements
        return [self._parse_statement(top_level=False)]

    def _parse_estimate_prob(self) -> EstimateProbStatement:
        tok = self._advance()
        example = 'e.g. estimateProb 0.05 1000 "Clopper-Pearson" :'
        delta_tok = self._consume('NUMBER', f"Expected delta after estimateProb, {example}")
        trials_tok = self._consume('NUMBER', f"Expected the number of trials after delta, {example}")
        if '.' in trials_tok.value:
            raise ParseError(f"Number of trials must be an integer, got {trials_tok.value}", trials_tok.line)
        method = self._consume('STRING', f"Expected the certifier name as a string, {example}").value
        self._consume('COLON', "Expected ':' after the certifier name")
        body, obs, obs_line = self._parse_estimate_body()
        return EstimateProbStatement(
            float(delta_tok.value), int(trials_tok.value), method, body, obs,
            line=tok.line, to_estimate_line=obs_line,
        )

    def _parse_estimate_body(self) -> Tuple[List[Statement], Expression, int]:
        body: List[Statement] = []
        if not self._match('NEWLINE'):
            # Inline body on the header line: `... : s1; s2; toEstimate(obs)`
            if not self._check('TOESTIMATE'):
                body.extend(self._parse_line(top_level=False, in_estimate=True))
            return (body, *self._expect_to_estimate())
        self._consume('INDENT', "Expected an indented block after 'estimateProb ... :'")
        while not self._check('DEDENT'):
            if self._check('TOESTIMATE'):
                obs_line = self._peek().line
                obs = self._parse_to_estimate()
                self._end_of_line()
                if not self._check('DEDENT'):
                    raise ParseError(
                        "toEstimate(...) must be the last line of the estimateProb block",
                        self._peek().line,
                    )
                self._advance()  # DEDENT
                return body, obs, obs_line
            body.extend(self._parse_line(top_level=False, in_estimate=True))
        self._advance()  # DEDENT
        return (body, *self._expect_to_estimate())

    def _expect_to_estimate(self) -> Tuple[Expression, int]:
        tok = self._peek()
        if tok.type != 'TOESTIMATE':
            raise ParseError(
                f"Expected toEstimate(...) to close the estimateProb block, got {self._describe(tok)}",
                tok.line,
            )
        return self._parse_to_estimate(), tok.line

    def _parse_to_estimate(self) -> Expression:
        self._advance()  # 'toEstimate'
        self._consume('LPAREN', "Expected '(' after toEstimate")
        obs = self._parse_expression()
        self._consume('RPAREN', "Expected ')' to close toEstimate(...)")
        return obs

    # --- Expressions --------------------------------------------------------

    def _parse_expression(self) -> Expression:
        return self._parse_or()

    def _parse_or(self) -> Expression:
        left = self._parse_and()
        while self._match('OR'):
            left = BinaryOperation(left, 'OR', self._parse_and())
        return left

    def _parse_and(self) -> Expression:
        left = self._parse_not()
        while self._match('AND'):
            left = BinaryOperation(left, 'AND', self._parse_not())
        return left

    def _parse_not(self) -> Expression:
        if self._match('NOT'):
            return UnaryOperation('NOT', self._parse_not())
        return self._parse_comparison()

    def _parse_comparison(self) -> Expression:
        left = self._parse_sum()
        if self._peek().type not in _COMPARISON_OPS:
            return left
        operator = self._advance().type
        right = self._parse_sum()
        if self._peek().type in _COMPARISON_OPS:
            raise ParseError("Comparisons cannot be chained; combine them with 'and'", self._peek().line)
        return BinaryOperation(left, operator, right)

    def _parse_sum(self) -> Expression:
        left = self._parse_product()
        while self._peek().type in ('PLUS', 'MINUS'):
            operator = self._advance().type
            left = BinaryOperation(left, operator, self._parse_product())
        return left

    def _parse_product(self) -> Expression:
        left = self._parse_unary()
        while self._peek().type in ('MULTIPLY', 'DIVIDE'):
            operator = self._advance().type
            left = BinaryOperation(left, operator, self._parse_unary())
        return left

    def _parse_unary(self) -> Expression:
        if self._match('MINUS'):
            return UnaryOperation('MINUS', self._parse_unary())
        return self._parse_postfix()

    def _parse_postfix(self) -> Expression:
        expr = self._parse_primary()
        while True:
            if self._match('LPAREN'):
                expr = CallExpression(expr, self._parse_arguments())
            elif self._match('LBRACKET'):
                index = self._parse_expression()
                self._consume('RBRACKET', "Expected ']' after the index")
                expr = IndexExpression(expr, index)
            elif self._match('DOT'):
                member = self._consume('IDENTIFIER', "Expected a member name after '.', e.g. v.L or G.N")
                expr = MemberExpression(expr, member.value)
            else:
                return expr

    def _parse_arguments(self) -> List[Expression]:
        """Arguments after an already-consumed '('."""
        arguments: List[Expression] = []
        if not self._check('RPAREN'):
            arguments.append(self._parse_expression())
            while self._match('COMMA'):
                arguments.append(self._parse_expression())
        self._consume('RPAREN', "Expected ',' or ')' in the argument list")
        return arguments

    def _parse_primary(self) -> Expression:
        tok = self._peek()
        kind = tok.type
        if kind == 'NUMBER':
            self._advance()
            return NumberLiteral(float(tok.value) if '.' in tok.value else int(tok.value))
        if kind == 'STRING':
            self._advance()
            return StringLiteral(tok.value)
        if kind in ('TRUE', 'FALSE'):
            self._advance()
            return BoolLiteral(kind == 'TRUE')
        if kind == 'IDENTIFIER':
            if tok.value == 'L' and self._peek(1).type == 'LBRACE':
                raise ParseError("L{...} attribute lists can only appear inside Node(...) or Edge(...)", tok.line)
            self._advance()
            return Identifier(tok.value)
        if kind == 'LPAREN':
            self._advance()
            expr = self._parse_expression()
            self._consume('RPAREN', "Expected ')'")
            return expr
        if kind == 'LBRACE':
            return self._parse_set()
        if kind == 'SAMPLE':
            return self._parse_sample()
        if kind == 'FORMAT':
            template, graph = self._keyword_arguments('format', 2, 2)
            return FormatExpression(template, graph)
        if kind == 'NB':
            args = self._keyword_arguments('NB', 1, 2)
            return NBExpression(args[0], args[1] if len(args) == 2 else None)
        if kind == 'GETEDGE':
            src, dst, graph = self._keyword_arguments('getEdge', 3, 3)
            return GetEdgeExpression(src, dst, graph)
        if kind == 'GRAPH':
            nodes, edges = self._keyword_arguments('Graph', 2, 2)
            return GraphConstructor(nodes, edges)
        if kind == 'TOOLCALL':
            tool, prompt = self._keyword_arguments('tool-call', 2, 2)
            return ToolCallExpression(tool, prompt)
        if kind in ('NODE', 'EDGE'):
            raise ParseError(
                f"{tok.value}(...) must be declared as its own variable, e.g. v := {tok.value}(...)",
                tok.line,
            )
        raise ParseError(f"Unexpected {self._describe(tok)}", tok.line)

    def _keyword_arguments(self, name: str, min_args: int, max_args: int) -> List[Expression]:
        tok = self._advance()
        self._consume('LPAREN', f"Expected '(' after '{name}'")
        args = self._parse_arguments()
        if not min_args <= len(args) <= max_args:
            expected = str(min_args) if min_args == max_args else f"{min_args} or {max_args}"
            raise ParseError(f"{name}(...) takes {expected} arguments, got {len(args)}", tok.line)
        return args

    def _parse_sample(self) -> SampleExpression:
        tok = self._advance()
        self._consume('LPAREN', "Expected '(' after 'sample'")
        measure = self._peek()
        if measure.type != 'IDENTIFIER' or self._peek(1).type != 'LPAREN':
            raise ParseError("sample(...) takes a measure applied to a set, e.g. sample(meas(S))", tok.line)
        self._advance()
        self._advance()
        elements = self._parse_expression()
        self._consume('RPAREN', f"Expected ')' to close {measure.value}(...)")
        self._consume('RPAREN', "Expected ')' to close sample(...)")
        return SampleExpression(measure.value, elements)

    def _parse_set(self) -> Expression:
        """{}, {e1, ...}, {a, ..., b} or {e | x in S [and b]} (after the '{')."""
        self._advance()  # '{'
        if self._match('RBRACE'):
            return OrderedSetLiteral([])
        first = self._parse_expression()
        if self._match('PIPE'):
            var = self._consume('IDENTIFIER', "Expected the bound variable after '|', e.g. {v | v in S and ...}")
            self._consume('IN', f"Expected 'in' after '{var.value}' in the set-builder")
            source = self._parse_sum()
            condition = self._parse_expression() if self._match('AND') else None
            self._consume('RBRACE', "Expected 'and' or '}' in the set-builder")
            return SetBuilder(first, var.value, source, condition)
        if self._check('COMMA') and self._peek(1).type == 'ELLIPSIS':
            self._advance()
            self._advance()
            self._consume('COMMA', "A range is written {start, ..., end}")
            end = self._parse_expression()
            self._consume('RBRACE', "A range is written {start, ..., end}")
            return RangeExpression(first, end)
        elements = [first]
        while self._match('COMMA'):
            if self._check('ELLIPSIS'):
                raise ParseError("A range is written {start, ..., end}", self._peek().line)
            elements.append(self._parse_expression())
        self._consume('RBRACE', "Expected ',' or '}' in the set literal")
        return OrderedSetLiteral(elements)

    # --- Token helpers ------------------------------------------------------

    def _ended_with_block(self) -> bool:
        """True right after an indented block was closed."""
        return self.pos > 0 and self.tokens[self.pos - 1].type == 'DEDENT'

    def _end_of_line(self):
        if self._match('NEWLINE'):
            return
        tok = self._peek()
        raise ParseError(f"Expected end of line or ';', got {self._describe(tok)}", tok.line)

    def _peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.pos + offset, len(self.tokens) - 1)]

    def _check(self, token_type: str) -> bool:
        return self._peek().type == token_type

    def _advance(self) -> Token:
        tok = self._peek()
        if tok.type != 'EOF':
            self.pos += 1
        return tok

    def _match(self, token_type: str) -> bool:
        if self._check(token_type):
            self._advance()
            return True
        return False

    def _consume(self, token_type: str, message: str) -> Token:
        if self._check(token_type):
            return self._advance()
        tok = self._peek()
        raise ParseError(f"{message}, got {self._describe(tok)}", tok.line)

    @staticmethod
    def _describe(tok: Token) -> str:
        if tok.type in _TOKEN_NAMES:
            return _TOKEN_NAMES[tok.type]
        if tok.type == 'STRING':
            return f'string "{tok.value}"'
        return f"'{tok.value}'"
