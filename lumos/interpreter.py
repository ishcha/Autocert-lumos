"""Interpreter for Lumos programs.

Values: numbers, strings, booleans, nodes, edges, graphs, ordered sets,
lists (attribute lists v.L), external callables, and `null` (getEdge with no
edge). Undefined operations raise UndefinedValue (the paper's ⊥); inside
estimateProb such a trial is discarded and redrawn by the Certifier.
"""

import operator
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from .ast import *
from .env import Environment
from .errors import RuntimeErrorLumos, UndefinedValue
from .graph import AttributedEdge, AttributedNode, Graph, load_graph_csv
from .measures import choose_weighted, sample_from_measure
from .values import NULL, OrderedSet, lumos_equal, sequence_items


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_NUMERIC_STRING = re.compile(r"\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*")
_SYMBOLS = {
    'EQ': '=', 'LT': '<', 'GT': '>', 'LE': '<=', 'GE': '>=', 'IN': 'in',
    'PLUS': '+', 'MINUS': '-', 'MULTIPLY': '*', 'DIVIDE': '/',
    'AND': 'and', 'OR': 'or', 'NOT': 'not',
}
_COMPARISONS = {'EQ', 'LT', 'GT', 'LE', 'GE', 'IN'}
_ORDERING = {'LT': operator.lt, 'LE': operator.le, 'GT': operator.gt, 'GE': operator.ge}
_ARITHMETIC = {'MINUS': operator.sub, 'MULTIPLY': operator.mul}
# Operators of equal precedence, which form left-nested chains.
_CHAIN_GROUPS = {'PLUS': 'sum', 'MINUS': 'sum', 'MULTIPLY': 'product', 'DIVIDE': 'product',
                 'AND': 'and', 'OR': 'or'}


def type_name(value: Any) -> str:
    if value is NULL:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, OrderedSet):
        return "set"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, AttributedNode):
        return "node"
    if isinstance(value, AttributedEdge):
        return "edge"
    if isinstance(value, Graph):
        return "graph"
    if callable(value):
        return "function"
    return type(value).__name__


def to_lumos_str(value: Any) -> str:
    """Render a scalar the way it is written in Lumos source."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_sequence_items = sequence_items


def from_external(value: Any) -> Any:
    """A value returned by (or bound from) Python code, as a Lumos value.

    Judges built on numpy / torch return numpy scalars, numpy booleans or
    tensors; these become Python numbers and booleans, arrays become lists.
    """
    if value is None or isinstance(value, (bool, int, float, str, OrderedSet, AttributedNode,
                                           AttributedEdge, Graph)):
        return value
    if callable(getattr(value, "lumos_clone_for_trial", None)):
        return value  # it decides its own per-run copies
    if isinstance(value, (list, tuple)):
        return [from_external(v) for v in value]
    item, shape = getattr(value, "item", None), getattr(value, "shape", None)
    if callable(item) and shape == ():  # numpy scalar, 0-d array, torch scalar tensor
        return from_external(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):  # numpy / torch arrays
        return from_external(tolist())
    return value


class Interpreter:
    """Evaluates statements and expressions; estimateProb is run by the Certifier."""

    max_loop_iterations = 1_000_000

    def __init__(self, seed: int = 0, base_dir: Union[str, Path, None] = None):
        self.env = Environment()
        self.rng = random.Random(seed)
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        # Set by the Certifier while a trial body runs.
        self.in_trial = False
        # Graphs bound at the top level; NB(v) without a graph uses the only one.
        self.program_graphs: Dict[str, Graph] = {}
        # Set-builders whose value is the same in every trial of the running
        # estimateProb (see certify.trial_invariant_set_builders), and their values.
        self.invariant_set_builders: frozenset = frozenset()
        self.invariant_values: Dict[int, Any] = {}
        self.env.define("concat", self._builtin_concat)
        self.env.define("len", self._builtin_len)
        self.env.define("getGraph", self._builtin_get_graph)
        self._handlers: Dict[type, Callable[[Any], Any]] = {
            NumberLiteral: lambda e: e.value,
            StringLiteral: lambda e: e.value,
            BoolLiteral: lambda e: e.value,
            Identifier: lambda e: self.env.get(e.name),
            NodeConstructor: lambda e: AttributedNode(e.name, e.attributes),
            EdgeConstructor: self._eval_edge,
            GraphConstructor: self._eval_graph,
            OrderedSetLiteral: self._eval_set_literal,
            RangeExpression: self._eval_range,
            SetBuilder: self._eval_set_builder,
            SampleExpression: self._eval_sample,
            FormatExpression: self._eval_format,
            NBExpression: self._eval_nb,
            GetEdgeExpression: self._eval_get_edge,
            ToolCallExpression: self._eval_tool_call,
            MemberExpression: self._eval_member,
            IndexExpression: self._eval_index,
            CallExpression: self._eval_call,
            BinaryOperation: self._eval_binary,
            UnaryOperation: self._eval_unary,
        }

    # --- Statements ---------------------------------------------------------

    def eval_program(self, program: Program) -> Any:
        """Run a program without estimateProb statements (see Certifier.run_program)."""
        result = None
        for stmt in program.statements:
            result = self.eval_statement(stmt)
        return result

    def eval_statement(self, stmt: Statement) -> Any:
        try:
            return self._execute(stmt)
        except RuntimeErrorLumos as exc:
            if exc.line is None and stmt.line:
                exc.line = stmt.line
            raise

    def _execute(self, stmt: Statement) -> Any:
        if isinstance(stmt, AssignStatement):
            value = self.eval_expression(stmt.value)
            self._assign(stmt.name, value)
            return value
        if isinstance(stmt, GraphBindStatement):
            value = self._not_null(self.eval_expression(stmt.value), "Graph(N, E) := e")
            graph = Graph.lift(value)
            if graph is None:
                raise RuntimeErrorLumos(
                    f"{stmt.name} := Graph({stmt.nodes_name}, {stmt.edges_name}) := e needs e to be a graph "
                    f"(or a node or edge, which lifts to one), got a {type_name(value)}"
                )
            self._assign(stmt.name, graph)
            self._assign(stmt.nodes_name, graph.node_set())
            self._assign(stmt.edges_name, graph.edge_set())
            return graph
        if isinstance(stmt, IfStatement):
            if self._condition(stmt.condition, "if"):
                return self._run_block(stmt.then_block)
            if stmt.else_block is not None:
                return self._run_block(stmt.else_block)
            return None
        if isinstance(stmt, WhileStatement):
            result, iterations = None, 0
            while self._condition(stmt.condition, "while"):
                iterations += 1
                if iterations > self.max_loop_iterations:
                    raise RuntimeErrorLumos(
                        f"while loop exceeded {self.max_loop_iterations} iterations"
                    )
                result = self._run_block(stmt.body)
            return result
        if isinstance(stmt, ForEachStatement):
            result = None
            for element in self._elements(self.eval_expression(stmt.iterable), "a bounded while loop"):
                self.env.set(stmt.variable, element)
                result = self._run_block(stmt.body)
            return result
        if isinstance(stmt, EstimateProbStatement):
            raise RuntimeErrorLumos("estimateProb must be run by the Certifier")
        raise RuntimeErrorLumos(f"Unknown statement type {type(stmt).__name__}")

    def _assign(self, name: str, value: Any) -> None:
        self.env.set(name, value)
        if not self.in_trial:  # graphs bound at the top level are the program graphs NB(v) refers to
            if isinstance(value, Graph):
                self.program_graphs[name] = value
            else:
                self.program_graphs.pop(name, None)

    def _run_block(self, statements: List[Statement]) -> Any:
        result = None
        for stmt in statements:
            result = self.eval_statement(stmt)
        return result

    # --- Expressions --------------------------------------------------------

    def eval_expression(self, expr: Expression) -> Any:
        handler = self._handlers.get(type(expr))
        if handler is None:
            raise RuntimeErrorLumos(f"Unknown expression type {type(expr).__name__}")
        return handler(expr)

    def _eval_edge(self, expr: EdgeConstructor) -> AttributedEdge:
        src = self._node(self.eval_expression(expr.src), "Edge endpoint")
        dst = self._node(self.eval_expression(expr.dst), "Edge endpoint")
        return AttributedEdge(src, dst, expr.attributes)

    def _eval_graph(self, expr: GraphConstructor) -> Graph:
        graph = Graph()
        for node in self._elements(self.eval_expression(expr.nodes), "Graph(nodes, ...)"):
            graph.add_node(self._node(node, "Graph(...) node set element"))
        for edge in self._elements(self.eval_expression(expr.edges), "Graph(..., edges)"):
            if not isinstance(edge, AttributedEdge):
                raise RuntimeErrorLumos(f"Graph(...) edge set contains a {type_name(edge)}, not an edge")
            graph.add_edge(edge)
        return graph

    def _eval_set_literal(self, expr: OrderedSetLiteral) -> OrderedSet:
        return OrderedSet(self._not_null(self.eval_expression(e), "a set element") for e in expr.elements)

    def _eval_range(self, expr: RangeExpression) -> OrderedSet:
        start = self._integer(self.eval_expression(expr.start), "range start")
        end = self._integer(self.eval_expression(expr.end), "range end")
        return OrderedSet(range(start, end + 1))

    def _eval_set_builder(self, expr: SetBuilder) -> OrderedSet:
        key = id(expr)
        if key not in self.invariant_set_builders:
            return self._compute_set_builder(expr)
        value = self.invariant_values.get(key)
        if value is None:
            value = self.invariant_values[key] = self._compute_set_builder(expr)
        return value

    def _compute_set_builder(self, expr: SetBuilder) -> OrderedSet:
        source = self._elements(self.eval_expression(expr.source), "a set-builder")
        result = OrderedSet()
        missing = object()
        bindings = self.env.bindings
        previous = bindings.get(expr.variable, missing)
        try:
            for element in source:
                bindings[expr.variable] = element
                if expr.condition is None or self._condition(expr.condition, "set-builder"):
                    result.add(self._not_null(self.eval_expression(expr.element), "a set element"))
        finally:
            if previous is missing:
                bindings.pop(expr.variable, None)
            else:
                bindings[expr.variable] = previous
        return result

    def _eval_sample(self, expr: SampleExpression) -> Any:
        elements = self._elements(self.eval_expression(expr.elements), "sample(meas(...))")
        return self._sample(elements, expr.measure_name)

    def _sample(self, elements: List[Any], measure_name: str) -> Any:
        """sample(m(S)): `meas` may be supplied by the environment; other
        measures come from the registry (lumos.measures.register_measure)."""
        if not elements:
            raise UndefinedValue(f"sample({measure_name}(S)) with S empty")
        custom = self.env.bindings.get(measure_name) if measure_name == "meas" else None
        if callable(custom):
            weights = self._call(custom, [list(elements)], "measure 'meas'")
            if not isinstance(weights, list):
                raise RuntimeErrorLumos(
                    f"measure 'meas' must return a list of weights, got a {type_name(weights)}"
                )
            return choose_weighted(elements, weights, self.rng, measure_name)
        return sample_from_measure(elements, measure_name, self.rng)

    def _eval_format(self, expr: FormatExpression) -> str:
        template = self._not_null(self.eval_expression(expr.template), "format")
        graph = self._not_null(self.eval_expression(expr.graph), "format")
        if not isinstance(template, str):
            raise RuntimeErrorLumos(f"format(str, g) needs a string template, got a {type_name(template)}")
        if not isinstance(graph, Graph):
            raise RuntimeErrorLumos(
                f"format(str, g) needs a graph, got a {type_name(graph)} (lift nodes with +, e.g. v1 + v2)"
            )

        def replace(match: re.Match) -> str:
            value = self.env.bindings.get(match.group(1))
            if isinstance(value, (AttributedNode, AttributedEdge)) and graph.contains(value):
                return to_lumos_str(self._sample(value.attributes, "meas"))
            return match.group(0)  # not an element of g: left as written

        return _PLACEHOLDER.sub(replace, template)

    def _eval_nb(self, expr: NBExpression) -> OrderedSet:
        node = self._node(self.eval_expression(expr.node), "NB(v)")
        if expr.graph is not None:
            graph = self._not_null(self.eval_expression(expr.graph), "NB(v, G)")
            if not isinstance(graph, Graph):
                raise RuntimeErrorLumos(f"NB(v, G) needs a graph, got a {type_name(graph)}")
        else:
            graph = self._program_graph()
        return OrderedSet(graph.neighbors(node))

    def _program_graph(self) -> Graph:
        graphs = list({id(g): g for g in self.program_graphs.values()}.values())
        if len(graphs) == 1:
            return graphs[0]
        if not self.program_graphs:
            raise RuntimeErrorLumos(
                "NB(v) needs a graph declared before estimateProb (G := Graph(...) or getGraph(...))"
            )
        names = ", ".join(sorted(self.program_graphs))
        raise RuntimeErrorLumos(
            f"NB(v) is ambiguous: the program declares several graphs ({names}); write NB(v, G)"
        )

    def _eval_get_edge(self, expr: GetEdgeExpression) -> Any:
        src = self._node(self.eval_expression(expr.src), "getEdge")
        dst = self._node(self.eval_expression(expr.dst), "getEdge")
        graph = self._not_null(self.eval_expression(expr.graph), "getEdge")
        if not isinstance(graph, Graph):
            raise RuntimeErrorLumos(f"getEdge(v1, v2, G) needs a graph, got a {type_name(graph)}")
        edge = graph.get_edge(src, dst)
        return NULL if edge is None else edge

    def _eval_tool_call(self, expr: ToolCallExpression) -> Any:
        tool = self._not_null(self.eval_expression(expr.tool), "tool-call")
        prompt = self._not_null(self.eval_expression(expr.prompt), "tool-call")
        name = expr.tool.name if isinstance(expr.tool, Identifier) else "tool"
        return self._call(tool, [prompt], f"tool-call({name}, ...)")

    def _eval_member(self, expr: MemberExpression) -> Any:
        base = self.eval_expression(expr.base)
        member = expr.member
        if base is NULL:
            if member == 'L':
                return NULL
            raise UndefinedValue(f"null (getEdge with no edge) has no member '.{member}'")
        if member == 'L' and isinstance(base, (AttributedNode, AttributedEdge)):
            return list(base.attributes)
        if member == 'N' and isinstance(base, Graph):
            return base.node_set()
        if member == 'E' and isinstance(base, Graph):
            return base.edge_set()
        raise RuntimeErrorLumos(
            f"A {type_name(base)} has no member '.{member}' (nodes and edges have .L; graphs have .N and .E)"
        )

    def _eval_index(self, expr: IndexExpression) -> Any:
        base = self.eval_expression(expr.base)
        index = self._integer(self.eval_expression(expr.index), "an index")
        if base is NULL:
            return NULL
        if not isinstance(base, (OrderedSet, list, tuple, str)):
            raise RuntimeErrorLumos(f"Only sets, lists and strings can be indexed, not a {type_name(base)}")
        try:
            return base[index]
        except IndexError:
            raise UndefinedValue(
                f"index {index} is out of range for a {type_name(base)} of length {len(base)}"
            ) from None

    def _eval_call(self, expr: CallExpression) -> Any:
        name = expr.callee.name if isinstance(expr.callee, Identifier) else "call"
        callee = self._not_null(self.eval_expression(expr.callee), f"{name}(...)")
        args = [self._not_null(self.eval_expression(a), f"{name}(...)") for a in expr.arguments]
        return self._call(callee, args, f"{name}(...)")

    def _call(self, fn: Any, args: List[Any], what: str) -> Any:
        if not callable(fn):
            raise RuntimeErrorLumos(f"{what}: a {type_name(fn)} is not a function")
        try:
            result = fn(*args)
        except RuntimeErrorLumos:
            raise
        except Exception as exc:
            raise RuntimeErrorLumos(f"{what} failed: {type(exc).__name__}: {exc}") from exc
        return from_external(result)

    def _eval_binary(self, expr: BinaryOperation) -> Any:
        group = _CHAIN_GROUPS.get(expr.operator)
        # Flatten a left-nested chain of same-precedence operators: x op1 r1 op2 r2 ...
        steps, node = [], expr
        while isinstance(node, BinaryOperation) and _CHAIN_GROUPS.get(node.operator) == group and group:
            steps.append((node.operator, node.right))
            node = node.left
        if not steps:  # comparisons: one operator
            return self._apply_binary(expr.operator, self.eval_expression(expr.left), expr.right)
        steps.reverse()
        value = self.eval_expression(node)
        for op, right in steps:
            if op in ('AND', 'OR'):
                # Short-circuit: the right operand runs only when it decides the result.
                left = self._boolean(value, _SYMBOLS[op])
                if (op == 'AND' and not left) or (op == 'OR' and left):
                    return left
                value = self._boolean(self.eval_expression(right), _SYMBOLS[op])
            else:
                value = self._apply_binary(op, value, right)
        return value

    def _apply_binary(self, op: str, left: Any, right_expr: Expression) -> Any:
        symbol = _SYMBOLS.get(op, op)
        right = self.eval_expression(right_expr)
        if left is NULL or right is NULL:
            if op in _COMPARISONS:
                return False
            raise UndefinedValue(f"null (getEdge with no edge) cannot be used with '{symbol}'")
        if op == 'EQ':
            return lumos_equal(left, right)
        if op == 'IN':
            return self._member_of(left, right)
        if op in _ORDERING:
            return _ORDERING[op](self._real(left, symbol), self._real(right, symbol))
        if op == 'PLUS':
            return self._plus(left, right)
        if op in _ARITHMETIC:
            return _ARITHMETIC[op](self._number(left, symbol), self._number(right, symbol))
        if op == 'DIVIDE':
            numerator, denominator = self._number(left, symbol), self._number(right, symbol)
            if denominator == 0:
                raise UndefinedValue("division by zero")
            return numerator / denominator
        raise RuntimeErrorLumos(f"Unknown operator {op}")

    def _eval_unary(self, expr: UnaryOperation) -> Any:
        value = self.eval_expression(expr.operand)
        if expr.operator == 'NOT':
            return not self._boolean(value, "not")
        if expr.operator == 'MINUS':
            return -self._number(value, "-")
        raise RuntimeErrorLumos(f"Unknown unary operator {expr.operator}")

    def _plus(self, left: Any, right: Any) -> Any:
        """`+` is set union, graph union, string concatenation or addition."""
        left_items, right_items = _sequence_items(left), _sequence_items(right)
        if left_items is not None and right_items is not None:
            return OrderedSet(left_items) + OrderedSet(right_items)
        left_graph, right_graph = Graph.lift(left), Graph.lift(right)
        if left_graph is not None and right_graph is not None:
            return left_graph.union(right_graph)
        if isinstance(left, str) or isinstance(right, str):
            other = right if isinstance(left, str) else left
            if isinstance(other, (str, int, float)):
                return to_lumos_str(left) + to_lumos_str(right)
        elif _is_number(left) and _is_number(right):
            return left + right
        raise RuntimeErrorLumos(f"Cannot add a {type_name(left)} and a {type_name(right)}")

    def _member_of(self, value: Any, collection: Any) -> bool:
        if isinstance(collection, OrderedSet):
            return value in collection
        if isinstance(collection, (list, tuple)):
            return any(lumos_equal(value, item) for item in collection)
        if isinstance(collection, Graph):
            raise RuntimeErrorLumos("'in' needs a set; use G.N or G.E for the nodes or edges of a graph")
        raise RuntimeErrorLumos(f"'in' needs a set or list on the right, got a {type_name(collection)}")

    # --- Value checks -------------------------------------------------------

    def _condition(self, expr: Expression, where: str) -> bool:
        return self._boolean(self.eval_expression(expr), where)

    @staticmethod
    def _not_null(value: Any, where: str) -> Any:
        if value is NULL:
            raise UndefinedValue(f"null (getEdge with no edge) cannot be used in {where}")
        return value

    def _boolean(self, value: Any, where: str) -> bool:
        self._not_null(value, f"'{where}'")
        if not isinstance(value, bool):
            raise RuntimeErrorLumos(f"'{where}' needs a boolean, got a {type_name(value)}")
        return value

    def _number(self, value: Any, where: str) -> Union[int, float]:
        self._not_null(value, f"'{where}'")
        if not _is_number(value):
            raise RuntimeErrorLumos(f"'{where}' needs numbers, got a {type_name(value)}")
        return value

    def _real(self, value: Any, where: str) -> Union[int, float]:
        """Operand of < > <= >=: a number, or a string holding a number (e.g. an attribute "0.91")."""
        if _is_number(value):
            return value
        if isinstance(value, str) and _NUMERIC_STRING.fullmatch(value):
            return float(value)
        raise RuntimeErrorLumos(f"'{where}' compares numbers, got a {type_name(value)} {value!r}")

    def _integer(self, value: Any, what: str) -> int:
        self._not_null(value, what)
        if not _is_number(value) or int(value) != value:
            raise RuntimeErrorLumos(f"{what} must be an integer, got {value!r}")
        return int(value)

    def _node(self, value: Any, where: str) -> AttributedNode:
        self._not_null(value, where)
        if not isinstance(value, AttributedNode):
            raise RuntimeErrorLumos(f"{where} needs a node, got a {type_name(value)}")
        return value

    def _elements(self, value: Any, where: str) -> List[Any]:
        self._not_null(value, where)
        items = _sequence_items(value)
        if items is not None:
            return items
        if isinstance(value, Graph):
            raise RuntimeErrorLumos(f"{where} needs a set; use G.N or G.E for the nodes or edges of a graph")
        raise RuntimeErrorLumos(f"{where} needs a set or list, got a {type_name(value)}")

    # --- Built-in functions -------------------------------------------------

    def _builtin_concat(self, *values: Any) -> str:
        """concat(S): the elements of S joined into one string."""
        if len(values) == 1 and _sequence_items(values[0]) is not None:
            values = tuple(_sequence_items(values[0]))
        for value in values:
            if not isinstance(value, (str, int, float)):
                raise RuntimeErrorLumos(f"concat(...) joins strings, got a {type_name(value)}")
        return "".join(to_lumos_str(v) for v in values)

    def _builtin_len(self, value: Any) -> int:
        if isinstance(value, (OrderedSet, list, tuple, str)):
            return len(value)
        raise RuntimeErrorLumos(f"len(...) needs a set, list or string, got a {type_name(value)}")

    def _builtin_get_graph(self, path: Any) -> Graph:
        if not isinstance(path, str):
            raise RuntimeErrorLumos(f"getGraph(path) needs a string path, got a {type_name(path)}")
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.base_dir / resolved
        return load_graph_csv(resolved)
