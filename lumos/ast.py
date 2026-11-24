"""AST nodes for Lumos programs."""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass(frozen=True)
class Node:
    """Base class for all AST nodes."""


@dataclass(frozen=True)
class Statement(Node):
    """Base class for statements; `line` is where the statement starts."""
    line: int = field(default=0, kw_only=True, compare=False)


@dataclass(frozen=True)
class Expression(Node):
    """Base class for expressions."""


@dataclass(frozen=True)
class Program(Node):
    statements: List[Statement]


# --- Statements -------------------------------------------------------------

@dataclass(frozen=True)
class AssignStatement(Statement):
    """x := e"""
    name: str
    value: Expression


@dataclass(frozen=True)
class GraphBindStatement(Statement):
    """g := Graph(N, E) := e   (the paper's sugar: also binds N and E to g's node and edge sets)"""
    name: str
    nodes_name: str
    edges_name: str
    value: Expression


@dataclass(frozen=True)
class IfStatement(Statement):
    """if b then s1 [else s2]"""
    condition: Expression
    then_block: List[Statement]
    else_block: Optional[List[Statement]] = None


@dataclass(frozen=True)
class WhileStatement(Statement):
    """while b do s"""
    condition: Expression
    body: List[Statement]


@dataclass(frozen=True)
class ForEachStatement(Statement):
    """Bounded loop `while x in S s` (no `do`): runs s once per element of S."""
    variable: str
    iterable: Expression
    body: List[Statement]


@dataclass(frozen=True)
class EstimateProbStatement(Statement):
    """estimateProb delta n "certifier" : body  toEstimate(obs)"""
    delta: float
    trials: int
    method: str
    body: List[Statement]
    to_estimate: Expression
    to_estimate_line: int = field(default=0, kw_only=True, compare=False)


# --- Expressions ------------------------------------------------------------

@dataclass(frozen=True)
class NumberLiteral(Expression):
    value: Union[int, float]


@dataclass(frozen=True)
class StringLiteral(Expression):
    value: str


@dataclass(frozen=True)
class BoolLiteral(Expression):
    value: bool


@dataclass(frozen=True)
class Identifier(Expression):
    name: str


@dataclass(frozen=True)
class NodeConstructor(Expression):
    """Node(L{...}); only valid as `name := Node(...)`, and named after `name`."""
    name: str
    attributes: List[str]


@dataclass(frozen=True)
class EdgeConstructor(Expression):
    """Edge(v1, v2[, L{...}]) - an undirected edge."""
    src: Expression
    dst: Expression
    attributes: List[str]


@dataclass(frozen=True)
class GraphConstructor(Expression):
    """Graph({nodes}, {edges})"""
    nodes: Expression
    edges: Expression


@dataclass(frozen=True)
class OrderedSetLiteral(Expression):
    """{e1, e2, ...}"""
    elements: List[Expression]


@dataclass(frozen=True)
class RangeExpression(Expression):
    """{start, ..., end} - the integers from start to end inclusive."""
    start: Expression
    end: Expression


@dataclass(frozen=True)
class SetBuilder(Expression):
    """{element | variable in source [and condition]}"""
    element: Expression
    variable: str
    source: Expression
    condition: Optional[Expression] = None


@dataclass(frozen=True)
class SampleExpression(Expression):
    """sample(meas(S))"""
    measure_name: str
    elements: Expression


@dataclass(frozen=True)
class FormatExpression(Expression):
    """format(str, g)"""
    template: Expression
    graph: Expression


@dataclass(frozen=True)
class NBExpression(Expression):
    """NB(v) - neighbours of v in the program graph (or in an explicit graph)."""
    node: Expression
    graph: Optional[Expression] = None


@dataclass(frozen=True)
class GetEdgeExpression(Expression):
    """getEdge(v1, v2, G)"""
    src: Expression
    dst: Expression
    graph: Expression


@dataclass(frozen=True)
class ToolCallExpression(Expression):
    """tool-call(t, p)"""
    tool: Expression
    prompt: Expression


@dataclass(frozen=True)
class MemberExpression(Expression):
    """v.L, e.L, G.N, G.E"""
    base: Expression
    member: str


@dataclass(frozen=True)
class IndexExpression(Expression):
    """S[i]"""
    base: Expression
    index: Expression


@dataclass(frozen=True)
class CallExpression(Expression):
    """f(a1, ..., an) - a call to an external or built-in function."""
    callee: Expression
    arguments: List[Expression]


@dataclass(frozen=True)
class BinaryOperation(Expression):
    left: Expression
    operator: str
    right: Expression


@dataclass(frozen=True)
class UnaryOperation(Expression):
    operator: str
    operand: Expression
