"""Graphs for Lumos: attributed nodes and undirected attributed edges."""

import csv
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .errors import RuntimeErrorLumos
from .values import OrderedSet

_uids = itertools.count()


class AttributedNode:
    """A node: a display name and an ordered attribute list L.

    Every evaluation of Node(...), and every node read by getGraph, is a
    distinct node: nodes are equal only to themselves, even when their names
    and attributes coincide (e.g. v_e1 := Node(L{""}) and v_e2 := Node(L{""})).
    """

    __slots__ = ('name', 'attributes', 'uid')

    def __init__(self, name: str, attributes: List[str]):
        self.name = name
        self.attributes = list(attributes)
        self.uid = next(_uids)

    # Equality and hashing are identity (the object defaults).

    def __deepcopy__(self, memo):
        return self  # immutable

    def __repr__(self) -> str:
        attrs = ", ".join(f'"{a}"' for a in self.attributes)
        return f"Node({self.name}, L{{{attrs}}})"


def pair_key(a: AttributedNode, b: AttributedNode) -> Tuple[int, int]:
    """Order-independent key for an unordered pair of nodes."""
    return (a.uid, b.uid) if a.uid <= b.uid else (b.uid, a.uid)


class AttributedEdge:
    """An undirected edge between two nodes, with an attribute list L.

    Edge(a, b) and Edge(b, a) with the same attributes are equal.
    """

    __slots__ = ('src_node', 'dst_node', 'attributes')

    def __init__(self, src: AttributedNode, dst: AttributedNode, attributes: List[str]):
        self.src_node = src
        self.dst_node = dst
        self.attributes = list(attributes)

    @property
    def src(self) -> str:
        return self.src_node.name

    @property
    def dst(self) -> str:
        return self.dst_node.name

    @property
    def key(self) -> Tuple[int, int]:
        return pair_key(self.src_node, self.dst_node)

    def __eq__(self, other) -> bool:
        if not isinstance(other, AttributedEdge) or self.attributes != other.attributes:
            return False
        a, b, c, d = self.src_node, self.dst_node, other.src_node, other.dst_node
        return (a is c and b is d) or (a is d and b is c)

    def __hash__(self) -> int:
        return hash((self.key, tuple(self.attributes)))

    def __deepcopy__(self, memo):
        return self  # immutable

    def __repr__(self) -> str:
        attrs = ", ".join(f'"{a}"' for a in self.attributes)
        return f"Edge({self.src}, {self.dst}, L{{{attrs}}})"


class Graph:
    """An undirected graph with at most one edge per pair of nodes.

    Graphs do not change once built. Two graphs are equal when they have the
    same nodes and the same edges.
    """

    def __init__(self):
        self._order: Dict[AttributedNode, int] = {}
        self._adjacency: Dict[AttributedNode, List[AttributedNode]] = {}
        self.edges: List[AttributedEdge] = []
        self._edge_index: Dict[Tuple[int, int], AttributedEdge] = {}
        self._node_set: Optional[OrderedSet] = None
        self._edge_set: Optional[OrderedSet] = None
        self._hash: Optional[int] = None

    @classmethod
    def lift(cls, value) -> Optional["Graph"]:
        """A graph as is; a node as Graph({v}, {}); an edge as Graph({a, b}, {e})."""
        if isinstance(value, Graph):
            return value
        if isinstance(value, AttributedNode):
            graph = cls()
            graph.add_node(value)
            return graph
        if isinstance(value, AttributedEdge):
            graph = cls()
            graph.add_node(value.src_node)
            graph.add_node(value.dst_node)
            graph.add_edge(value)
            return graph
        return None

    @property
    def nodes(self) -> List[AttributedNode]:
        """The nodes, in order."""
        return list(self._order)

    def node_named(self, name: str) -> AttributedNode:
        """The first node with this name (names need not be unique)."""
        for node in self._order:
            if node.name == name:
                return node
        raise KeyError(name)

    def add_node(self, node: AttributedNode) -> AttributedNode:
        if node not in self._order:
            self._order[node] = len(self._order)
            self._adjacency[node] = []
            self._node_set = self._hash = None
        return node

    def add_edge(self, edge: AttributedEdge) -> AttributedEdge:
        for endpoint in (edge.src_node, edge.dst_node):
            if endpoint not in self._order:
                raise RuntimeErrorLumos(
                    f"Edge {edge.src}-{edge.dst}: endpoint '{endpoint.name}' is not a node of the graph"
                )
        existing = self._edge_index.get(edge.key)
        if existing is not None:
            if existing == edge:
                return existing
            raise RuntimeErrorLumos(
                f"Graph already has an edge between '{edge.src}' and '{edge.dst}' "
                "(at most one edge per pair of nodes)"
            )
        self.edges.append(edge)
        self._edge_index[edge.key] = edge
        self._adjacency[edge.src_node].append(edge.dst_node)
        if edge.dst_node is not edge.src_node:
            self._adjacency[edge.dst_node].append(edge.src_node)
        self._edge_set = self._hash = None
        return edge

    def contains(self, value) -> bool:
        if isinstance(value, AttributedNode):
            return value in self._order
        if isinstance(value, AttributedEdge):
            return self._edge_index.get(value.key) == value
        return False

    def node_set(self) -> OrderedSet:
        """G.N, built once per graph."""
        if self._node_set is None:
            self._node_set = OrderedSet(self._order)
        return self._node_set

    def edge_set(self) -> OrderedSet:
        """G.E, built once per graph."""
        if self._edge_set is None:
            self._edge_set = OrderedSet(self.edges)
        return self._edge_set

    def neighbors(self, node: AttributedNode) -> List[AttributedNode]:
        """NB(v) = {u | u in G.N and (u, v) in G.E}, in node order."""
        if node not in self._order:
            return []
        return sorted(self._adjacency[node], key=self._order.__getitem__)

    def get_edge(self, a: AttributedNode, b: AttributedNode) -> Optional[AttributedEdge]:
        if a not in self._order or b not in self._order:
            return None
        return self._edge_index.get(pair_key(a, b))

    def includes(self, other: "Graph") -> bool:
        return all(n in self._order for n in other._order) and all(self.contains(e) for e in other.edges)

    def union(self, other: "Graph") -> "Graph":
        """g1 + g2: the nodes and edges of g1, then the new ones of g2.

        When g1 already includes g2 the result is g1 itself (graphs are
        immutable), e.g. G + v for a node v of G.
        """
        if self.includes(other):
            return self
        result = Graph()
        for graph in (self, other):
            for node in graph._order:
                result.add_node(node)
        for graph in (self, other):
            for edge in graph.edges:
                result.add_edge(edge)
        return result

    def __eq__(self, other) -> bool:
        if not isinstance(other, Graph):
            return False
        if self is other:
            return True
        return (len(self._order) == len(other._order) and len(self.edges) == len(other.edges)
                and self.includes(other))

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((frozenset(self._order), frozenset(self.edges)))
        return self._hash

    def __deepcopy__(self, memo):
        return self  # immutable once built; trials share it

    def __repr__(self) -> str:
        return f"Graph({len(self._order)} nodes, {len(self.edges)} edges)"


def load_graph_csv(path: Path) -> Graph:
    """Read a graph from a CSV edge list, streaming row by row.

    The header must contain `src_id` and `dst_id`. Other columns starting
    with `src_` / `dst_` are that endpoint's attributes and every remaining
    column is an edge attribute; in each case L follows the column order.
    A node that appears in several rows must have the same attributes in
    every row. Rows with an empty `dst_id` declare an isolated node.
    """
    try:
        handle = open(path, newline='', encoding='utf-8-sig')
    except OSError as exc:
        raise RuntimeErrorLumos(f"getGraph: cannot open '{path}': {exc.strerror}") from exc
    old_limit = csv.field_size_limit()
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))  # long context attributes
    try:
        with handle:
            return _read_edge_list(csv.reader(handle), path)
    finally:
        csv.field_size_limit(old_limit)


def _read_edge_list(reader, path: Path) -> Graph:
    header = next(reader, None)
    if not header or 'src_id' not in header or 'dst_id' not in header:
        raise RuntimeErrorLumos(f"getGraph: '{path}' needs a header with src_id and dst_id columns")
    src_id, dst_id = header.index('src_id'), header.index('dst_id')
    src_cols = [i for i, h in enumerate(header) if h.startswith('src_') and i != src_id]
    dst_cols = [i for i, h in enumerate(header) if h.startswith('dst_') and i != dst_id]
    edge_cols = [i for i in range(len(header)) if i not in src_cols + dst_cols + [src_id, dst_id]]

    graph = Graph()
    by_id: Dict[str, AttributedNode] = {}
    strings: Dict[str, str] = {}
    intern = lambda value: strings.setdefault(value, value)

    def fail(message):
        raise RuntimeErrorLumos(f"getGraph: '{path}' line {reader.line_num}: {message}")

    def node_from(row, id_col, cols):
        name = row[id_col]
        if name == '':
            fail(f"empty {header[id_col]}")
        attributes = [intern(row[i]) for i in cols]
        existing = by_id.get(name)
        if existing is None:
            node = by_id[name] = AttributedNode(name, attributes)
            return graph.add_node(node)
        if existing.attributes != attributes:
            fail(f"node '{name}' has different attributes than in an earlier row")
        return existing

    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            fail(f"{len(row)} columns, expected {len(header)}")
        src = node_from(row, src_id, src_cols)
        if row[dst_id] == '':
            continue
        dst = node_from(row, dst_id, dst_cols)
        try:
            graph.add_edge(AttributedEdge(src, dst, [intern(row[i]) for i in edge_cols]))
        except RuntimeErrorLumos as exc:
            fail(exc.message)
    return graph
