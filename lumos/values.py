"""Runtime values shared by the interpreter, graphs and sets: null, equality, ordered sets."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Optional


class _Null:
    """Result of getEdge when the two nodes are not adjacent.

    `.L` and `[i]` on null give null, comparisons with null are false, and
    any other use of null is ⊥.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_Null, ())

    def __repr__(self) -> str:
        return "null"


NULL = _Null()


def sequence_items(value: Any) -> Optional[List[Any]]:
    """The elements of an ordered set or list, else None."""
    if isinstance(value, OrderedSet):
        return value.to_list()
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def lumos_equal(left: Any, right: Any) -> bool:
    """`=`: booleans are never equal to numbers; ordered sets and lists compare
    element by element, in order; nodes are equal only to themselves; edges and
    graphs compare by content; anything compared with null is false."""
    if left is NULL or right is NULL:
        return False
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    left_items, right_items = sequence_items(left), sequence_items(right)
    if left_items is not None or right_items is not None:
        if left_items is None or right_items is None or len(left_items) != len(right_items):
            return False
        return all(lumos_equal(a, b) for a, b in zip(left_items, right_items))
    return bool(left == right)


def _key(value: Any) -> Any:
    """Hash key that keeps true/false distinct from 1/0 (Python conflates them)."""
    if isinstance(value, bool):
        return ("bool", value)
    hash(value)  # TypeError for unhashable values (lists, ordered sets)
    return value


class OrderedSet:
    """An insertion-ordered set whose membership is Lumos equality (`=`).

    Hashable elements are found through a dict (O(1)); lists and nested
    ordered sets, which are unhashable, are compared one by one.
    """

    def __init__(self, values: Iterable[Any] = ()):
        self._items: list[Any] = []
        self._index: dict[Any, None] = {}
        self._unhashable: list[Any] = []
        for value in values:
            self.add(value)

    def add(self, value: Any):
        try:
            key = _key(value)
        except TypeError:
            if not any(lumos_equal(value, other) for other in self._unhashable):
                self._unhashable.append(value)
                self._items.append(value)
            return
        if key not in self._index:
            self._index[key] = None
            self._items.append(value)

    def extend(self, values: Iterable[Any]):
        for value in values:
            self.add(value)

    def to_list(self) -> list[Any]:
        return list(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, item: Any) -> bool:
        try:
            return _key(item) in self._index
        except TypeError:
            return any(lumos_equal(item, other) for other in self._unhashable)

    def __getitem__(self, index: int) -> Any:
        return self._items[index]

    def __add__(self, other: "OrderedSet") -> "OrderedSet":
        """Union: this set's order first, then new elements of `other`."""
        if not isinstance(other, OrderedSet):
            return NotImplemented
        merged = OrderedSet(self._items)
        merged.extend(other._items)
        return merged

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, OrderedSet) and lumos_equal(self, other)

    __hash__ = None  # mutable during construction; never used as a key

    def __deepcopy__(self, memo):
        # Never mutated once built (every operation makes a new set), so trials share it.
        return self

    def __repr__(self) -> str:
        return "{" + ", ".join(repr(item) for item in self._items) + "}"
