"""Error classes for the Lumos interpreter."""


class ParseError(Exception):
    """A syntax error, reported with the line where it was detected."""

    def __init__(self, message: str, line: int = 0):
        self.message = message
        self.line = line
        super().__init__(f"Parse error at line {line}: {message}")


class RuntimeErrorLumos(Exception):
    """An error while evaluating a program.

    `line` is filled in by the interpreter with the line of the innermost
    statement being executed when the code raising the error did not know it.
    """

    def __init__(self, message: str, line: int | None = None):
        self.message = message
        self.line = line
        super().__init__(message)

    def __str__(self) -> str:
        where = f" at line {self.line}" if self.line else ""
        return f"Runtime error{where}: {self.message}"


class UndefinedValue(RuntimeErrorLumos):
    """The paper's ⊥: an operation with no defined result, e.g. sample(∅).

    Inside estimateProb, a trial that produces ⊥ is discarded and redrawn
    (rejection sampling), so the certified quantity is conditional on the
    scenario being defined.
    """
