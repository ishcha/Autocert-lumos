"""Helpers shared by the Lumos tests (stdlib unittest; pytest also runs them)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lumos.certify import Certifier  # noqa: E402
from lumos.interpreter import Interpreter  # noqa: E402
from lumos.lexer import Lexer  # noqa: E402
from lumos.parser import Parser  # noqa: E402


def token_types(source: str):
    return [t.type for t in Lexer(source).tokenize()]


def parse(source: str):
    return Parser(Lexer(source).tokenize()).parse()


def run(source: str, seed: int = 0, **externals):
    """Run a program; returns (interpreter, certification results)."""
    interpreter = Interpreter(seed=seed)
    for name, value in externals.items():
        interpreter.env.set(name, value)
    results = Certifier().run_program(parse(source), interpreter)
    return interpreter, results


def env_after(source: str, seed: int = 0, **externals):
    """Run a program without estimateProb and return its variable bindings."""
    interpreter, _ = run(source, seed, **externals)
    return interpreter.env.bindings


def value_of(expression: str, setup: str = "", **externals):
    """Evaluate `expression` after running `setup`."""
    return env_after(f"{setup}\n__result := {expression}\n", **externals)["__result"]
