"""Inputs for nl2lumos/spec_progs_claude/prime.lumos on the full PrimeKG.

    python scripts/primekg_to_lumos.py data/primekg      # once: writes data/primekg/primekg.csv
    python cli.py nl2lumos/spec_progs_claude/prime.lumos --world worlds/prime_primekg.py \
        --hf-model Qwen/Qwen2.5-0.5B-Instruct --trials 20

getGraph("primekg.csv") resolves against BASE_DIR. J_QA is external in the
spec; the one below checks which option label the reply picks first.
"""

import re

BASE_DIR = "../data/primekg"

_LABEL = re.compile(r"\b([ABCD])\)")


def J_QA(reply, label):
    """1 if the first option label in the reply ("A)".."D)") is `label`."""
    match = _LABEL.search(reply)
    return 1 if match and f"{match.group(1)})" == label else 0


def externals():
    return {"J_QA": J_QA}
