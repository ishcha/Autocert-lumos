"""Runs real Lumos specs end-to-end against mock worlds with known answers.

Each sidecar module in tests/specs/ (specs the interpreter was developed
against) and tests/heldout/ (specs kept out of development; see its README)
defines:

    PROGRAM                     .lumos path relative to the repository root
                                (default: the .lumos file next to the sidecar)
    SEED                        interpreter seed (default 0)
    BASE_DIR                    directory getGraph(...) paths resolve against,
                                relative to the sidecar's own directory
                                (default: the program's directory)
    externals()                 fresh mock externals: L, judges, tools, inputs
    check(test, results, ext)   assertions on the CertificationResults

The runner itself knows nothing about individual specs.
"""

import importlib.util
import unittest
from pathlib import Path

from lumos_test_utils import REPO
from lumos.cli import execute

TESTS = Path(__file__).resolve().parent
SPEC_DIRS = [TESTS / "specs", TESTS / "heldout"]


def load_sidecar(path: Path):
    spec = importlib.util.spec_from_file_location(f"lumos_spec_{path.parent.name}_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_spec(sidecar: Path):
    module = load_sidecar(sidecar)
    program_path = REPO / module.PROGRAM if hasattr(module, "PROGRAM") else sidecar.with_suffix(".lumos")
    base_dir = sidecar.parent / module.BASE_DIR if hasattr(module, "BASE_DIR") else program_path.parent
    externals = module.externals()
    results = execute(program_path, seed=getattr(module, "SEED", 0), externals=externals, data_dir=base_dir)
    return module, results, externals


class SpecTests(unittest.TestCase):
    """One test per sidecar, generated below."""


def _make_test(sidecar: Path):
    def test(self):
        module, results, externals = run_spec(sidecar)
        module.check(self, results, externals)
    test.__doc__ = f"{sidecar.parent.name}/{sidecar.stem}"
    return test


for _directory in SPEC_DIRS:
    for _sidecar in sorted(_directory.glob("*.py")) if _directory.is_dir() else []:
        if not _sidecar.name.startswith("_"):
            setattr(SpecTests, f"test_{_directory.name}_{_sidecar.stem}", _make_test(_sidecar))


if __name__ == "__main__":
    unittest.main()
