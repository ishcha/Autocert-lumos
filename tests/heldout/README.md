# Held-out specs

Specs in this directory are kept out of interpreter development, so that
passing them shows the interpreter generalizes rather than being fitted to the
specs it was developed against (those are in `tests/specs/`).

Each held-out spec is two files with the same stem:

- `name.lumos`: the program.
- `name.py`: a sidecar with `externals()` and `check(test, results, externals)`,
  plus optional `SEED` and `BASE_DIR` (relative to the sidecar). It follows the same format as
  `tests/specs/*.py`, and the helpers are in `lumos/testing.py`.

`tests/test_specs.py` picks them up automatically:

```bash
python -m unittest discover -s tests -p "test_specs.py" -v
```

Protocol: whoever changes the interpreter should only run these and see
pass/fail, not read them. A spec that has been read while debugging a failure
moves to `tests/specs/`.
