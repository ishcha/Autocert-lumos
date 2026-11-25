# Lumos reference

The detailed documentation for this Lumos interpreter. For an overview, installation and the demo, see the [README](../README.md).

- [Running specs](#running-specs): the Python API, the command line, externals and worlds, the specs in this repo, PrimeKG
- [Language reference](#language-reference): syntax, semantics, external functions
- [Differences from the paper](#differences-from-the-paper)
- [Tests](#tests)
- [Generating specs from natural language](#generating-specs-from-natural-language)
- [Repository layout](#repository-layout)

## Running specs

### From Python

```python
from lumos.cli import execute
from lumos.local_model import LocalModel

L = LocalModel("Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=64)   # any HuggingFace chat model

def J_safe(reply):                    # your judge: any Python function returning 1 or 0
    return 1 if "can't" in reply.lower() else 0

(result,) = execute("nl2lumos/spec_progs_claude/cat_risk_random_walk.lumos",
                    trials=20, externals={"L": L, "J_safe": J_safe})
print(result)
```

To run a program held in a string instead of a file, pass it as `execute(source=spec_text, ...)`; `getGraph` paths then resolve against the working directory.

`execute` returns one result per `estimateProb` in the program. Each result has the interval (`lower_ci`, `upper_ci`), the observations, the seeds of every run, and the number of runs discarded as undefined.

`LocalModel` runs the model in-process with transformers. It applies the model's chat template, answers without thinking for models that think first (such as Qwen3 and Qwen3.5) unless `thinking=True`, and takes `max_new_tokens`, `temperature`, `top_p`, `top_k`, `system_message` and `device`.

### Externals and worlds

Everything the program calls but does not define is an **external**. That covers the target `L`, judges, tools, inputs such as `M`, and a custom `meas`. Externals are passed in `externals`, or by a world file, which is a Python file whose `externals()` returns that dict (`--world`; see `worlds/`). An external can be any Python callable, such as a transformers pipeline, an API client or a plain function.

### From the command line

```bash
lumos nl2lumos/spec_progs_claude/cat_risk_random_walk.lumos \
    --hf-model Qwen/Qwen2.5-0.5B-Instruct --with-basic-oracles --trials 20 --trace 1
```

| Option | Meaning |
|---|---|
| `--hf-model ID`, `--device D` | the target `L` is a local HuggingFace model (GPU index, `cpu` or `auto`) |
| `--model ID` / `--server-addr URL` | the target `L` is served by vLLM (started for you, or already running) |
| `--world FILE.py` | bind externals from `externals()` in that file; its `BASE_DIR` sets where `getGraph` looks |
| `--trials N` | override `n` in every `estimateProb` (quick runs) |
| `--trace K` | print every external call of the first `K` runs, and how each run ended |
| `--with-basic-oracles` | keyword-based `J_safe` / `J_comply` (refusal detection), for quick checks |
| `--gen-length`, `--temperature`, `--top-p`, `--system-message` | generation settings for `L` |
| `--seed`, `--data-dir`, `-v` | interpreter seed, directory for `getGraph` paths, verbose |

### The specs in this repo

The paper's specs are in `nl2lumos/spec_progs_claude/`:

| spec | what it certifies | externals | run with |
|---|---|---|---|
| `cat_risk_random_walk` | safety over a random 3-turn harmful conversation | `L`, `J_safe` | `--with-basic-oracles` or your judges |
| `cat_risk_adaptive` | safety when the next question adapts to whether the model complied | `L`, `J_safe`, `J_comply` | `--with-basic-oracles` or your judges |
| `counterfactual_bias` | equal treatment of a male and a female prompt under random jailbreak prefixes | `L`, `J_bias`, `M`, `meas` | `--world worlds/counterfactual_bias.py` (placeholder `M` and `J_bias`) |
| `prime` | medical question answering over the PrimeKG knowledge graph | `L`, `J_QA`, the PrimeKG graph | `--world worlds/prime_primekg.py`, after converting PrimeKG ([below](#primekg-data)) |
| `vlm_braking_rain` | a vision model's braking decision in simulated rain | a vision model `L`, the `Scenic` tool, `J_VLM_QA` | needs a VLM and Scenic (not connected in this version) |
| `beaver` | bounds on leaking private data, from the Beaver verifier | `Beaver`, `M`, `P`, `meas` | `--world worlds/beaver_enron.py --with-beaver` with a vLLM server (not connected in this version) |

### PrimeKG data


Download `kg.csv`, `drug_features.csv` and `disease_features.csv` from [Harvard Dataverse](https://doi.org/10.7910/DVN/IXA7BM) into `data/primekg/`, then run `python scripts/primekg_to_lumos.py data/primekg`, which takes about a minute.
- Every node becomes `L = [type, name, context]`, where the context is a drug description or a disease definition, and every edge becomes `L = [relation]`.
- `indication` and `contraindication` are renamed to the spec's `treat` and `contraindicate`.
- A pair of nodes listed with several relations keeps one of them, in the priority order indication, contraindication, off-label use.
- The full graph (129,375 nodes and 4,049,642 edges) loads in about 45 seconds and uses about 1.2 GB of memory.

## Language reference

### Syntax

| Construct | Form |
|---|---|
| Comment | `# to end of line` |
| Assignment | `x := e` (binds or rebinds; there is one global store) |
| Sequencing | one statement per line, or `s1; s2` on one line |
| Conditional | `if b then s1 else s2`; `else` is optional |
| Loop | `while b do s` |
| Bounded loop | `while i in {0, ..., q} s` (no `do`): runs `s` once for each element, in order |
| Certification | `estimateProb δ n "Certifier" :` then the body, then `toEstimate(obs)` |
| Node | `v := Node(L{"a1", ..., "ak"})` (string attributes; by convention `ak` is long context) |
| Edge | `e := Edge(v1, v2)` or `Edge(v1, v2, L{...})`, undirected |
| Graph | `G := Graph({v1, ...}, {e1, ...})` or `G := getGraph("file.csv")` |
| Naming a graph's sets | `G := Graph(N, E) := e` binds `G` to the graph `e`, `N` to its node set and `E` to its edge set |
| Graph access | `G.N`, `G.E`, `v.L[i]`, `v.L[-1]`, `NB(v)`, `getEdge(v1, v2, G)` |
| Graph union | `g1 + g2`; bare nodes and edges lift to one-element graphs |
| Sets | `{e1, e2}`, `{}`, `{a, ..., b}` (integers a..b inclusive), `{e \| x in S and b}`, `S1 + S2`, `S[i]`, `S[-i]` |
| Sampling | `sample(meas(S))` |
| Prompts | `p1 + p2`, `format("... {v} ...", g)`, `L(p)`, `tool-call(t, p)`, `concat(S)` |
| Booleans | `e1 = e2`, `e1 < e2`, `e in S`, `and`, `or`, `not`, `true`, `false` |
| Reals | numbers, `+`, `*` |

Bodies of `if`, `while` and `estimateProb` are either an indented block on the following lines or a single statement on the same line. `toEstimate(obs)` may be the last line inside the `estimateProb` block or the first line after it. `estimateProb` appears only at the top level, and a program may contain several.

A line continues onto the next one when a bracket is still open, when it ends with an operator or a comma, or when the next line starts with an operator, for example a leading `+`.

Precedence, loosest first: `or`, `and`, `not`, comparisons (`= < > <= >= in`, which cannot be chained), `+ -`, `* /`, unary `-`, then postfix call, index and member access.

Beyond the paper's grammar, the interpreter also accepts `>`, `<=`, `>=`, binary `-`, `/` and `len(S)`. C-style operators (`==`, `!=`, `&&`, `||`, `!`, `//`) are rejected with a hint.

### Semantics

These follow the paper. Where the paper is silent, the choice made is stated.

- **Graphs** are undirected, with at most one edge per pair of nodes, and every edge's endpoints must be in the graph's node set.
  - **Nodes:** every evaluation of `Node(...)`, and every node read by `getGraph`, is a distinct node. A node is equal only to itself, even when another node has the same attributes. Reassigning `v := Node(...)` makes a new node; edges declared earlier keep the old one. The variable name is only a label.
  - **Edges and graphs** compare by content. Two graphs are equal when they have the same nodes and edges.
- **`NB(v)`** gives the neighbours of `v` in the program graph, meaning the graph bound at the top level, in node order. If several graphs are bound at the top level, write `NB(v, G)`.
- **`getEdge(a, b, G)`** works in either order. When there is no edge it returns **null**. Null can be assigned to a variable. `.L` and `[i]` on null give null. Comparisons with null are false, even `e = e` when `e` is null, so set-builders simply filter such elements out. Any other use of null, such as passing it to a function, is ⊥.
- **⊥ (undefined)** comes from `sample` of an empty set, an index out of range, division by zero, or a misused null. Inside `estimateProb`, a run that hits ⊥ is **discarded and redrawn**. This is rejection sampling: the accepted runs are IID draws from the scenario distribution conditioned on being defined, and the result reports how many runs were discarded. If more than `max(100, 10n)` runs are discarded, the certification gives up with an error. Any other runtime error aborts immediately and reports the run index and seed.
- **Comparisons** `< > <= >=` work on numbers and on strings that hold numbers, such as an attribute `"0.91"`, in any mix: `"0.91" > 0.5` and `"50" < "120"` are both true. `=` is structural and never converts types: `1 = 1.0` is true, `"1" = 1` is false, and `1 = true` is false. Ordered sets compare in order.
- **`and` / `or`** short-circuit, so the right side is not evaluated, and does not consume randomness or call models, when the left side decides the result.
- **`+`** means ordered-set union or graph union (in both, the elements of the left operand come first, then the new ones of the right), string concatenation (numbers and booleans are rendered as written: `"n=" + 1` gives `"n=1"`), or addition. Other combinations are errors.
- **`sample(meas(S))`** is uniform by default. If the environment binds `meas` to a function from a list of elements to non-negative weights, that function decides. All-zero weights mean uniform. Other measure names must be registered with `lumos.measures.register_measure`.
- **`format(str, g)`** replaces each `{x}` whose value is a node or edge of `g` with one of its attributes, sampled with `meas`. Every other placeholder is left as written.
- **`estimateProb δ n "C"`**: every run starts from the same state, with its own PRNG seed derived from the interpreter seed, and contributes exactly one observation, which is checked as soon as the run ends. Clopper-Pearson (`"Clopper-Pearson"` or `"CP"`) needs a boolean per run; compare a judge's score with `toEstimate(J(r) = 1)`. Bernstein and Hoeffding accept a real in [0, 1] (or a boolean), or one `{lb, ub}` pair per run as returned by Beaver. Hoeffding's interval is mean ± √(ln(2/δ) / 2n); for pairs it is [mean lb − r, mean ub + r] with the same radius r. A `{…}` written directly inside `toEstimate` is a list of observations, as in the paper's grammar, so `toEstimate({lb, ub})` keeps both values even when lb = ub. Clopper-Pearson is the exact two-sided interval with δ/2 in each tail, computed without third-party packages.

### External functions

- **Sharing between runs.** External functions are shared by all runs, as the paper requires them to be stateless (§5), so a model or pipeline is never copied. Data inputs are copied for each run. An external that needs fresh state in every run can define `lumos_clone_for_trial(seed)`, which is called once per run to produce its copy.
- **Arguments.** Functions receive Lumos values as they are:
  - Strings, numbers and booleans are the Python values.
  - Ordered sets are `lumos.values.OrderedSet`, which iterates in order and supports `len`.
  - Nodes and edges have `.attributes`.
  - The `meas` function receives a plain Python list of the set's elements and returns one weight per element, in the same order.
- **Return values.** Values returned from Python are converted to Lumos values. Numpy and torch scalars become numbers, numpy booleans become booleans, and arrays become lists. A judge returning `np.int64(1)` therefore works with `J(r) = 1`.
- **Errors.** An error inside an external stops the certification, and the message names the function, the run and its seed.

`getGraph(path)` reads a CSV edge list, resolved relative to the program's directory (or `--data-dir`, or a world's `BASE_DIR`). The header needs `src_id` and `dst_id` columns. The other `src_*` and `dst_*` columns are the endpoints' attributes, and every remaining column is an edge attribute, each in column order. A row with an empty `dst_id` declares an isolated node.

## Differences from the paper

The interpreter follows the paper's grammar (§4) and semantics (§5, Appendix A). It also follows the concrete syntax in the paper's generator prompt (Appendix G, `nl2lumos/claude_prompt.jinja`), which is what the specs are written in. Where the paper is silent or inconsistent, this is what the interpreter does:

- **Bounded loop.** The paper defines `while i in {0, ..., q}` as `i := 0; while i < q: i := i + 1; …`, which runs q times with i = 1..q. Its examples (`while i ∈ [1..q]`, and `{1, ..., len(M)-1}` in the Beaver spec) need one iteration per element. The interpreter runs once per element, so `{0, ..., q}` runs q + 1 times with i = 0..q.
- **Node identity.** In the formal semantics a node denotes its attribute list, so two nodes with the same attributes would be one node. The interpreter keeps every `Node(...)` distinct, which the paper's Beaver spec needs: it declares several nodes as `Node(L{})`.
- **A missing edge.** `getEdge` with no edge is undefined in the paper. Here it returns null, which makes comparisons false. That lets the paper's medical spec filter drugs by `"treat" in getEdge(v, d, G).L`.
- **Undefined runs.** The paper does not say what `estimateProb` does when a run is undefined. The interpreter discards the run and draws another, and reports the count. The certificate is then for the probability conditioned on a defined scenario.
- **Short-circuit `and`/`or`.** This differs from the paper's ∧/∨ only when the skipped side would be undefined.
- **Extensions:** `>`, `<=`, `>=`, `-`, `/`, `len`, numeric strings in comparisons, `if` without `else`, and the certifiers listed above.
- **Not supported:**
  - Inline `Node(...)`/`Edge(...)` inside `Graph(...)`; the paper's generator prompt also forbids it.
  - `while b` without `do`.
  - The tuple notation `(u, v) in G.E` used to define `NB`.

## Tests

- `tests/test_lexer.py`, `test_parser.py`, `test_semantics.py` and `test_certify.py` test one construct or rule at a time, with expected values worked out by hand. The Clopper-Pearson checks compare against an independent exact binomial inversion.
- `tests/test_specs.py` runs every real spec end-to-end. Each spec has a sidecar in `tests/specs/` that supplies a mock world (`externals()`) and checks the results against an exact reference model of the spec (`check(...)`): the scenario distribution, the prompts threaded through each turn, the per-run observations and the discard rate.
- `tests/test_cli.py` covers the runner: worlds, `--trials`, tracing and exit codes. `tests/test_externals.py` covers externals of every kind: numpy and torch results, objects shared rather than copied, a transformers model in-process, and a judge over an OpenAI-compatible API.
- `tests/test_regressions.py` has one test per correctness bug found in review: the invariant cache, observation checks, node identity, graph equality and union order, null members, list/set membership, CSV loading, world loading and long operator chains.
- `tests/test_primekg.py` runs prime.lumos on the full PrimeKG against a reference computed directly from the CSV. It runs only once `data/primekg/primekg.csv` exists, and takes about 5 minutes.
- Tests that need a server run with `LUMOS_TEST_SERVER=http://host:port` (any OpenAI-compatible endpoint, such as vLLM).
- `tests/heldout/` holds specs kept out of interpreter development; see its README.
- Helpers for writing worlds and checks (call recorders, exact distributions, rate and discard checks) are in `lumos/testing.py`.

## Generating specs from natural language

`nl2lumos/` holds the tooling the paper uses to have an LLM write Lumos programs (§6.5). It contains the natural-language specs (`nlspecs/`), the prompt (`claude_prompt.jinja`), the generated programs (`spec_progs_claude/`), and generated Python baselines (`py_specs_claude/`) for comparison. The generation scripts call Claude on Amazon Bedrock and read the API key from `AWS_BEARER_TOKEN_BEDROCK`; see [`nl2lumos/README.md`](../nl2lumos/README.md).

## Repository layout

```
lumos/      the interpreter: lexer.py, parser.py, ast.py, interpreter.py, certify.py (certifiers),
            values.py, graph.py, measures.py, env.py, errors.py, cli.py,
            local_model.py (HuggingFace in-process), llm.py (vLLM), beaver_bridge.py, testing.py
demo/       tutorial.ipynb (write a spec step by step), helpers.py (run and check a spec, draw its graph, the Qwen3Guard judge)
worlds/     externals for running the specs (inputs, placeholder judges, data locations)
nl2lumos/   natural-language specs, the generator prompt, generated Lumos and Python programs
scripts/    primekg_to_lumos.py
docs/       reference.md (this page)
tests/      unit tests, spec sidecars (tests/specs), held-out specs (tests/heldout)
example.lumos, cli.py
```
