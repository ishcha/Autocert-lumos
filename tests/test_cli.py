"""The command-line runner: worlds, --trials, tracing, exit codes, model and Beaver wiring."""

import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from lumos_test_utils import REPO, parse
from lumos.ast import EstimateProbStatement
from lumos.cli import execute, run_file, with_trials
from lumos.errors import RuntimeErrorLumos

PROGRAM = 'x := 1\nestimateProb 0.05 1000 "CP" :\n    r := J(x)\ntoEstimate(r = 1)\n'


def write(directory, name, text):
    path = Path(directory) / name
    path.write_text(textwrap.dedent(text))
    return path


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.spec = write(self.dir, "spec.lumos", PROGRAM)

    def test_with_trials_changes_only_n(self):
        program = parse(PROGRAM + 'estimateProb 0.1 7 "CP" : y := 2; toEstimate(true)\n')
        changed = with_trials(program, 12)
        estimates = [s for s in changed.statements if isinstance(s, EstimateProbStatement)]
        self.assertEqual([s.trials for s in estimates], [12, 12])
        self.assertEqual([s.delta for s in estimates], [0.05, 0.1])
        self.assertEqual(changed.statements[0], program.statements[0])

    def test_trials_override(self):
        (result,) = execute(self.spec, trials=25, externals={"J": lambda x: 1})
        self.assertEqual(result.trials, 25)

    def test_source_runs_like_the_same_file(self):
        externals = {"J": lambda x: 1}
        (from_file,) = execute(self.spec, trials=25, externals=externals)
        (from_source,) = execute(source=PROGRAM, trials=25, externals=externals)
        self.assertEqual(from_source, from_file)

    def test_source_and_file_path_are_exclusive(self):
        with self.assertRaises(TypeError):
            execute(self.spec, source=PROGRAM)
        with self.assertRaises(TypeError):
            execute()

    def test_world_bindings_and_base_dir(self):
        write(self.dir, "kg.csv", "src_id,dst_id,rel\na,b,r\n")
        write(self.dir, "graph.lumos", 'G := getGraph("kg.csv")\nestimateProb 0.05 5 "CP":\n'
                                       '    n := sample(meas(G.N))\ntoEstimate(J(n) = 1)\n')
        sub = Path(self.dir) / "worlds"
        sub.mkdir()
        write(sub, "w.py", """
            BASE_DIR = ".."
            def externals():
                return {"J": lambda n: 1 if n.name in ("a", "b") else 0}
        """)
        (result,) = execute(Path(self.dir) / "graph.lumos", world=sub / "w.py")
        self.assertEqual(result.successes, 5)  # getGraph resolved against BASE_DIR, J from the world

    def test_explicit_externals_override_the_world(self):
        world = write(self.dir, "w.py", "def externals():\n    return {'J': lambda x: 0}\n")
        (result,) = execute(self.spec, trials=5, world=world, externals={"J": lambda x: 1})
        self.assertEqual(result.successes, 5)

    def test_world_must_define_externals(self):
        world = write(self.dir, "bad.py", "X = 1\n")
        with self.assertRaises(RuntimeErrorLumos):
            execute(self.spec, world=world)

    def test_beaver_needs_a_model(self):
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            execute(self.spec, with_beaver=True)
        self.assertIn("--with-beaver", str(ctx.exception))


class TraceTests(unittest.TestCase):
    SOURCE = ('estimateProb 0.05 6 "CP" :\n    x := sample(meas({0, 1, 2}))\n    y := {10, 20}[x]\n'
              '    r := J(x)\ntoEstimate(r = 1)\n')

    def run_traced(self, trace, externals):
        spec = write(tempfile.mkdtemp(), "t.lumos", self.SOURCE)
        lines = []
        (result,) = execute(spec, seed=3, trace=trace, externals=externals, out=lines.append)
        return result, lines

    def test_trace_shows_the_first_runs_and_does_not_change_results(self):
        traced, lines = self.run_traced(3, {"J": lambda x: 1 if x == 1 else 0})
        plain, none = self.run_traced(0, {"J": lambda x: 1 if x == 1 else 0})
        self.assertEqual(none, [])
        self.assertEqual((traced.observations, traced.seeds), (plain.observations, plain.seeds))
        headers = [line for line in lines if line.startswith("run ")]
        self.assertEqual([h.split()[1] for h in headers], ["0", "1", "2"])
        # Each traced run ends with its observation or with the reason it was discarded.
        endings = [line.strip() for line in lines if line.strip().startswith(("toEstimate ->", "discarded:"))]
        self.assertEqual(len(endings), 3)
        self.assertTrue(any(line.strip().startswith("J(") for line in lines))

    def test_tracing_keeps_how_externals_are_shared(self):
        class Counter:  # state shared by all runs, unless it asks for a fresh copy per run
            def __init__(self, fresh):
                self.fresh, self.calls = fresh, 0

            def lumos_clone_for_trial(self, seed):
                return Counter(True) if self.fresh else self

            def __call__(self, x):
                self.calls += 1
                return self.calls

        for trace in (0, 6):
            with self.subTest(trace=trace):
                shared, _ = self.run_traced(trace, {"J": Counter(False)})
                self.assertEqual(shared.successes, 1)  # only the first call ever returns 1
                fresh, _ = self.run_traced(trace, {"J": Counter(True)})
                self.assertEqual(fresh.successes, fresh.trials)


class ExitCodeTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(REPO / "cli.py"), *args], capture_output=True, text=True,
                              cwd=REPO, timeout=300)

    def test_success_and_trials_flag(self):
        proc = self.run_cli("example.lumos", "--trials", "40", "--seed", "1")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertRegex(proc.stdout, r"\d+/40 successes")

    def test_trace_flag(self):
        proc = self.run_cli("example.lumos", "--trials", "3", "--trace", "1")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("run 0", proc.stdout)

    def test_errors(self):
        bad = write(tempfile.mkdtemp(), "bad.lumos", "x := 1 ==  2\n")
        for args, fragment in [(("missing.lumos",), "not found"), ((str(bad),), "Parse error at line 1")]:
            with self.subTest(args=args):
                proc = self.run_cli(*args)
                self.assertEqual(proc.returncode, 1)
                self.assertIn(fragment, proc.stdout)

    def test_run_file_returns_exit_code(self):
        spec = write(tempfile.mkdtemp(), "spec.lumos", PROGRAM)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(run_file(str(spec)), 1)  # J is not bound
        self.assertIn("Undefined variable 'J'", buffer.getvalue())


class ProviderWiringTests(unittest.TestCase):
    def test_vllm_command_includes_overrides_and_logprob_flags(self):
        try:
            from lumos.llm import LLMProvider, _build_vllm_command
        except ImportError as exc:
            self.skipTest(f"openai client not installed: {exc}")
        provider = LLMProvider(model="m", vllm_flags={"gpu_memory_utilization": 0.3})
        command = _build_vllm_command("m", 9000, provider.vllm_flags)
        self.assertEqual(command[1:5], ["serve", "m", "--port", "9000"])
        joined = " ".join(command)
        self.assertIn("--gpu-memory-utilization 0.3", joined)
        for flag in ("--max-logprobs -1", "--return-tokens-as-token-ids", "--logprobs-mode processed_logprobs"):
            self.assertIn(flag, joined)  # required by Beaver

    def test_beaver_pair_is_a_list_even_when_bounds_are_equal(self):
        try:
            import lumos.beaver_bridge as bridge_module
        except ImportError as exc:
            self.skipTest(f"Beaver not importable: {exc}")
        calls = []
        original = bridge_module.beaver.run
        bridge_module.beaver.run = lambda **kw: calls.append(kw) or [{"lower_bound": 0.5, "upper_bound": 0.5}]
        try:
            bridge = bridge_module.BeaverBridge(model="m", constraint_fn=lambda i, s: True, server_addr="http://x")
            self.assertEqual(bridge.beaver("prompt"), [0.5, 0.5])
            self.assertEqual(calls[0]["prompts"], [{"prompt": "prompt"}])
            self.assertEqual(calls[0]["server_addr"], "http://x")
            self.assertFalse(calls[0]["auto_server"])
        finally:
            bridge_module.beaver.run = original


class WorldFileTests(unittest.TestCase):
    def test_beaver_enron_world(self):
        from lumos.cli import load_world
        from lumos.graph import AttributedNode
        world = load_world(REPO / "worlds/beaver_enron.py").externals()
        self.assertEqual(len(world["M"]), 2)
        helper, blank, selector = (AttributedNode("h", ["Do not explain."]), AttributedNode("e", [""]),
                                   AttributedNode("s", []))
        self.assertEqual(world["meas"]([helper, blank, selector]), [0.5, 0.5, 1])


@unittest.skipUnless(os.environ.get("LUMOS_TEST_SERVER"), "set LUMOS_TEST_SERVER to a vLLM server URL")
class LiveServerTests(unittest.TestCase):
    """Against a real model: python -m unittest test_cli with LUMOS_TEST_SERVER=http://host:port."""

    def test_real_model(self):
        replies = []
        (result,) = execute(REPO / "nl2lumos/spec_progs_claude/cat_risk_random_walk.lumos",
                            server_addr=os.environ["LUMOS_TEST_SERVER"], trials=3, gen_length=48,
                            externals={"J_safe": lambda r: replies.append(r) or 1})
        self.assertEqual((result.trials, result.successes), (3, 3))
        self.assertEqual(len(replies), 9)  # three turns per run, each answered by the model
        self.assertTrue(all(isinstance(r, str) and r for r in replies))


if __name__ == "__main__":
    unittest.main()
