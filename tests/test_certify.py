"""estimateProb: trials, isolation, seeding, discard-and-redraw and the certifiers."""

import math
import unittest

from lumos_test_utils import run
from lumos.certify import Certifier, clopper_pearson_interval
from lumos.errors import RuntimeErrorLumos


def binomial_tail_ge(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def reference_clopper_pearson(k, n, delta):
    """Invert the exact binomial tails by bisection (independent of the Beta-function code)."""
    def solve(f, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            if (f(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda p: binomial_tail_ge(k, n, p), delta / 2, True)
    upper = 1.0 if k == n else solve(lambda p: 1 - binomial_tail_ge(k + 1, n, p), delta / 2, False)
    return lower, upper


def within(test, k, n, p, sigmas=4.0):
    tolerance = sigmas * math.sqrt(p * (1 - p) / n) + 1e-12
    test.assertLessEqual(abs(k / n - p), tolerance, f"{k}/{n} = {k / n:.4f} is not within {tolerance:.4f} of {p:.4f}")


class ClopperPearsonTests(unittest.TestCase):
    def test_closed_forms_at_the_extremes(self):
        for n in (1, 10, 200, 1000):
            with self.subTest(n=n):
                self.assertAlmostEqual(clopper_pearson_interval(0, n, 0.05)[1], 1 - 0.025 ** (1 / n), places=10)
                self.assertAlmostEqual(clopper_pearson_interval(n, n, 0.05)[0], 0.025 ** (1 / n), places=10)
                self.assertEqual(clopper_pearson_interval(0, n, 0.05)[0], 0.0)
                self.assertEqual(clopper_pearson_interval(n, n, 0.05)[1], 1.0)

    def test_matches_exact_binomial_inversion(self):
        for k, n in [(1, 2), (5, 20), (13, 40), (30, 60), (59, 60), (7, 11)]:
            for delta in (0.05, 0.01, 0.2):
                with self.subTest(k=k, n=n, delta=delta):
                    expected = reference_clopper_pearson(k, n, delta)
                    actual = clopper_pearson_interval(k, n, delta)
                    self.assertAlmostEqual(actual[0], expected[0], places=8)
                    self.assertAlmostEqual(actual[1], expected[1], places=8)

    def test_large_n_is_stable_and_symmetric(self):
        lo, hi = clopper_pearson_interval(5000, 10000, 0.05)
        self.assertAlmostEqual(lo + hi, 1.0, places=9)
        self.assertLess(hi - lo, 0.021)


class TrialTests(unittest.TestCase):
    def test_runs_are_isolated_and_reproducible(self):
        seen = []
        source = ('x := 5\nestimateProb 0.05 50 "Clopper-Pearson" :\n    x := x + 1\n'
                  '    y := sample(meas({1, ..., 1000}))\n    r := record(y)\ntoEstimate(x = 6)')
        interp, results = run(source, record=lambda y: seen.append(y) or 1)
        result = results[0]
        self.assertEqual((result.successes, result.trials, result.discarded), (50, 50, 0))
        self.assertEqual(interp.env.get("x"), 5)            # trial assignments do not leak
        self.assertNotIn("y", interp.env.bindings)
        self.assertEqual(len(seen), 50)
        self.assertGreater(len(set(seen)), 40)              # independent draws per run
        self.assertEqual(len(set(result.seeds)), 50)
        again = []
        _, results2 = run(source, record=lambda y: again.append(y) or 1)
        self.assertEqual(results2[0].seeds, result.seeds)
        self.assertEqual(again, seen)
        other = []
        _, results3 = run(source, seed=1, record=lambda y: other.append(y) or 1)
        self.assertNotEqual(other, seen)

    def test_one_boolean_observation_per_run(self):
        _, results = run('estimateProb 0.05 400 "Clopper-Pearson" :\n    x := sample(meas({1, 2, 3, 4}))\n'
                         'toEstimate(x = 1 or x = 2)')
        result = results[0]
        self.assertEqual(len(result.observations), 400)
        within(self, result.successes, 400, 0.5)
        self.assertEqual((result.lower_ci, result.upper_ci),
                         clopper_pearson_interval(result.successes, 400, 0.05))

    def test_observation_errors_for_clopper_pearson(self):
        for obs, fragment in [("1", "compare the score"), ("{true, false}", "Bernstein"), ('"yes"', "boolean")]:
            with self.subTest(obs=obs):
                with self.assertRaises(RuntimeErrorLumos) as ctx:
                    run(f'estimateProb 0.05 5 "Clopper-Pearson" : x := 1; toEstimate({obs})')
                self.assertIn(fragment, str(ctx.exception))

    def test_program_graph_is_fixed_inside_trials(self):
        source = ('a := Node(L{"a"})\nb := Node(L{"b"})\nab := Edge(a, b)\nG := Graph({a, b}, {ab})\n'
                  'estimateProb 0.05 10 "CP":\n    H := Graph({a}, {})\n    n := NB(a)\ntoEstimate(b in n)')
        interp, results = run(source)
        self.assertEqual(results[0].successes, 10)
        self.assertEqual(list(interp.program_graphs), ["G"])

    def test_multiple_estimate_probs_and_statements_in_order(self):
        source = ('k := 1\nestimateProb 0.05 10 "CP": x := k; toEstimate(x = 1)\nk := 2\n'
                  'estimateProb 0.05 10 "CP": x := k; toEstimate(x = 1)')
        _, results = run(source)
        self.assertEqual([r.successes for r in results], [10, 0])

    def test_errors_abort_with_run_and_line(self):
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            run('estimateProb 0.05 10 "CP" :\n    x := 1\n    y := x + {1}\ntoEstimate(true)')
        self.assertEqual(ctx.exception.line, 3)
        self.assertIn("estimateProb run 0", str(ctx.exception))
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            run('estimateProb 0.05 10 "CP" :\n    x := 1\ntoEstimate(x < "a")')
        self.assertEqual(ctx.exception.line, 3)

    def test_header_validation_happens_before_any_run(self):
        calls = []
        for header, fragment in [('0.05 10 "Chernoff"', "Unknown certifier"),
                                 ('1.5 10 "CP"', "delta"),
                                 ('0.05 0 "CP"', "positive")]:
            with self.subTest(header=header):
                with self.assertRaises(RuntimeErrorLumos) as ctx:
                    run(f'estimateProb {header} : r := L("p"); toEstimate(true)', L=lambda p: calls.append(p))
                self.assertIn(fragment, str(ctx.exception))
        self.assertEqual(calls, [])


class DiscardTests(unittest.TestCase):
    """A run that hits ⊥ is discarded and redrawn (rejection sampling)."""

    def test_conditional_estimate_and_discard_count(self):
        # x = 2 makes the index undefined, so accepted runs have x in {0, 1}: P(x = 1 | defined) = 1/2.
        _, results = run('estimateProb 0.05 2000 "CP" :\n    x := sample(meas({0, 1, 2}))\n'
                         '    y := {10, 20}[x]\ntoEstimate(x = 1)')
        result = results[0]
        self.assertEqual(result.trials, 2000)
        within(self, result.successes, 2000, 0.5)
        # Failures before 2000 successes with success probability 2/3: mean 1000, sd ~38.7.
        self.assertLess(abs(result.discarded - 1000), 4 * 38.8)
        self.assertEqual(len(result.discarded_seeds), result.discarded)
        self.assertIn("discarded", str(result))

    def test_gives_up_when_everything_is_undefined(self):
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            run('estimateProb 0.05 5 "CP" : x := sample(meas({})); toEstimate(true)')
        message = str(ctx.exception)
        self.assertIn("gave up", message)
        self.assertIn("S empty", message)

    def test_undefined_observation_is_discarded_too(self):
        _, results = run('estimateProb 0.05 20 "CP" :\n    x := sample(meas({0, 1}))\ntoEstimate({true}[x])')
        self.assertEqual(results[0].successes, 20)
        self.assertGreater(results[0].discarded, 0)


class BernsteinTests(unittest.TestCase):
    def test_real_observations(self):
        _, results = run('estimateProb 0.05 1000 "Bernstein" :\n    s := sample(meas({0.2, 0.4}))\ntoEstimate(s)')
        result = results[0]
        self.assertEqual(result.observation_mode, "real")
        self.assertAlmostEqual(result.empirical_prob, sum(result.observations) / 1000)
        self.assertLess(abs(result.empirical_prob - 0.3), 0.02)
        self.assertLess(result.lower_ci, 0.3)
        self.assertGreater(result.upper_ci, 0.3)
        _, small = run('estimateProb 0.05 100 "Bernstein" :\n    s := sample(meas({0.2, 0.4}))\ntoEstimate(s)')
        self.assertLess(result.upper_ci - result.lower_ci, small[0].upper_ci - small[0].lower_ci)

    def test_booleans_count_as_zero_or_one(self):
        _, results = run('estimateProb 0.05 50 "Bernstein" : x := 1; toEstimate(x = 1)')
        self.assertEqual(results[0].empirical_prob, 1.0)

    def test_pair_observations(self):
        _, results = run('estimateProb 0.1 200 "Bernstein" : b := B("p"); toEstimate(b)', B=lambda p: [0.25, 0.75])
        result = results[0]
        self.assertEqual(result.observation_mode, "pair")
        self.assertEqual(result.observations[0], (0.25, 0.75))
        self.assertLessEqual(result.lower_ci, 0.25)
        self.assertGreaterEqual(result.upper_ci, 0.75)
        self.assertAlmostEqual(result.empirical_prob, 0.5)

    def test_bad_observations(self):
        cases = [
            ("B(\"p\")", lambda p: [0.8, 0.2], "lb <= ub"),
            ("B(\"p\")", lambda p: [0.1, 0.2, 0.3], "pair"),
            ("B(\"p\")", lambda p: 1.5, "[0, 1]"),
            ("B(\"p\")", lambda p: "x", "[0, 1]"),
        ]
        for obs, fn, fragment in cases:
            with self.subTest(fragment=fragment):
                with self.assertRaises(RuntimeErrorLumos) as ctx:
                    run(f'estimateProb 0.05 5 "Bernstein" : b := {obs}; toEstimate(b)', B=fn)
                self.assertIn(fragment, str(ctx.exception))

    def test_modes_cannot_mix(self):
        flip = iter([0.5, [0.1, 0.2]] * 5)
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            run('estimateProb 0.05 4 "Bernstein" : b := B("p"); toEstimate(b)', B=lambda p: next(flip))
        self.assertIn("mix", str(ctx.exception))


class HoeffdingTests(unittest.TestCase):
    """mean ± sqrt(ln(2/δ) / 2n), clipped to [0, 1]."""

    def test_real_observations_give_the_closed_form(self):
        _, results = run('estimateProb 0.05 200 "Hoeffding" :\n    s := sample(meas({0.2, 0.4}))\ntoEstimate(s)')
        result = results[0]
        mean = sum(result.observations) / 200
        radius = math.sqrt(math.log(2 / 0.05) / (2 * 200))
        self.assertEqual(result.observation_mode, "real")
        self.assertEqual(set(result.observations), {0.2, 0.4})
        self.assertAlmostEqual(result.empirical_prob, mean)
        self.assertAlmostEqual(result.lower_ci, mean - radius)
        self.assertAlmostEqual(result.upper_ci, mean + radius)
        self.assertEqual(result.one_minus_delta, 0.95)

    def test_known_radius(self):
        # n = 100, δ = 0.05: sqrt(ln 40 / 200) = 0.13581 (worked by hand).
        result = Certifier().certify("Hoeffding", 0.95, [True] * 30 + [False] * 70)
        self.assertAlmostEqual(result.empirical_prob, 0.3)
        self.assertAlmostEqual(result.lower_ci, 0.3 - 0.13581, places=5)
        self.assertAlmostEqual(result.upper_ci, 0.3 + 0.13581, places=5)
        # Wider than the exact binomial interval on the same data.
        low, high = clopper_pearson_interval(30, 100, 0.05)
        self.assertLess(result.lower_ci, low)
        self.assertGreater(result.upper_ci, high)

    def test_name_is_case_insensitive_and_the_interval_is_clipped(self):
        _, results = run('estimateProb 0.05 50 "hoeffding" : toEstimate(true)')
        result = results[0]
        self.assertEqual(result.empirical_prob, 1.0)
        self.assertEqual(result.upper_ci, 1.0)
        self.assertAlmostEqual(result.lower_ci, 1.0 - math.sqrt(math.log(40) / 100))
        low = Certifier().certify("Hoeffding", 0.95, [0.0] * 10)
        self.assertEqual(low.lower_ci, 0.0)

    def test_pair_observations(self):
        _, results = run('estimateProb 0.1 200 "Hoeffding" : b := B("p"); toEstimate(b)', B=lambda p: [0.25, 0.75])
        result = results[0]
        radius = math.sqrt(math.log(2 / 0.1) / (2 * 200))
        self.assertEqual(result.observation_mode, "pair")
        self.assertEqual(result.observations[0], (0.25, 0.75))
        self.assertAlmostEqual(result.lower_ci, 0.25 - radius)
        self.assertAlmostEqual(result.upper_ci, 0.75 + radius)
        self.assertAlmostEqual(result.empirical_prob, 0.5)

    def test_coverage(self):
        import random
        rng = random.Random(7)
        misses = 0
        for _ in range(400):
            sample = [rng.random() < 0.3 for _ in range(50)]
            result = Certifier().certify("Hoeffding", 0.95, sample)
            misses += not (result.lower_ci <= 0.3 <= result.upper_ci)
        self.assertLessEqual(misses, 20)  # at most δ = 5% of 400; Hoeffding is conservative

    def test_bad_observations(self):
        cases = [(lambda p: [0.8, 0.2], "lb <= ub"), (lambda p: [0.1, 0.2, 0.3], "pair"),
                 (lambda p: 1.5, "[0, 1]"), (lambda p: "x", "[0, 1]")]
        for fn, fragment in cases:
            with self.subTest(fragment=fragment):
                with self.assertRaises(RuntimeErrorLumos) as ctx:
                    run('estimateProb 0.05 5 "Hoeffding" : b := B("p"); toEstimate(b)', B=fn)
                self.assertIn(fragment, str(ctx.exception))
                self.assertIn("Hoeffding", str(ctx.exception))
        flip = iter([0.5, [0.1, 0.2]] * 5)
        with self.assertRaises(RuntimeErrorLumos) as ctx:
            run('estimateProb 0.05 4 "Hoeffding" : b := B("p"); toEstimate(b)', B=lambda p: next(flip))
        self.assertIn("mix", str(ctx.exception))


class InvariantCachingTests(unittest.TestCase):
    """Set-builders that are the same in every trial are computed once per estimateProb."""

    SOURCE = '''
a := Node(L{"x"})
b := Node(L{"x"})
c := Node(L{"y"})
ab := Edge(a, b)
bc := Edge(b, c)
G := Graph({a, b, c}, {ab, bc})
estimateProb 0.05 20 "CP" :
    S := {v | v in G.N and v.L[0] = "x"}
    n := sample(meas(G.N))
    T := {v | v in G.N and not (v = n)}
    U := {w | w in S and {u | u in NB(w) and not (u = c)} = {u | u in NB(w)}}
    S := S + {c}
toEstimate(n in T or len(U) = 1)
'''

    def evaluations(self, cache):
        from lumos.interpreter import Interpreter
        from lumos_test_utils import parse
        interp = Interpreter(seed=0)
        counts = {}
        compute = interp._compute_set_builder

        def counting(expr):
            counts[expr.variable] = counts.get(expr.variable, 0) + 1
            return compute(expr)
        interp._compute_set_builder = counting
        results = Certifier(cache_invariants=cache).run_program(parse(self.SOURCE), interp)
        return counts, results[0]

    def test_only_invariant_builders_are_cached(self):
        counts, result = self.evaluations(cache=True)
        # S reads only G (the body reassigns S, but the builder itself reads only G): once.
        # T depends on the sampled n, and the outer U reads S, which the body assigns: every run.
        # The inner {u | ...} builders read U's binder w, so they run per element of every run.
        self.assertEqual(counts["v"], 1 + 20)
        self.assertEqual(counts["w"], 20)
        self.assertEqual(counts["u"], 20 * 2 * 2)
        uncached, same = self.evaluations(cache=False)
        self.assertEqual(uncached["v"], 20 + 20)
        self.assertEqual(result.observations, same.observations)
        self.assertEqual(result.seeds, same.seeds)

    def test_sets_named_by_a_graph_binding_in_the_body_are_not_invariant(self):
        # N is assigned by `H := Graph(N, E) := ...` in every run, so {u | u in N} must not be cached.
        source = '''
a := Node(L{"x"})
b := Node(L{"y"})
c := Node(L{"z"})
G := Graph({a, b, c}, {})
estimateProb 0.05 60 "CP" :
    v := sample(meas(G.N))
    H := Graph(N, E) := v + c
    S := {u | u in N}
toEstimate(len(S) = len(H.N) and v in S)
'''
        _, results = run(source)
        self.assertEqual(results[0].successes, 60)

    def test_caching_does_not_change_a_real_spec(self):
        from lumos_test_utils import REPO, parse
        from lumos.interpreter import Interpreter
        source = (REPO / "nl2lumos/spec_progs_claude/prime.lumos").read_text()
        runs = []
        for cache in (True, False):
            interp = Interpreter(seed=5, base_dir=REPO / "tests/specs/fixtures")
            interp.env.set("L", lambda p: "A)")
            interp.env.set("J_QA", lambda r, gt: 1 if r == gt else 0)
            runs.append(Certifier(cache_invariants=cache).run_program(parse(source), interp)[0])
        self.assertEqual(runs[0].observations, runs[1].observations)
        self.assertEqual(runs[0].seeds, runs[1].seeds)
        self.assertEqual(runs[0].discarded_seeds, runs[1].discarded_seeds)


class CertifierApiTests(unittest.TestCase):
    def test_certify_directly(self):
        result = Certifier().certify("Clopper-Pearson", 0.95, [True] * 7 + [False] * 3)
        self.assertEqual((result.successes, result.trials), (7, 10))
        self.assertEqual((result.lower_ci, result.upper_ci), clopper_pearson_interval(7, 10, 0.05))

    def test_statsmodels_methods_need_statsmodels(self):
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            with self.assertRaises(RuntimeErrorLumos) as ctx:
                Certifier().certify("wilson", 0.95, [True, False])
            self.assertIn("statsmodels", str(ctx.exception))


def _has_statsmodels():
    try:
        import statsmodels  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_statsmodels(), "statsmodels is not installed")
class StatsmodelsCertifierTests(unittest.TestCase):
    CASES = [(0, 10), (3, 10), (10, 10), (17, 50), (480, 1000), (999, 1000)]

    def test_methods_match_proportion_confint(self):
        from statsmodels.stats.proportion import proportion_confint
        for method, sm_method in [("wilson", "wilson"), ("normal", "normal"), ("agresti_coull", "agresti_coull"),
                                  ("Agresti-Coull", "agresti_coull")]:
            for k, n in self.CASES:
                for delta in (0.05, 0.1):
                    with self.subTest(method=method, k=k, n=n, delta=delta):
                        result = Certifier().certify(method, 1 - delta, [True] * k + [False] * (n - k))
                        expected = proportion_confint(k, n, alpha=delta, method=sm_method)
                        self.assertAlmostEqual(result.lower_ci, float(expected[0]), places=12)
                        self.assertAlmostEqual(result.upper_ci, float(expected[1]), places=12)

    def test_native_clopper_pearson_matches_statsmodels_beta(self):
        from statsmodels.stats.proportion import proportion_confint
        for k, n in self.CASES + [(1, 1), (0, 1), (250, 5000), (4999, 5000)]:
            for delta in (0.05, 0.01):
                with self.subTest(k=k, n=n, delta=delta):
                    expected = proportion_confint(k, n, alpha=delta, method="beta")
                    actual = clopper_pearson_interval(k, n, delta)
                    self.assertAlmostEqual(actual[0], float(expected[0]), places=9)
                    self.assertAlmostEqual(actual[1], float(expected[1]), places=9)

    def test_statsmodels_certifier_in_a_program(self):
        _, results = run('estimateProb 0.05 200 "wilson" :\n    x := sample(meas({1, 2}))\ntoEstimate(x = 1)')
        result = results[0]
        self.assertEqual(result.method, "wilson")
        self.assertLess(result.lower_ci, result.empirical_prob)
        self.assertGreater(result.upper_ci, result.empirical_prob)


if __name__ == "__main__":
    unittest.main()
