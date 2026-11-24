"""Certification engine for Lumos: runs estimateProb trials and certifies them.

Following the paper's LMS-Certification rule, every trial runs the body from
the same initial state with an independent PRNG seed and contributes exactly
one observation. A trial that hits ⊥ (UndefinedValue) is discarded and
redrawn, i.e. rejection sampling: the accepted trials are IID draws from the
scenario distribution conditioned on the scenario being defined.
"""

import math
from dataclasses import dataclass, fields
from typing import Any, FrozenSet, Iterable, List, Optional, Set, Tuple

from .ast import *
from .errors import RuntimeErrorLumos, UndefinedValue
from .interpreter import Interpreter, type_name
from .ordered_set import OrderedSet


@dataclass
class CertificationResult:
    """Result of one estimateProb statement."""
    successes: int
    trials: int
    empirical_prob: float
    lower_ci: Optional[float] = None
    upper_ci: Optional[float] = None
    method: Optional[str] = None
    one_minus_delta: Optional[float] = None
    seeds: Optional[List[int]] = None
    observation_mode: Optional[str] = None  # 'binary', 'real' or 'pair'
    observations: Optional[List[Any]] = None
    discarded: int = 0
    discarded_seeds: Optional[List[int]] = None

    def __str__(self) -> str:
        interval = ""
        if self.lower_ci is not None and self.upper_ci is not None:
            interval = f", CI[{self.lower_ci:.4f}, {self.upper_ci:.4f}] via {self.method}"
        if self.observation_mode in ("real", "pair"):
            text = f"Certification Result: mean={self.empirical_prob:.4f} over {self.trials} runs{interval}"
        else:
            text = (
                f"Certification Result: {self.successes}/{self.trials} successes "
                f"({self.empirical_prob:.4f} probability){interval}"
            )
        if self.discarded:
            text += (
                f"; {self.discarded} runs discarded as undefined (⊥), so the bound is "
                "conditional on a defined scenario"
            )
        return text


# --- Clopper-Pearson without SciPy/statsmodels ----------------------------

def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 100_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return h


def regularized_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the CDF of Beta(a, b) at x."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _beta_continued_fraction(a, b, x) / a
    return 1.0 - math.exp(log_front) * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_quantile(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if regularized_beta(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def clopper_pearson_interval(successes: int, trials: int, delta: float) -> Tuple[float, float]:
    """Exact two-sided (1 - delta) Clopper-Pearson interval."""
    lower = 0.0 if successes == 0 else _beta_quantile(delta / 2, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else _beta_quantile(1 - delta / 2, successes + 1, trials - successes)
    return lower, upper


# --- Trial-invariant set-builders -------------------------------------------

_NEVER_INVARIANT = (SampleExpression, FormatExpression, CallExpression, ToolCallExpression,
                    NodeConstructor, EdgeConstructor)


def _children(node: Node) -> Iterable[Node]:
    for f in fields(node):
        value = getattr(node, f.name)
        if isinstance(value, Node):
            yield value
        elif isinstance(value, list):
            yield from (v for v in value if isinstance(v, Node))


def _walk(node: Node) -> Iterable[Node]:
    yield node
    for child in _children(node):
        yield from _walk(child)


def _is_invariant(expr: Expression, varying: Set[str], bound: FrozenSet[str]) -> bool:
    if isinstance(expr, _NEVER_INVARIANT):
        return False  # draws randomness or calls an external function
    if isinstance(expr, Identifier):
        return expr.name in bound or expr.name not in varying
    if isinstance(expr, SetBuilder):
        inner = bound | {expr.variable}
        return (
            _is_invariant(expr.source, varying, bound)
            and _is_invariant(expr.element, varying, inner)
            and (expr.condition is None or _is_invariant(expr.condition, varying, inner))
        )
    return all(_is_invariant(child, varying, bound) for child in _children(expr))


_PER_RUN_NAMES = {"__lumos_trial_index__", "__lumos_trial_seed__"}


def trial_invariant_set_builders(stmt: EstimateProbStatement, bindings: Optional[dict] = None) -> FrozenSet[int]:
    """Set-builders in an estimateProb whose value is identical in every trial.

    Such a set-builder draws no randomness, calls no external function, and
    reads only variables that hold the same value at the start of every trial:
    the trial body never assigns them, they are not the injected run index or
    seed, and their binding is not re-created per run by lumos_clone_for_trial.
    Computing it once is therefore invisible to the program: same values, same
    PRNG draws. Example: drugs := {v | v in G.N and v.L[0] = "drug"}.
    """
    nodes = [n for root in [*stmt.body, stmt.to_estimate] for n in _walk(root)]
    varying: Set[str] = set(_PER_RUN_NAMES)
    for name, value in (bindings or {}).items():
        if callable(getattr(value, "lumos_clone_for_trial", None)):
            varying.add(name)
    for node in nodes:
        if isinstance(node, AssignStatement):
            varying.add(node.name)
        elif isinstance(node, GraphBindStatement):
            varying.update((node.name, node.nodes_name, node.edges_name))
        elif isinstance(node, ForEachStatement):
            varying.add(node.variable)
        elif isinstance(node, SetBuilder):
            varying.add(node.variable)  # differs per element when used inside another builder
    return frozenset(
        id(node) for node in nodes
        if isinstance(node, SetBuilder) and _is_invariant(node, varying, frozenset())
    )


_BINOMIAL_METHODS = {
    "clopper-pearson": "beta",
    "cp": "beta",
    "beta": "beta",
    "wilson": "wilson",
    "normal": "normal",
    "agresti_coull": "agresti_coull",
    "agresti-coull": "agresti_coull",
}
_MEAN_METHODS = {"bernstein", "hoeffding"}  # bound a mean of values in [0, 1]
_SUPPORTED = "Clopper-Pearson, Bernstein, Hoeffding, wilson, normal, agresti_coull"


class Certifier:
    """Runs estimateProb statements."""

    def __init__(self, max_discard_factor: int = 10, cache_invariants: bool = True, observer=None):
        # A run may discard up to max(100, max_discard_factor * n) trials as ⊥.
        self.max_discard_factor = max_discard_factor
        # Compute trial-invariant set-builders once per estimateProb (a pure speed-up).
        self.cache_invariants = cache_invariants
        # Optional hooks: run_started(index, seed), run_finished(index, observation),
        # run_discarded(index, error), estimate_finished(). Used for tracing.
        self.observer = observer

    def run_program(self, program: Program, interpreter: Interpreter) -> List[CertificationResult]:
        """Execute top-level statements in order, certifying each estimateProb."""
        results = []
        for stmt in program.statements:
            if isinstance(stmt, EstimateProbStatement):
                results.append(self.run_estimate_prob(stmt, interpreter))
            else:
                interpreter.eval_statement(stmt)
        return results

    def run_estimate_prob(self, stmt: EstimateProbStatement, interpreter: Interpreter) -> CertificationResult:
        kind = self._method_kind(stmt.method, stmt.line)  # fail before running any trial
        if stmt.trials <= 0:
            raise RuntimeErrorLumos(f"Number of trials must be positive, got {stmt.trials}", stmt.line)
        if not 0.0 < stmt.delta < 1.0:
            raise RuntimeErrorLumos(f"delta must be in (0, 1), got {stmt.delta}", stmt.line)

        budget = max(100, self.max_discard_factor * stmt.trials)
        observations: List[Any] = []
        seeds: List[int] = []
        discarded_seeds: List[int] = []
        first_undefined: Optional[UndefinedValue] = None
        mean_mode: List[Optional[str]] = [None]
        if self.cache_invariants:
            interpreter.invariant_set_builders = trial_invariant_set_builders(stmt, interpreter.env.bindings)
        try:
            while len(observations) < stmt.trials:
                seed = interpreter.rng.getrandbits(64)
                index = len(observations) + len(discarded_seeds)
                self._notify("run_started", index, seed)
                try:
                    observation = self._run_trial(stmt, interpreter, seed, index)
                except UndefinedValue as exc:
                    self._notify("run_discarded", index, exc)
                    discarded_seeds.append(seed)
                    first_undefined = first_undefined or exc
                    if len(discarded_seeds) > budget:
                        raise RuntimeErrorLumos(
                            f"estimateProb gave up: {len(discarded_seeds)} runs hit an undefined value (⊥) "
                            f"before {stmt.trials} defined runs were drawn ({len(observations)} so far). "
                            f"First: {first_undefined}",
                            stmt.line,
                        ) from first_undefined
                    continue
                except RuntimeErrorLumos as exc:
                    exc.message += f" [estimateProb run {index}, seed {seed}]"
                    raise
                try:  # check each observation as soon as its run ends
                    observation = self._check_observation(kind, stmt.method, observation, mean_mode)
                except RuntimeErrorLumos as exc:
                    exc.line = exc.line or stmt.to_estimate_line or stmt.line
                    exc.message += f" [estimateProb run {index}, seed {seed}]"
                    raise
                self._notify("run_finished", index, observation)
                observations.append(observation)
                seeds.append(seed)
        finally:
            interpreter.invariant_set_builders = frozenset()
            interpreter.invariant_values = {}
            self._notify("estimate_finished")

        result = self.certify(stmt.method, 1.0 - stmt.delta, observations)
        result.seeds = seeds
        result.discarded = len(discarded_seeds)
        result.discarded_seeds = discarded_seeds
        return result

    def _notify(self, event: str, *args) -> None:
        handler = getattr(self.observer, event, None)
        if handler is not None:
            handler(*args)

    @staticmethod
    def _run_trial(stmt: EstimateProbStatement, interpreter: Interpreter, seed: int, index: int) -> Any:
        original_env, original_rng = interpreter.env, interpreter.rng
        trial_env = original_env.copy_for_trial(seed)
        trial_env.bindings["__lumos_trial_index__"] = index
        trial_env.bindings["__lumos_trial_seed__"] = seed
        interpreter.env = trial_env
        interpreter.rng = type(original_rng)(seed)
        interpreter.in_trial = True
        try:
            for body_stmt in stmt.body:
                interpreter.eval_statement(body_stmt)
            try:
                obs = stmt.to_estimate
                if isinstance(obs, OrderedSetLiteral):
                    # toEstimate({e1, e2, ...}) is a list of observations (paper: obs ::=
                    # {expr_r, ...}), so equal values are kept, e.g. {lb, ub} with lb = ub.
                    return [interpreter._not_null(interpreter.eval_expression(e), "toEstimate")
                            for e in obs.elements]
                return interpreter.eval_expression(obs)
            except RuntimeErrorLumos as exc:
                if exc.line is None:
                    exc.line = stmt.to_estimate_line or stmt.line
                raise
        finally:
            interpreter.env, interpreter.rng = original_env, original_rng
            interpreter.in_trial = False

    # --- Certifiers ---------------------------------------------------------

    @staticmethod
    def _method_kind(method: str, line: Optional[int] = None) -> str:
        normalized = method.strip().lower()
        if normalized in _MEAN_METHODS:
            return normalized
        if normalized in _BINOMIAL_METHODS:
            return _BINOMIAL_METHODS[normalized]
        raise RuntimeErrorLumos(f"Unknown certifier '{method}' (supported: {_SUPPORTED})", line)

    def certify(self, method: str, one_minus_delta: float, observations: List[Any]) -> CertificationResult:
        """certify(cert, 1 - delta, v) over one observation per run."""
        if not observations:
            raise RuntimeErrorLumos("No observations to certify")
        kind = self._method_kind(method)
        delta = 1.0 - one_minus_delta
        if kind in _MEAN_METHODS:
            mode, values = self._mean_observations(observations, method)
            return self._certify_mean(kind, method, one_minus_delta, values, mode)

        outcomes = [self._binary_observation(o, method) for o in observations]
        successes, trials = sum(outcomes), len(outcomes)
        if kind == "beta":
            lower, upper = clopper_pearson_interval(successes, trials, delta)
        else:
            try:
                from statsmodels.stats.proportion import proportion_confint
            except ImportError as exc:
                raise RuntimeErrorLumos(
                    f"The '{method}' certifier needs statsmodels (pip install statsmodels); "
                    "Clopper-Pearson and Bernstein work without it"
                ) from exc
            lower, upper = proportion_confint(successes, trials, alpha=delta, method=kind)
        return CertificationResult(
            successes=successes,
            trials=trials,
            empirical_prob=successes / trials,
            lower_ci=float(lower),
            upper_ci=float(upper),
            method=method,
            one_minus_delta=one_minus_delta,
            observation_mode="binary",
            observations=outcomes,
        )

    @staticmethod
    def _binary_observation(value: Any, method: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            hint = "; compare the score to get one, e.g. toEstimate(J(r) = 1)"
        elif isinstance(value, (OrderedSet, list, tuple)):
            hint = "; lists are only accepted by Bernstein and Hoeffding, as one {lb, ub} pair per run"
        else:
            hint = ""
        raise RuntimeErrorLumos(
            f"{method} needs one boolean observation per run, got a {type_name(value)}{hint}"
        )

    @staticmethod
    def _unit(value: Any, what: str) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise RuntimeErrorLumos(f"{what} must be a number in [0, 1], got {value!r}")
        return float(value)

    def _check_observation(self, kind: str, method: str, value: Any, mean_mode: List[Optional[str]]) -> Any:
        """Validate one run's observation; mean_mode[0] remembers the mode so far."""
        if kind not in _MEAN_METHODS:
            return self._binary_observation(value, method)
        mode, item = self._mean_item(value, method)
        if mean_mode[0] is not None and mode != mean_mode[0]:
            raise RuntimeErrorLumos(f"{method} observations mix single values and {{lb, ub}} pairs across runs")
        mean_mode[0] = mode
        return item

    def _mean_item(self, value: Any, method: str) -> Tuple[str, Any]:
        """('real', x) for a real in [0, 1] or a boolean, ('pair', (lb, ub)) for a pair."""
        if isinstance(value, (OrderedSet, list, tuple)):
            items = list(value)
            if len(items) != 2:
                hint = " (a set {lb, ub} with lb = ub has one element; write toEstimate({lb, ub}))" \
                    if len(items) == 1 else ""
                raise RuntimeErrorLumos(
                    f"{method} list observations must be one {{lb, ub}} pair per run, got {len(items)} values{hint}"
                )
            lb = self._unit(items[0], f"{method} lower bound")
            ub = self._unit(items[1], f"{method} upper bound")
            if lb > ub:
                raise RuntimeErrorLumos(f"{method} pair must satisfy lb <= ub, got {{{lb}, {ub}}}")
            return "pair", (lb, ub)
        return "real", self._unit(value, f"{method} observation")

    def _mean_observations(self, observations: List[Any], method: str) -> Tuple[str, List[Any]]:
        """One real in [0, 1] (or boolean) per run, or one {lb, ub} pair per run."""
        mode: List[Optional[str]] = [None]
        values = [self._check_observation(self._method_kind(method), method, value, mode) for value in observations]
        return mode[0], values

    @staticmethod
    def _hoeffding_radius(n: int, delta: float) -> float:
        """t with P(|mean - mu| >= t) <= delta for n IID values in [0, 1]: 2 exp(-2 n t^2) = delta."""
        return math.sqrt(math.log(2.0 / delta) / (2.0 * n))

    @staticmethod
    def _empirical_bernstein_interval(samples: List[float], delta: float) -> Tuple[float, float]:
        n = len(samples)
        mean = sum(samples) / n
        if n == 1:
            radius = math.sqrt(math.log(2.0 / delta) / 2.0)
        else:
            variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
            log_term = math.log(3.0 / delta)
            radius = math.sqrt((2.0 * variance * log_term) / n) + (3.0 * log_term) / n
        return max(0.0, mean - radius), min(1.0, mean + radius)

    def _certify_mean(self, kind: str, method: str, one_minus_delta: float, values: List[Any], mode: str) -> CertificationResult:
        """Bernstein or Hoeffding bounds on the mean of one value, or one {lb, ub} pair, per run."""
        delta = 1.0 - one_minus_delta
        n = len(values)
        if mode == "real":
            lows = highs = values
            empirical = sum(values) / n
        else:
            lows, highs = [lb for lb, _ in values], [ub for _, ub in values]
            empirical = sum((lb + ub) / 2.0 for lb, ub in values) / n
        if kind == "hoeffding":
            # Two-sided at delta. For pairs: each one-sided bound at delta/2 (union bound) has
            # radius sqrt(ln(2/delta) / 2n), the same as the two-sided radius.
            radius = self._hoeffding_radius(n, delta)
            lower_ci = max(0.0, sum(lows) / n - radius)
            upper_ci = min(1.0, sum(highs) / n + radius)
        elif mode == "real":
            lower_ci, upper_ci = self._empirical_bernstein_interval(values, delta)
        else:
            # Lower bound from the lbs, upper bound from the ubs, delta/2 each (union bound).
            lower_ci, _ = self._empirical_bernstein_interval(lows, delta / 2.0)
            _, upper_ci = self._empirical_bernstein_interval(highs, delta / 2.0)
        return CertificationResult(
            successes=int(round(empirical * n)),
            trials=n,
            empirical_prob=empirical,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            method=method,
            one_minus_delta=one_minus_delta,
            observation_mode=mode,
            observations=values,
        )
