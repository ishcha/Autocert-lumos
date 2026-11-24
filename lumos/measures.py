"""Measure functions for discrete probability distributions.

Measure functions assign non-negative weights to elements. Weights are normalized
to form a probability distribution for sampling. Per the paper: meas(ω) ≥ 0,
P(ω) = meas(ω) / Σ meas(ω').

This module exposes a small registry-based framework that lets developers plug in
new heuristics for sampling arbitrary Python values (nodes, prompts, numbers, etc.).
"""

import math
import random
from typing import List, Any, Callable, Dict, TypeVar

from .errors import RuntimeErrorLumos, UndefinedValue

# A measure function scores a homogeneous list of elements and returns one
# non-negative weight per element.
MeasureFn = Callable[..., List[float]]


def uniform(elements: List[Any], rng: random.Random) -> List[float]:
    """Uniform measure: equal weight for all elements.
    
    Returns weight 1.0 for each element → uniform distribution.
    """
    if not elements:
        raise RuntimeErrorLumos("Cannot compute measure over empty set")
    return [1.0] * len(elements)


def constant(elements: List[Any], rng: random.Random, weight: float = 1.0) -> List[float]:
    """Constant measure: same weight for all elements (configurable).
    
    Equivalent to uniform when weight=1.0.
    """
    if not elements:
        raise RuntimeErrorLumos("Cannot compute measure over empty set")
    if weight < 0:
        raise RuntimeErrorLumos(f"Measure weight must be non-negative, got {weight}")
    return [weight] * len(elements)


def one_hot(elements: List[Any], rng: random.Random, index: int = 0) -> List[float]:
    """One-hot measure: weight 1 for element at index, 0 elsewhere.
    
    Deterministic selection of a single element.
    """
    if not elements:
        raise RuntimeErrorLumos("Cannot compute measure over empty set")
    if index < 0 or index >= len(elements):
        raise RuntimeErrorLumos(f"Index {index} out of range for {len(elements)} elements")
    weights = [0.0] * len(elements)
    weights[index] = 1.0
    return weights


def geometric(elements: List[Any], rng: random.Random, p: float = 0.5) -> List[float]:
    """Geometric decay: weight decreases exponentially by index.
    
    weight[i] = (1-p)^i. Earlier elements are more likely.
    """
    if not elements:
        raise RuntimeErrorLumos("Cannot compute measure over empty set")
    if not 0 < p < 1:
        raise RuntimeErrorLumos(f"Geometric p must be in (0,1), got {p}")
    return [(1 - p) ** i for i in range(len(elements))]


def reciprocal(elements: List[Any], rng: random.Random) -> List[float]:
    """Reciprocal measure: weight[i] = 1/(i+1).
    
    Earlier elements are more likely; harmonic decay.
    """
    if not elements:
        raise RuntimeErrorLumos("Cannot compute measure over empty set")
    return [1.0 / (i + 1) for i in range(len(elements))]


# Registry of named measures for sample(name(S)). `meas` defaults to uniform;
# a program environment that binds `meas` to a callable overrides it.
MEASURE_REGISTRY: Dict[str, MeasureFn] = {
    "uniform": uniform,
    "meas": uniform,  # Paper alias: meas defaults to uniform
    "constant": constant,
    "one_hot": one_hot,
    "geometric": geometric,
    "reciprocal": reciprocal,
}


def get_measure(name: str) -> MeasureFn:
    """Get a registered measure by name; unknown names are an error."""
    try:
        return MEASURE_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(MEASURE_REGISTRY))
        raise RuntimeErrorLumos(f"Unknown measure '{name}' (registered: {known})") from None


def register_measure(name: str, fn: MeasureFn | None = None):
    """Register a new measure.

    Can be used as:

        register_measure("my_measure", fn)

    or as a decorator:

        @register_measure("my_measure")
        def my_measure(elements, rng, ...):
            ...
    """
    if fn is None:
        def decorator(f: MeasureFn) -> MeasureFn:
            MEASURE_REGISTRY[name] = f
            return f
        return decorator
    MEASURE_REGISTRY[name] = fn
    return fn


def choose_weighted(elements: List[Any], weights: List[float], rng: random.Random,
                    measure_name: str = "meas") -> Any:
    """Pick one element with probability proportional to its weight.

    Following the paper, all-zero weights mean the uniform distribution.
    """
    if len(weights) != len(elements):
        raise RuntimeErrorLumos(
            f"Measure '{measure_name}' returned {len(weights)} weights for "
            f"{len(elements)} elements"
        )
    for w in weights:
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
            raise RuntimeErrorLumos(
                f"Measure '{measure_name}' must return finite non-negative numbers, got {w!r}"
            )
    largest = max(weights)
    weights = [1.0] * len(elements) if largest == 0 else [w / largest for w in weights]
    return rng.choices(elements, weights=weights, k=1)[0]


def sample_from_measure(
    elements: List[Any],
    measure_name: str,
    rng: random.Random,
    **measure_kwargs,
) -> Any:
    """sample(meas(S)) with a registered measure. sample(∅) is ⊥."""
    if not elements:
        raise UndefinedValue(f"sample({measure_name}(S)) with S empty")
    weights = get_measure(measure_name)(elements, rng, **measure_kwargs)
    return choose_weighted(elements, list(weights), rng, measure_name)
