"""Beaver verification bridge for Lumos.

Wraps beaver.run() as Lumos-injectable callables.
Inject into Lumos as:
    interpreter.env.define("beaver_ub", bridge.certify)   # returns UB
    interpreter.env.define("beaver_lb", bridge.get_lb)    # returns LB from last call
"""

import sys
from pathlib import Path

# Make beaver importable
_BEAVER_ROOT = Path(__file__).resolve().parent.parent.parent / "Beaver"
if str(_BEAVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_BEAVER_ROOT))

try:
    import beaver
except Exception as exc:  # missing, or a broken install (e.g. an editable install in someone else's home)
    raise ImportError(
        f"Beaver is not importable ({type(exc).__name__}: {exc}). Install it with "
        f"`pip install -e path/to/Beaver`, or put the Beaver repository at {_BEAVER_ROOT}."
    ) from exc


class BeaverBridge:
    """Wraps beaver.run() as two Lumos callables: certify() and get_lb().

    certify(prompt) runs Beaver on a single prompt and returns the upper bound.
    get_lb() returns the lower bound from the most recent certify() call.

    Usage from Lumos:
        bounds := Beaver(full_prompt)   # [lb, ub]
        ub := beaver_ub(full_prompt)    # legacy helper
        lb := beaver_lb()               # legacy helper
    """

    def __init__(
        self,
        model,
        constraint_fn,
        check_call_fn=None,
        server_addr=None,
        auto_server=False,
        gen_length=32,
        temperature=1.0,
        top_p=0.99,
        max_iterations=100,
        epsilon=0.01,
        num_logprobs=100,
        max_workers=1,
        verifier="frontier",
        use_chat_template=True,
        log_dir="lumos_beaver_logs",
    ):
        self.model = model
        self.constraint_fn = constraint_fn
        self.check_call_fn = check_call_fn
        self.server_addr = server_addr
        self.auto_server = auto_server
        self.gen_length = gen_length
        self.temperature = temperature
        self.top_p = top_p
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.num_logprobs = num_logprobs
        self.max_workers = max_workers
        self.verifier = verifier
        self.use_chat_template = use_chat_template
        self.log_dir = log_dir
        self._last_lb = 0.0
        self._last_ub = 1.0
        self._call_count = 0
        # Bridge keeps run/log state and should be shared if injected directly.
        self.__lumos_shared_callable__ = True

    def certify(self, prompt):
        """Run beaver.run() on a single prompt. Returns upper bound (float)."""
        self._run(prompt)
        return self._last_ub

    def beaver(self, prompt):
        """Run beaver.run() and return the pair [lower_bound, upper_bound].

        A list, not an OrderedSet: a set would collapse the pair when lb == ub.
        """
        self._run(prompt)
        return [self._last_lb, self._last_ub]

    def _run(self, prompt):
        self._call_count += 1
        kwargs = dict(
            prompts=[{"prompt": prompt}],
            constraint_fn=self.constraint_fn,
            model=self.model,
            auto_server=self.auto_server,
            gen_length=self.gen_length,
            temperature=self.temperature,
            top_p=self.top_p,
            max_iterations=self.max_iterations,
            epsilon=self.epsilon,
            num_logprobs=self.num_logprobs,
            max_workers=self.max_workers,
            verifier=self.verifier,
            use_chat_template=self.use_chat_template,
            log_dir=f"{self.log_dir}/call_{self._call_count}",
        )
        if self.check_call_fn:
            kwargs["check_call_fn"] = self.check_call_fn
        if self.server_addr:
            kwargs["server_addr"] = self.server_addr
            kwargs["auto_server"] = False

        results = beaver.run(**kwargs)
        r = results[0]
        if r.get("error") or "lower_bound" not in r or "upper_bound" not in r:
            raise RuntimeError(f"Beaver did not return bounds: {r.get('error') or sorted(r)}")
        self._last_lb = float(r["lower_bound"])
        self._last_ub = float(r["upper_bound"])

    def get_lb(self):
        """Return lower bound from the most recent certify() call."""
        return self._last_lb
