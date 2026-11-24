"""Environment for variable bindings in Lumos DSL"""

import copy
from typing import Dict, Any
from .errors import RuntimeErrorLumos


class Environment:
    """Environment for storing variable bindings"""
    
    def __init__(self):
        self.bindings: Dict[str, Any] = {}
    
    def define(self, name: str, value: Any):
        """Define a new variable (error if already exists)"""
        if name in self.bindings:
            raise RuntimeErrorLumos(f"Variable '{name}' is already defined")
        self.bindings[name] = value
    
    def assign(self, name: str, value: Any):
        """Assign to an existing variable (error if undefined)"""
        if name not in self.bindings:
            raise RuntimeErrorLumos(f"Undefined variable '{name}'")
        self.bindings[name] = value
    
    def set(self, name: str, value: Any):
        """Bind or rebind a variable (the semantics of `x := e`)."""
        self.bindings[name] = value

    def get(self, name: str) -> Any:
        """Get the value of a variable (error if undefined)"""
        if name not in self.bindings:
            raise RuntimeErrorLumos(f"Undefined variable '{name}'")
        return self.bindings[name]
    
    def copy(self) -> 'Environment':
        """Create a copy of this environment"""
        new_env = Environment()
        new_env.bindings = self.bindings.copy()
        return new_env

    def copy_for_trial(self, trial_seed: int | None = None) -> 'Environment':
        """The state a run starts from: data values are deep-copied, functions shared.

        External functions (models, judges, tools, API clients, transformers
        pipelines) are shared by every run, as the paper requires them to be
        stateless. An external that needs fresh state per run can define
        lumos_clone_for_trial(seed), which is called for each run instead.
        """
        new_env = Environment()
        for name, value in self.bindings.items():
            new_env.bindings[name] = self._clone_binding_for_trial(name, value, trial_seed)
        return new_env

    @staticmethod
    def _clone_binding_for_trial(name: str, value: Any, trial_seed: int | None):
        clone_hook = getattr(value, "lumos_clone_for_trial", None)
        if callable(clone_hook):
            try:
                return clone_hook(trial_seed)
            except Exception as exc:
                raise RuntimeErrorLumos(
                    f"lumos_clone_for_trial of '{name}' failed: {type(exc).__name__}: {exc}"
                ) from exc
        if callable(value):
            return value
        try:
            return copy.deepcopy(value)
        except Exception as exc:
            raise RuntimeErrorLumos(
                f"Binding '{name}' could not be copied for a new run"
            ) from exc

    def __str__(self) -> str:
        """String representation of the environment"""
        return f"Environment({dict(self.bindings)})"
