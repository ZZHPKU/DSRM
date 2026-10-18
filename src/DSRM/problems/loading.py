"""Load a user-supplied factory without a central registry."""
from importlib import import_module

from .base import PDEProblem


def load_problem(factory: str, kwargs: dict) -> PDEProblem:
    module, separator, name = factory.partition(":")
    if not separator or not module or not name:
        raise ValueError("problem.factory must be 'importable.module:function'")
    problem = getattr(import_module(module), name)(**kwargs)
    if not isinstance(problem, PDEProblem):
        raise TypeError("Problem factory must return a PDEProblem")
    return problem
