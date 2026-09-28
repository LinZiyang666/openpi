"""Python 3.8 compatibility confined to the opt-in client process."""
import builtins
import __future__
import importlib.abc
import importlib.machinery
import itertools
import sys


class DeferredSourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Do not reuse a pyc compiled without postponed annotations.
        path = self.get_filename(fullname)
        return compile(self.get_data(path), path, "exec",
                       flags=__future__.annotations.compiler_flag, dont_inherit=True)


class DeferredClientImports(importlib.abc.MetaPathFinder):
    _p3_python38 = True

    def find_spec(self, fullname, path=None, target=None):
        if not any(fullname == prefix or fullname.startswith(prefix+".") for prefix in (
                "openpi.cache", "openpi.conductor", "examples.libero",
                "exp.gate_threshold_pareto", "exp.ablation_study.cache_size")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            spec.loader = DeferredSourceLoader(fullname, spec.origin)
        return spec


def install_import_compat():
    # The local stock cache.types/backend modules contain evaluated list[T] /
    # frozenset[T] annotations without a future import. Postpone annotations in
    # this client process only; never rewrite src or an island dependency.
    if sys.version_info < (3, 9) and not any(getattr(x, "_p3_python38", False) for x in sys.meta_path):
        sys.meta_path.insert(0, DeferredClientImports())


def zip_compat(*iterables, **kwargs):
    strict = kwargs.pop("strict", False)
    if kwargs:
        raise TypeError("unexpected zip keyword")
    if not strict:
        return builtins.zip(*iterables)
    def checked():
        sentinel = object()
        for row in itertools.zip_longest(*iterables, fillvalue=sentinel):
            if any(x is sentinel for x in row):
                raise ValueError("zip() arguments have different lengths")
            yield row
    return checked()


def install_driver_compat():
    # Existing driver sharding uses zip(strict=True), introduced in Python 3.10.
    # Module-local bindings leave builtins, files and other processes untouched.
    if sys.version_info < (3, 10):
        from openpi.conductor import sharding
        from openpi.cache import config
        from openpi.cache import orchestrator
        sharding.zip = config.zip = orchestrator.zip = zip_compat
