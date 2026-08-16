"""Tests for the cache-size ablation (X9b).

This package carries an ``__init__.py`` on purpose: ``tests/ablation_study/``
has no package marker and pytest runs in the default ``prepend`` import mode, so
two test modules sharing a basename would collide at collection time. The
package marker plus the ``cache_size_`` filename prefix keep this directory's
modules distinct from ``tests/ablation_study/test_*.py``.
"""
