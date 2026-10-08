This directory preserves the legacy repository-root marker used by
`exp.offline_search.closed_loop.ops.emit_arms`. R16's resident entry point
imports that module through `prepare.py`, even when it only reads an existing
registry. Runtime YAMLs remain supplied explicitly; no main-tree configuration
or running experiment is modified.
