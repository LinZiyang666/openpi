"""Single owner-controlled preparation switch; absent/unrecognised means OFF.

Frozen payloads retain their prepared behavior. Only new prepares read this file.
"""
from pathlib import Path
SWITCH = Path(__file__).with_name('ENABLED')
def enabled():
    try:return SWITCH.read_text().strip().lower() == 'on'
    except OSError:return False


def source(path):
    """Native bytes for NEW archives when OFF; never read or alter frozen runs."""
    path = Path(path)
    root = Path(__file__).resolve().parents[5]
    try:name = str(path.resolve().relative_to(root))
    except ValueError:return path.read_bytes()
    native = Path(__file__).with_name('native_sources') / name.replace('/', '__')
    if not enabled() and native.is_file():return native.read_bytes()
    return path.read_bytes()


def function_source(function):
    """Use the same policy for by-value RPC hooks as for archive members."""
    import ast
    import inspect
    path = inspect.getsourcefile(function)
    text = source(path).decode()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function.__name__:
            return ast.get_source_segment(text, node)+'\n'
    raise ValueError('preparation function missing from selected source: '+function.__name__)
