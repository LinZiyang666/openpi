"""Static Python 3.8 grammar/import/API inventory; does not import LIBERO."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

from .build_client_bundle import MODULES, HERE
ROOT = HERE.parents[4]
STOCK = ["examples/libero/episode_runner.py", "examples/libero/worker_entry.py",
         "examples/libero/main.py", "examples/libero/collect_util.py",
         "exp/gate_threshold_pareto/run_gtp.py", "exp/gate_threshold_pareto/emit_gtp_yamls.py",
         "exp/gate_threshold_pareto/libraries.py", "exp/ablation_study/cache_size/run_size_eval.py",
         "exp/offline_search/closed_loop/ops/remote/run_gtp_subset.py",
         "src/openpi/cache/config.py", "src/openpi/cache/types.py",
         "packages/openpi-client/src/openpi_client/websocket_client_policy.py",
         "packages/openpi-client/src/openpi_client/base_policy.py",
         "packages/openpi-client/src/openpi_client/msgpack_numpy.py",
         "packages/openpi-client/src/openpi_client/image_tools.py"]


def inspect_file(path, owned):
    src = path.read_text()
    tree = ast.parse(src, filename=str(path), feature_version=(3, 8))
    imports = sorted(set(("."*n.level+(n.module or "")+":"+",".join(a.name for a in n.names))
                         if isinstance(n, ast.ImportFrom) else ",".join(a.name for a in n.names)
                         for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))))
    issues = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = ast.unparse(n.func)
            if name == "zip" and any(k.arg == "strict" for k in n.keywords):
                issues.append(dict(line=n.lineno,api="zip(strict=)",resolution="client_compat module-local driver backport"))
            if name.endswith((".removeprefix", ".removesuffix", ".is_relative_to")):
                issues.append(dict(line=n.lineno,api=name,resolution="needs path audit"))
    postponed = any(isinstance(n, ast.ImportFrom) and n.module=="__future__" and any(a.name=="annotations" for a in n.names) for n in tree.body)
    if not postponed:
        annotations = [n.annotation for n in ast.walk(tree) if isinstance(n, (ast.AnnAssign, ast.arg)) and n.annotation is not None]
        annotations += [n.returns for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.returns is not None]
        for anno in annotations:
            if any((isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in {"list","tuple","dict","set","frozenset","type"})
                   or (isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr)) for n in ast.walk(anno)):
                issues.append(dict(line=anno.lineno,api="PEP585/604 evaluated annotation", resolution="client_compat postponed-annotation source loader on Python 3.8"))
    return dict(path=str(path.relative_to(ROOT)),owned_client=owned,grammar38=True,
                deferred_annotations=postponed,imports=imports,api_notes=issues,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


BASES = [ROOT, ROOT/"src", ROOT/"packages/openpi-client/src"]


def module_path(name):
    for base in BASES:
        file = base.joinpath(*name.split("."))
        for candidate in (file.with_suffix(".py"), file/"__init__.py"):
            if candidate.is_file():
                return candidate


def module_name(path):
    for base in reversed(BASES):
        try:
            parts = path.relative_to(base).with_suffix("").parts
            return ".".join(parts[:-1] if parts[-1]=="__init__" else parts)
        except ValueError:
            pass


def import_nodes(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id=="TYPE_CHECKING":
        return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from import_nodes(child)


def closure(paths):
    seen, pending = set(), list(paths)
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        name = module_name(path)
        package = name if path.name=="__init__.py" else name.rsplit(".", 1)[0]
        names = [".".join(name.split(".")[:i]) for i in range(1,len(name.split(".")))]
        for n in import_nodes(ast.parse(path.read_text())):
            if isinstance(n, ast.Import):
                names += [a.name for a in n.names]
            else:
                prefix = ".".join(package.split(".")[:len(package.split("."))-n.level+1]) if n.level else ""
                base = ".".join(x for x in (prefix,n.module) if x)
                names += [base] + [base+"."+a.name for a in n.names]
        for mod in names:
            candidate = module_path(mod)
            if candidate is not None and candidate not in seen:
                pending.append(candidate)
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path, help="readable complete source/import inventory")
    a = ap.parse_args()
    owned = [HERE/(x+".py") for x in MODULES] + [HERE/"install_client_payload.py"]
    stock = [ROOT/x for x in STOCK] + sorted((ROOT/"src/openpi/conductor").glob("*.py"))
    rows = [inspect_file(x, x in owned) for x in closure(owned+stock)]
    report = dict(local_python=sys.version,python38_on_path=shutil.which("python3.8"),
                  verification="AST feature_version=(3,8), not execution under Python 3.8", files=rows,
                  remote_source_equality="unverified; existing island dependencies not bundled or replaced",
                  runtime_typing="owned client has no PEP585/604 annotations; stock non-postponed annotations handled by opt-in source loader; deferred optional function imports listed but not recursively followed",
                  numpy="owned APIs: ndarray/asarray/array/generic/savez_compressed/random.get_state; bit_generator accessed only when present; no np.float/np.int/np.bool aliases")
    a.out.write_text(json.dumps(report, indent=2)+"\n")
    if a.markdown is not None:
        lines = ["# Client Python 3.8 import inventory", "",
                 "Generated by `audit_client_compat.py`; {} local repository files pass the Python 3.8 AST grammar check.".format(len(rows)),
                 "This command is a static audit. The separate real Python 3.8 compile/import check is recorded in `results/python38.json`; remote source-byte equality remains unverified.",
                 "The {} owned Python files are {} bundled modules and the temporary installer.".format(len(owned), len(MODULES)),
                 "Imports below include lazy imports; the recursively followed closure is module-level repository imports plus the explicitly listed stock entry dependencies.",
                 "Third-party implementations and optional function-only experiment dependencies are not recursively audited.",
                 "Runtime/API assumptions and compatibility fixes are in [CLIENT_DEPLOY.md](CLIENT_DEPLOY.md).", "",
                 "| Local source | Owned client | Postponed annotations in source | Imports (including lazy) | API notes |",
                 "|---|---|---|---|---|"]
        for row in rows:
            imports = "<br>".join("`"+x.replace("|", "&#124;")+"`" for x in row["imports"]) or "—"
            notes = "<br>".join("L{}: {} ({})".format(x["line"], x["api"], x["resolution"]) for x in row["api_notes"]) or "—"
            lines.append("| `{}` | {} | {} | {} | {} |".format(row["path"], row["owned_client"], row["deferred_annotations"], imports, notes))
        lines.extend(["", "Exact audited source SHA256s are recorded in `results/client_compat.json`.", ""])
        a.markdown.write_text("\n".join(lines))
    print(json.dumps(dict(files=len(rows),owned=len(owned),grammar38=True,python38=report['python38_on_path'])))


if __name__ == "__main__":
    main()
