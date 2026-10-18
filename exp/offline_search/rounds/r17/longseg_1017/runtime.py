"""Install the pinned segment-boundary fix in a new resident process only."""
import inspect
import textwrap
import ast
from pathlib import Path


def source():
    import importlib
    result=[]
    for module,name,methods in (
        ('exp.r17_integration.common.serving','ConnectionPolicy',('_reset','_request','acknowledge_execution','infer')),
        ('exp.r17_integration.search.cache_recipe','CacheStudyRecipe',('reset','on_execution','query')),
        ('exp.r17_integration.common.websocket_policy_server','_ControlSocket',('send',))):
        body=Path(importlib.import_module(module).__file__).read_text();tree=ast.parse(body)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
        code='\n\n'.join(textwrap.dedent(ast.get_source_segment(body,n)) for n in cls.body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in methods)+'\n'
        result.append(dict(module=module,name=name,methods=methods,body=code))
    return result


def installer(body):
    return ('import importlib as _ls_import,linecache as _ls_lines\n'
            'for _ls_patch in '+repr(body)+':\n'
            '    _ls_module=_ls_import.import_module(_ls_patch["module"])\n'
            '    _ls_body=_ls_patch["body"];_ls_file="longseg_1017_"+_ls_patch["name"]\n'
            '    _ls_lines.cache[_ls_file]=(len(_ls_body),None,_ls_body.splitlines(True),_ls_file)\n'
            '    _ls_ns=dict(vars(_ls_module));exec(compile(_ls_body,_ls_file,"exec"),_ls_ns)\n'
            '    for _ls_name in _ls_patch["methods"]:\n'
            '        setattr(getattr(_ls_module,_ls_patch["name"]),_ls_name,_ls_ns[_ls_name])\n')
