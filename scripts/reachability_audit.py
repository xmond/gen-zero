#!/usr/bin/env python3
"""Conservative source-level reachability inventory; never authorizes deletion alone.

Python import edges and Rust module declarations are explicit. Dynamic dispatch,
reflection, generated bindings and external callers remain unresolved by design.
"""
import ast
import json
import re
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/results/reachability.json'
PYROOT = ROOT / 'python'
RUSTROOT = ROOT / 'crates'


def python_module(path):
    parts = list(path.relative_to(PYROOT).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def main():
    files = sorted((PYROOT / 'gen_zero').rglob('*.py'))
    files = [p for p in files if 'tests' not in p.parts]
    modules = {python_module(p): p for p in files}
    edges = {m: set() for m in modules}
    definitions = []
    parse_errors = []
    for module, path in modules.items():
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            parse_errors.append(f'{path.relative_to(ROOT)}: {exc}')
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.append({'symbol': f'{module}.{node.name}', 'file': str(path.relative_to(ROOT)), 'line': node.lineno, 'status': 'unresolved_symbol_dispatch'})
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                package = module if path.name == '__init__.py' else module.rpartition('.')[0]
                base = '.'.join(package.split('.')[:max(0, len(package.split('.')) - node.level + 1)]) if node.level else ''
                prefix = f'{base}.{node.module}' if base and node.module else (base or node.module or '')
                names = [prefix] + [f'{prefix}.{alias.name}' for alias in node.names]
            else:
                continue
            for name in names:
                # Importing a.b.c also executes a/__init__.py and a/b/__init__.py,
                # so every ancestor package that exists as a module is reachable too,
                # not just the longest (most specific) match.
                while name:
                    if name in modules and name != module:
                        edges[module].add(name)
                    name = name.rpartition('.')[0]
    roots = {'gen_zero', 'gen_zero.__main__', 'gen_zero.cli', 'gen_zero.client', 'gen_zero.service.app', 'gen_zero.mcp.server', 'gen_zero.mcp.sse_transport'} & modules.keys()
    reached = set(roots)
    queue = deque(sorted(roots))
    while queue:
        for dst in sorted(edges[queue.popleft()] - reached):
            reached.add(dst)
            queue.append(dst)
    rust = []
    rust_roots = []
    rust_edges = {}
    for path in sorted(RUSTROOT.glob('*/src/**/*.rs')):
        rel = str(path.relative_to(ROOT))
        source = path.read_text(encoding='utf-8')
        declared = sorted(set(re.findall(r'(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;', source)))
        rust.append({'file': rel, 'declared_modules': declared, 'public_symbols': len(re.findall(r'(?m)^\s*pub(?:\([^)]*\))?\s+(?:async\s+)?(?:fn|struct|enum|trait)\s+\w+', source))})
        children = []
        # mod.rs/lib.rs/main.rs look for submodules beside themselves; any other
        # file `foo.rs` (Rust 2018 style) looks in a sibling `foo/` directory
        # (e.g. mount.rs's `mod tests;` resolves to mount/tests.rs, not src/tests.rs).
        mod_base = path.parent if path.stem in ('mod', 'lib', 'main') else path.parent / path.stem
        for name in declared:
            candidates = (mod_base / f'{name}.rs', mod_base / name / 'mod.rs')
            target = next((p for p in candidates if p.is_file()), None)
            if target is not None:
                children.append(str(target.relative_to(ROOT)))
        rust_edges[rel] = children
        if path.name in ('main.rs', 'lib.rs'):
            rust_roots.append(rel)
    if parse_errors:
        raise SystemExit('Python parse errors: ' + '; '.join(parse_errors))
    rust_reached = set(rust_roots)
    queue = deque(rust_roots)
    while queue:
        for dst in rust_edges[queue.popleft()]:
            if dst not in rust_reached:
                rust_reached.add(dst)
                queue.append(dst)
    report = {
        'schema': 1,
        'method': 'Python AST import reachability from production entry modules; Rust declared module reachability from crate entry files',
        'limitations': ['Python dynamic imports, reflection, attribute calls and external SDK callers are unresolved', 'Rust function calls, cross-crate edges, path attributes and macro expansions are unresolved', 'Unreached files and symbols are NOT deletion proof'],
        'entry_modules': sorted(roots), 'python_import_edges': [{'from': k, 'to': v} for k in sorted(edges) for v in sorted(edges[k])],
        'python_reached_modules': sorted(reached),
        'python_unreached_modules_requires_review': sorted(modules.keys() - reached),
        'python_symbols': definitions,
        'rust_entry_files': rust_roots, 'rust_module_edges': [{'from': k, 'to': v} for k in sorted(rust_edges) for v in rust_edges[k]],
        'rust_reached_files': sorted(rust_reached), 'rust_unreached_files_requires_review': sorted(set(rust_edges) - rust_reached), 'rust_files': rust,
        'deletion_evidence': [],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'{OUT.relative_to(ROOT)}: {len(reached)}/{len(modules)} Python modules import-reachable; {len(rust)} Rust files inventoried; 0 deletion claims')


if __name__ == '__main__':
    main()
