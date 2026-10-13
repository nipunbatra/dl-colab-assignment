"""Static student-release checks; never execute or supply assessed answers."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
STUBS = {'clip_symmetric_loss', 'semantic_search', 'infer', 'cbm_head', 'intervene'}


def check(root):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)

    notebook_path = root / 'assignment3.ipynb'
    nb = json.loads(notebook_path.read_text())
    require(nb.get('nbformat') == 4 and nb.get('nbformat_minor') == 5, 'Expected notebook format 4.5')
    require(isinstance(nb.get('metadata'), dict), 'Notebook metadata must be an object')
    require(isinstance(nb.get('cells'), list) and bool(nb['cells']), 'Notebook must have cells')
    ids = set()
    found = set()
    for index, cell in enumerate(nb.get('cells', [])):
        kind = cell.get('cell_type')
        require(kind in {'markdown', 'code'}, f'Cell {index}: unsupported type')
        require(isinstance(cell.get('metadata'), dict), f'Cell {index}: missing metadata')
        cell_id = cell.get('id', '')
        require(bool(re.fullmatch(r'[A-Za-z0-9_-]{1,64}', cell_id)) and cell_id not in ids,
                f'Cell {index}: invalid or duplicate id')
        ids.add(cell_id)
        source = cell.get('source', '')
        require(isinstance(source, str) or
                (isinstance(source, list) and all(isinstance(s, str) for s in source)),
                f'Cell {index}: invalid source')
        source = ''.join(source)
        if kind != 'code':
            continue
        require(cell.get('outputs') == [], f'Cell {index}: saved outputs')
        require('execution_count' in cell and cell['execution_count'] is None,
                f'Cell {index}: saved/missing execution count')
        # No installations or model calls execute during this check. Ignore IPython magic lines.
        lines = source.splitlines()
        if lines and lines[0].lstrip().startswith('%%'):
            continue
        clean = '\n'.join(line if not line.lstrip().startswith(('%', '!')) else '' for line in lines)
        try:
            tree = ast.parse(clean, filename=f'cell-{index}')
        except SyntaxError as exc:
            errors.append(f'Cell {index}: {exc}')
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in STUBS:
                found.add(node.name)
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    body = body[1:]
                require(len(body) == 1 and isinstance(body[0], ast.Raise) and
                        isinstance(body[0].exc, ast.Call) and
                        isinstance(body[0].exc.func, ast.Name) and
                        body[0].exc.func.id == 'NotImplementedError',
                        f'{node.name}: release must contain only a NotImplementedError stub')
    require(found == STUBS, f'Missing assessed stubs: {sorted(STUBS - found)}')

    # Use nbformat's full schema validator when available; structural checks above are always run.
    try:
        import nbformat
    except ImportError:
        errors.append('Install nbformat to run full notebook schema validation')
    else:
        try:
            nbformat.validate(nbformat.from_dict(nb))
        except Exception as exc:
            errors.append(f'Notebook schema: {exc}')

    from build_notebook import build
    require(nb == build(), 'Notebook differs from generator; run scripts/build_notebook.py')
    for relative in ('scripts/build_notebook.py', 'scripts/check_student_release.py', 'support.py'):
        try:
            ast.parse((root / relative).read_text(), filename=relative)
        except SyntaxError as exc:
            errors.append(str(exc))

    checksums = json.loads((root / 'data/checksums.json').read_text())
    data_root = (root / 'data').resolve()
    for relative, expected in checksums.items():
        path = (data_root / relative).resolve()
        require(path.is_relative_to(data_root), f'Unsafe dataset path: {relative}')
        if not path.is_relative_to(data_root):
            continue
        require(path.is_file(), f'Missing dataset: {relative}')
        if path.is_file():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            require(actual == expected, f'Dataset checksum mismatch: {relative}')
    manifest = json.loads((data_root / 'gallery/manifest.json').read_text())
    require(len(manifest) == 17, 'Gallery must contain 17 images')
    for record in manifest:
        require(checksums.get('gallery/' + record['file']) == record['sha256'],
                f"Gallery manifest checksum mismatch: {record['file']}")
    cbm = json.loads((data_root / 'cbm.json').read_text())
    for record in cbm['examples']:
        require(checksums.get(record['file']) == record['sha256'],
                f"Concept manifest checksum mismatch: {record['file']}")
    try:
        import numpy as np
        with np.load(data_root / 'digits.npz', allow_pickle=False) as digits:
            for split, count in [('train', 200), ('val', 50), ('test', 100)]:
                require(digits[f'{split}_images'].shape == (count, 28, 28), f'{split}: image schema')
                labels = digits[f'{split}_labels']
                require(labels.shape == (count,) and bool(((labels >= 0) & (labels <= 9)).all()),
                        f'{split}: label schema')
    except ImportError:
        errors.append('Install numpy to validate digit dataset schema')

    # Inspect the release tree, including untracked files, without touching collaborators' files.
    # Git internals and ordinary local caches are not release artifacts.
    ignored_dirs = {'.git', '__pycache__', '.venv', 'venv', '.pytest_cache'}
    suspicious_name = re.compile(r'(^|[_.-])(solutions?|instructor|checkpoints?)([_.-]|$)', re.I)
    secret_patterns = [
        re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
        re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
        re.compile(r'\bgithub_pat_[A-Za-z0-9_]{40,}\b'),
        re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),
        re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
        re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b'),
        re.compile(r'''(?im)^\s*(?:api_key|access_token|password|secret)\s*=\s*["'][^"'\n]{8,}["']'''),
    ]
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if any(part in ignored_dirs for part in relative.parts):
            continue
        require(path.suffix.lower() != '.ipynb' or path == notebook_path,
                f'Additional notebook requires explicit release review: {relative}')
        require(not any(suspicious_name.search(part) or part == '.ipynb_checkpoints'
                        for part in relative.parts), f'Unexpected private release artifact: {relative}')
        require(path.name not in {'.env', 'credentials.json', 'token.json'},
                f'Possible credential file: {relative}')
        if path.is_file() and path.suffix.lower() in {'.py', '.md', '.json', '.ipynb', '.txt', '.yaml', '.yml', '.toml'}:
            content = path.read_text(errors='replace')
            for pattern in secret_patterns:
                require(not pattern.search(content), f'Possible secret pattern: {relative}')
    if errors:
        for error in errors:
            print('FAIL:', error, file=sys.stderr)
        return 1
    print(f'PASS: notebook schema, empty outputs, syntax, five unanswered stubs, '
          f'generator parity, {len(checksums)} dataset checksums, release hygiene and secret scan')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        sys.exit(check(args.root.resolve()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        sys.exit(1)
