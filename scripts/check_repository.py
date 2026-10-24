"""Check Git submission contents without loading data or installing ML packages.

Uses the real Git index when this project is a repository. Before git init, an
isolated temporary Git directory checks the same worktree and .gitignore rules.
Never stages, commits, uploads or deletes project files.
"""
from pathlib import Path
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOTS = {'raw', 'subset', 'outputs', 'MedCLIP', 'tmp', '.cache', '.vscode', '.idea'}
BINARY_SUFFIXES = {'.zip', '.rar', '.7z', '.bin', '.pt', '.pth', '.safetensors', '.npz', '.npy',
                   '.jpg', '.jpeg', '.png', '.pdf', '.gz', '.pem', '.key', '.log'}
SECRET_PATTERNS = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{20,}\b'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{20,}\b'),
    re.compile(r'\bhf_[A-Za-z0-9]{20,}\b'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(r'(?i)https?://[^\s/:@]+:[^\s/@]+@'),
]


def git(executable, *args):
    return subprocess.run([executable, *map(str, args)], capture_output=True, check=True).stdout


def submission_files(executable):
    top = subprocess.run([executable, '-C', str(ROOT), 'rev-parse', '--show-toplevel'],
                         capture_output=True, text=True)
    if top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT:
        prefix = ['-C', ROOT]
        result = git(executable, *prefix, 'ls-files', '-z', '--cached', '--others', '--exclude-standard')
    else:
        cache = ROOT / '.cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='repository-check-', dir=cache) as directory:
            git(executable, 'init', '--quiet', directory)
            result = git(executable, f'--git-dir={Path(directory) / ".git"}',
                         f'--work-tree={ROOT}', 'ls-files', '-z', '--cached', '--others', '--exclude-standard')
    return sorted(set(name.decode('utf-8') for name in result.split(b'\0') if name))


def main():
    executable = shutil.which('git')
    if not executable:
        raise RuntimeError('Git is required to check the actual submission boundary')
    files = submission_files(executable)
    top = subprocess.run([executable, '-C', str(ROOT), 'rev-parse', '--show-toplevel'],
                         capture_output=True, text=True)
    has_index = top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT
    errors = []
    total = 0
    for name in files:
        relative = Path(name)
        path = ROOT / relative
        parts = relative.parts
        if (parts[0] in PRIVATE_ROOTS or parts[0].startswith('.venv')
                or '__pycache__' in parts or path.suffix.lower() in BINARY_SUFFIXES
                or (path.name.startswith('.env') and path.name != '.env.example')):
            errors.append(f'Local data, artifact or credential file is included: {name}')
            continue
        if path.is_symlink() or not path.is_file():
            errors.append(f'Unsupported symlink, missing file or nested repository: {name}')
            continue
        path.resolve().relative_to(ROOT)
        total += path.stat().st_size
        if path.stat().st_size > 1024 * 1024:
            errors.append(f'Unexpected large source file: {name}')
            continue
        try:
            source = path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            errors.append(f'Unexpected non-text submission: {name}')
            continue
        patient_header = ','.join(['patient_id', 'study_id', 'image_path', 'report', 'split', 'labels', 'mask'])
        versions = [('worktree', source)]
        if has_index:
            staged = subprocess.run([executable, '-C', str(ROOT), 'show', ':' + name], capture_output=True)
            if staged.returncode == 0:
                if len(staged.stdout) > 1024 * 1024:
                    errors.append(f'Unexpected large staged file: {name}')
                    continue
                try:
                    versions.append(('index', staged.stdout.decode('utf-8-sig')))
                except UnicodeError:
                    errors.append(f'Unexpected non-text staged submission: {name}')
                    continue
        for location, content in versions:
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                errors.append(f'Possible credential in {name} ({location}); inspect locally')
            if patient_header in content:
                errors.append(f'Possible real patient manifest in {name} ({location})')
            if path.suffix == '.py':
                ast.parse(content, filename=f'{name} ({location})')
            if path.suffix == '.json':
                json.loads(content)
    required = {'README.md', '.gitignore', 'configs/default.json', 'hypotheses.md',
                'requirements-medclip.txt', 'prepare_data.py', 'extract_features.py',
                'train_metric.py', 'mine_negatives.py', 'evaluate_retrieval.py', 'analyze_geometry.py',
                'scripts/download_images.py', 'scripts/audit_dataset.py', 'scripts/check_medclip.py',
                'scripts/check_repository.py', 'docs/handoff.md'}
    required.update({'mimic_pipeline/'+name+'.py' for name in ['__init__','common','checkpoint','data','features','metric','mining','retrieval','geometry']})
    errors.extend(f'Missing source entry: {name}' for name in sorted(required - set(files)))
    for name in files:
        print(name)
    for error in errors:
        print('FAIL:', error, file=sys.stderr)
    print(f'{"FAIL" if errors else "PASS"}: {len(files)} source files; {total} bytes; '
          f'{len(errors)} submission issues')
    print('Checks current tracked files and unignored files; does not certify past Git history or data-access eligibility.')
    return 1 if errors else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, SyntaxError) as error:
        print(f'STOPPED: {error}', file=sys.stderr)
        raise SystemExit(1)
