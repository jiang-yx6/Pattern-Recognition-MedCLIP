"""Download the frozen MIMIC-CXR-JPG manifest with GNU Wget.
Existing JPEGs are fully decoded before being skipped. Credentials are entered
only in Wget's terminal prompt. This script never changes dataset manifests.
"""
from pathlib import Path
import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from urllib.parse import urlsplit

from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[1]
URL_ROOT = 'https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/'
ImageFile.LOAD_TRUNCATED_IMAGES = False


def read_manifest(path, output_dir):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'download_url', 'image_path'}.issubset(reader.fieldnames or []):
            raise ValueError('Manifest requires download_url and image_path columns')
        rows = list(reader)
    if not rows:
        raise ValueError('Manifest is empty')
    seen = set()
    image_root = (output_dir / 'images').resolve()
    for row in rows:
        url = row['download_url']
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname != 'physionet.org' or not url.startswith(URL_ROOT):
            raise ValueError('Manifest contains an unexpected download URL')
        if row['image_path'] != 'images/' + url[len(URL_ROOT):]:
            raise ValueError('URL and image_path do not match')
        target = (output_dir / row['image_path']).resolve()
        target.relative_to(image_root)
        if target in seen:
            raise ValueError(f'Duplicate image path: {target}')
        seen.add(target)
    return rows


def inspect_image(path):
    if not path.is_file():
        return 'missing', 'File not found'
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.format != 'JPEG':
                raise ValueError('File content is not JPEG')
        return 'ok', ''
    except (OSError, ValueError, SyntaxError) as error:
        return 'invalid', str(error)


def write_status(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_path', 'status', 'error'])
        writer.writeheader()
        writer.writerows(rows)


def build_wget_command(executable, username, proxy, retries, timeout, image_dir, urls):
    command = [executable, '--continue', '--force-directories', '--no-host-directories',
               '--cut-dirs=4', '--user=' + username, '--ask-password',
               '--directory-prefix=' + str(image_dir), '--input-file=' + str(urls),
               '--tries=' + str(retries), '--timeout=' + str(timeout)]
    if proxy:
        parsed = urlsplit(proxy)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('--proxy requires a proxy URL without embedded credentials')
        command += ['-e', 'use_proxy=on', '-e', 'https_proxy=' + proxy]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'subset/mimic_5000_full.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'subset')
    parser.add_argument('--username', help='PhysioNet username; prompted only if downloads are needed')
    parser.add_argument('--proxy', help='Optional HTTP proxy, for example http://127.0.0.1:10808')
    parser.add_argument('--retries', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--check-only', action='store_true', help='Check files without downloading')
    args = parser.parse_args()
    if args.retries < 1 or args.timeout < 1:
        parser.error('--retries and --timeout must be positive')
    output_dir = args.output_dir.resolve()
    rows = read_manifest(args.manifest.resolve(), output_dir)
    config = json.loads((ROOT / 'configs/default.json').read_text(encoding='utf-8-sig'))
    log_dir = ROOT / config['step0_dir'] / 'checks' / 'download'
    log_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    pending = []
    for index, row in enumerate(rows, 1):
        status, error = inspect_image(output_dir / row['image_path'])
        statuses.append({'image_path': row['image_path'], 'status': status, 'error': error})
        if status != 'ok':
            pending.append(row)
        if index % 500 == 0 or index == len(rows):
            print(f'Checked {index}/{len(rows)}; pending={len(pending)}', flush=True)
    write_status(log_dir / 'status.csv', statuses)
    if not pending:
        print(f'All {len(rows)} images pass full JPEG decoding. No downloads needed.')
        return 0
    print(f'Complete: {len(rows)-len(pending)}; missing or invalid: {len(pending)}', flush=True)
    if args.check_only:
        return 1
    executable = shutil.which('wget.exe') or shutil.which('wget')
    if executable is None:
        raise RuntimeError('GNU Wget is required; install it and add it to PATH')
    username = (args.username or input('PhysioNet username: ')).strip()
    if not username:
        raise ValueError('PhysioNet username is required')
    urls = log_dir / 'pending_urls.txt'
    urls.write_text(''.join(row['download_url'] + '\n' for row in pending), encoding='utf-8')
    command = build_wget_command(executable, username, args.proxy, args.retries,
                                 args.timeout, output_dir / 'images', urls)
    # Preserve invalid files, then obtain clean replacements instead of trusting nonzero sizes.
    for row in pending:
        source = output_dir / row['image_path']
        if source.is_file():
            backup = log_dir / 'invalid_images' / row['image_path']
            if backup.exists():
                backup = backup.with_name(backup.name + '.' + datetime.now().strftime('%Y%m%d%H%M%S%f'))
            backup.parent.mkdir(parents=True, exist_ok=True)
            source.replace(backup)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    print('Starting GNU Wget. Enter your password in its terminal prompt.', flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    updated = {row['image_path']: inspect_image(output_dir / row['image_path']) for row in pending}
    for record in statuses:
        if record['image_path'] in updated:
            record['status'], record['error'] = updated[record['image_path']]
    write_status(log_dir / 'status.csv', statuses)
    failures = sum(record['status'] != 'ok' for record in statuses)
    print(f'Complete: {len(rows)-failures}/{len(rows)}; missing or invalid: {failures}', flush=True)
    if completed.returncode or failures:
        print(f'Download incomplete. Inspect Wget output and {log_dir / "status.csv"}; rerun to retry.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as error:
        print(f'STOPPED: {error}', file=sys.stderr)
        raise SystemExit(1)
