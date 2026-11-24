"""Shared, deterministic task-preparation helpers; no agent/model imports."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def opaque_id(kind, value, seed):
    return kind + '_' + hashlib.sha256(f'pathorob-camelyon-v1|{seed}|{kind}|{value}'.encode()).hexdigest()[:24]


def file_inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file()}
