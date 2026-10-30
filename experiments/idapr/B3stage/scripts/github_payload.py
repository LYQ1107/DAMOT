"""Bounded binary-safe payloads for publishing the recorded experiment package."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

Q = Path(__file__).resolve().parents[1]
MANIFEST = Q / 'manifests/github_upload_files.json'
CACHE = Q / 'manifests/github_blob_uploads.json'


def files():
    rows = json.loads(MANIFEST.read_text())
    for row in rows:
        assert row['path'].startswith('experiments/idapr/B3stage/')
        assert Path(row['source']).is_relative_to(Q / 'publication')
    return rows


parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['pending', 'content', 'record', 'manifest'])
parser.add_argument('--sha')
parser.add_argument('--offset', type=int, default=0)
parser.add_argument('--count', type=int, default=40)
parser.add_argument('--record-json')
args = parser.parse_args()
rows = files()
if args.action == 'manifest':
    print(json.dumps(rows[args.offset:args.offset + args.count]))
elif args.action == 'pending':
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    unique = {row['git_blob_sha']: row for row in rows}
    pending = [row for sha, row in unique.items() if cache.get(sha) != sha]
    print(json.dumps(dict(files=len(rows), unique=len(unique), pending_count=len(pending),
                          pending=pending[args.offset:args.offset + args.count])))
elif args.action == 'content':
    row = next(row for row in rows if row['git_blob_sha'] == args.sha)
    data = Path(row['source']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    assert hashlib.sha1(('blob %d\0' % len(data)).encode() + data).hexdigest() == args.sha
    print(base64.b64encode(data[args.offset:args.offset + 6144]).decode())
elif args.action == 'record':
    added = json.loads(args.record_json)
    assert isinstance(added, list)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for sha in added:
        assert len(sha) == 40 and all(c in '0123456789abcdef' for c in sha)
        assert any(row['git_blob_sha'] == sha for row in rows)
        cache[sha] = sha
    CACHE.write_text(json.dumps(cache, indent=2) + '\n')
    print('RECORDED', len(added), 'TOTAL_KNOWN', len(cache))
