"""Verify published Git trees against local bytes and preserve unrelated paths."""
from common import *


def read(name):
    return json.loads((Q / 'manifests' / name).read_text())


def entries(tree):
    assert not tree.get('truncated', False), 'Incomplete remote Git tree'
    return {r['path']: (r['mode'], r['type'], r['sha']) for r in tree['tree']}


publication = read('github_publication.json')
before_root = entries(read('github_parent_root_tree.json'))
after_root = entries(read('github_published_root_tree.json'))
before_exp = entries(read('github_parent_experiments_tree.json'))
after_exp = entries(read('github_published_experiments_tree.json'))
assert {k: v for k, v in before_root.items() if k != 'experiments'} == {
    k: v for k, v in after_root.items() if k != 'experiments'}
assert {k: v for k, v in before_exp.items() if k != 'ics_v2'} == {
    k: v for k, v in after_exp.items() if k != 'ics_v2'}
assert 'ics_v2' not in before_exp, 'Inspect any pre-existing task prefix first'
package = entries(read('github_published_package_tree.json'))
remote = {k: v for k, v in package.items() if v[1] == 'blob'}
manifest = read('github_upload_files.json')
local = {}
for row in manifest:
    assert row['path'].startswith('experiments/ics_v2/')
    data = Path(row['source']).read_bytes()
    assert len(data) == row['bytes']
    assert hashlib.sha256(data).hexdigest() == row['sha256']
    blob = hashlib.sha1(('blob %d\0' % len(data)).encode() + data).hexdigest()
    assert blob == row['git_blob_sha']
    local[row['path'][len('experiments/ics_v2/'):]] = ('100644', 'blob', blob)
assert remote == local, 'Remote files differ from the staged package'
commit = read('github_published_commit.json')
assert commit['sha'] == publication['commit_sha']
assert [p['sha'] for p in commit['parents']] == [publication['parent_sha']]
assert commit['tree']['sha'] == publication['tree_sha']
assert after_root['experiments'][2] == read('github_published_experiments_tree.json')['sha']
assert after_exp['ics_v2'][2] == read('github_published_package_tree.json')['sha']
assert read('github_published_root_tree.json')['sha'] == publication['tree_sha']
result = dict(status='passed', method='GitHub API Git tree readback',
              commit_sha=publication['commit_sha'], parent_sha=publication['parent_sha'],
              prefix='experiments/ics_v2/', files=len(local),
              bytes=sum(r['bytes'] for r in manifest),
              every_remote_git_blob_matches_local_bytes=True,
              every_local_sha256_rechecked=True,
              all_paths_outside_prefix_unchanged=True,
              remote_binary_sha256_download=False)
dump(Q / 'manifests/github_publication_verification.json', result)
print(json.dumps(result), flush=True)
