"""Pin native acquisition-aware baseline before defining a new experiment."""
import base64, datetime, hashlib, json, pathlib, re
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = 'ant-intl/Timeflies'

def get(url):
    s = requests.Session(); s.trust_env = False
    r = s.get(url, timeout=(5, 25)); r.raise_for_status(); return r.json()

def main():
    out = ROOT / 'outputs'; out.mkdir(parents=True, exist_ok=True)
    tp = out / 'timeflies_tree.json'; cp = out / 'timeflies_commit.json'
    if tp.exists():
        tree = json.loads(tp.read_text(encoding='utf-8')); commit = json.loads(cp.read_text(encoding='utf-8'))
    else:
        meta = get('https://api.github.com/repos/' + REPO)
        commit = get('https://api.github.com/repos/' + REPO + '/commits/' + meta['default_branch'])
        cp.write_text(json.dumps(commit, indent=2), encoding='utf-8')
        tree = get('https://api.github.com/repos/' + REPO + '/git/trees/' + commit['sha'] + '?recursive=1')
        tp.write_text(json.dumps(tree, indent=2), encoding='utf-8')
    assert not tree.get('truncated')
    blobs = [r for r in tree['tree'] if r['type'] == 'blob']
    agents = [r for r in blobs if pathlib.PurePosixPath(r['path']).name == 'AGENTS.md']
    suffixes = {'.py', '.sh', '.md', '.txt', '.json', '.yaml', '.yml', '.toml'}
    selected = [r for r in blobs if (pathlib.PurePosixPath(r['path']).suffix in suffixes or pathlib.PurePosixPath(r['path']).name.startswith('LICENSE')) and r.get('size', 0) < 2_000_000]
    selected = list({r['path']: r for r in agents + selected}.values())
    assert len(selected) <= 200, len(selected)
    folder = ROOT / 'sources/timeflies'; rows = []; flags = []
    for row in selected:
        rel = pathlib.PurePosixPath(row['path']); assert not rel.is_absolute() and '..' not in rel.parts
        path = folder.joinpath(*rel.parts)
        if path.exists():
            raw = path.read_bytes()
        else:
            blob = get(row['url']); assert blob['encoding'] == 'base64'; raw = base64.b64decode(blob['content'])
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == row['sha']
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        rows.append({'path': row['path'], 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob': row['sha']})
        # Flag filenames only. Never print a possible credential value.
        if re.search(rb'(?:WANDB_API_KEY|HF_TOKEN|OPENAI_API_KEY)\s*[=:\]]\s*[\x22\x27][A-Za-z0-9_\-]{15,}', raw):
            flags.append(row['path'])
    pin = {'repository': REPO, 'revision': commit['sha'], 'commit_date': commit['commit']['committer']['date'],
           'tree_revision': tree['sha'], 'complete_tree_files': len(blobs), 'retained_files': len(rows),
           'AGENTS_files': [r['path'] for r in agents], 'credential_pattern_flag_files': flags,
           'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': rows,
           'scope': 'Native source audit only. No launcher/import/external service or scientific result claimed.'}
    (folder / 'SOURCE_PIN.json').write_text(json.dumps(pin, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in pin.items() if k != 'files'}), flush=True)

if __name__ == '__main__':
    main()
