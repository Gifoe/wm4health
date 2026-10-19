"""Reuse a verified public series; pin two small native benchmark series."""
import datetime, hashlib, json, pathlib
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
REV = '733d986d349e9e71ded5d9cdeeeed8d3323d83b4'

def main():
    out = ROOT / 'outputs'; out.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'data'; data.mkdir(exist_ok=True)
    rows = []
    existing = ROOT.parent / 'cycle5/data/ETTh1.csv'
    assert existing.is_file()
    rows.append({'key': 'ETTh1', 'path': '../cycle5/data/ETTh1.csv', 'source': 'read-only reuse of sealed cycle5 public source',
                 'bytes': existing.stat().st_size, 'sha256': hashlib.sha256(existing.read_bytes()).hexdigest()})
    s = requests.Session(); s.trust_env = False
    tp = out / 'prodyn_data_tree.json'
    if tp.exists():
        tree = json.loads(tp.read_text(encoding='utf-8'))
    else:
        r = s.get(f'https://api.github.com/repos/ARBrachet/PRO-DYN/git/trees/{REV}?recursive=1', timeout=(5, 25))
        r.raise_for_status(); tree = r.json(); tp.write_text(json.dumps(tree, indent=2), encoding='utf-8')
    assert not tree.get('truncated')
    for key in ['ILI', 'NYSE']:
        candidates = [r for r in tree['tree'] if r['type'] == 'blob' and pathlib.PurePosixPath(r['path']).name.lower() == key.lower() + '.csv']
        if not candidates:
            print(json.dumps({'dataset': key, 'status': 'not_in_pinned_tree'}), flush=True); continue
        assert len(candidates) == 1
        item = candidates[0]; assert item['size'] < 10_000_000
        url = f'https://raw.githubusercontent.com/ARBrachet/PRO-DYN/{REV}/' + item['path']
        dest = data / (key + '.csv')
        if not dest.exists():
            r = s.get(url, timeout=(5, 25)); r.raise_for_status(); dest.write_bytes(r.content)
        raw = dest.read_bytes()
        assert len(raw) == item['size'] and hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == item['sha']
        rows.append({'key': key, 'path': dest.relative_to(ROOT).as_posix(), 'source': url,
                     'source_revision': REV, 'git_blob': item['sha'], 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    report = {'datasets': rows, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Public raw observed numerical series for a new controlled acquisition-policy experiment, not native benchmark results.'}
    (out / 'data_registry.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
