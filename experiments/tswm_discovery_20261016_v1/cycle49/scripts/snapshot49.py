"""Compact evidence snapshot, excluding signals, weights, environments and secrets."""
import datetime
import hashlib
import json
import pathlib
import zipfile

R = pathlib.Path(__file__).resolve().parents[1]
ROOT = R.parent
out = R / 'outputs/discovery_progress_snapshot_v1.zip'
receipt = R / 'outputs/SNAPSHOT_RECEIPT.json'
assert not out.exists() and not receipt.exists()
paths = []
for folder in [ROOT / 'cycle48', R]:
    for p in folder.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        relative = p.relative_to(folder)
        if relative.parts[0] in ['source_archive', 'assets', 'vendor']:
            continue
        if relative.parts[0] == 'data' and p.name != 'MANIFEST.json':
            continue
        if p.suffix not in ['.py', '.md', '.json', '.csv', '.png', '.svg', '.log']:
            continue
        if p.stat().st_size > 10 * 1024**2:
            continue
        paths.append(p)
paths.extend(ROOT / name for name in ['CURRENT_RESEARCH_STATUS.md', 'literature_review.md', 'baseline_registry.md',
                                      'experimental_results.md', 'negative_evidence.md', 'hypothesis_space.md',
                                      'research_ideas.md', 'final_research_direction.md'])
records = []
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(set(paths)):
        data = path.read_bytes()
        name = path.relative_to(ROOT).as_posix()
        archive.writestr(name, data)
        records.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    archive.writestr('SNAPSHOT_MANIFEST.json', json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                                        'files': records, 'goal_complete': False}, indent=2))
value = {'path': out.name, 'bytes': out.stat().st_size, 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'files': len(records),
         'scope': 'C48 completed compact evidence and C49 dated live-progress snapshot; raw data and checkpoints remain on server', 'goal_complete': False}
receipt.write_text(json.dumps(value, indent=2), encoding='utf-8')
print(json.dumps(value), flush=True)
