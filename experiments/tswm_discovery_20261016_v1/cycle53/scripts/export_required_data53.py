"""Record public source-file identities required by the native Greenhouse loader."""
from native53 import *

out = R / 'required_data_manifest_v1.json'; assert not out.exists()
root = pathlib.Path(STORAGE['data_root']) / 'greenhouse3_original_v1'
entries = []
for experiment in ['experiment_1', 'experiment_2']:
    folder = root / experiment / 'TimeSeries'
    teams = sorted(p for p in folder.iterdir() if p.is_dir() and (p / 'GreenhouseClimate.xlsx').exists())
    assert len(teams) == 6
    paths = [folder / 'Weather.xlsx']
    for team in teams: paths.extend([team / 'GreenhouseClimate.xlsx', team / 'GreenhouseCrop.xlsx'])
    for path in paths:
        assert path.is_file()
        entries.append({'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)})
assert len(entries) == 26
dump(out, {'utc': now(), 'source': 'https://doi.org/10.4121/21960932.v1',
    'provenance': 'C52 selectively acquired original archive members, CRC and SHA verified; large-disk copies verified. Weather archive-member placement for experiment 2 is recorded in C52.',
    'scope': 'Only the 26 original XLSX files read by the native climate/crop/weather loader. This contains no signal values; optional energy/document files are outside this input contract.',
    'files': entries})
print(json.dumps({'status': 'EXPORTED_NATIVE_REQUIRED_PUBLIC_FILE_IDENTITIES', 'files': len(entries), 'sha256': sha(out)}), flush=True)
