"""Read public DOI metadata only; do not download greenhouse signals/images."""
import datetime
import hashlib
import json
import pathlib
import urllib.error
import urllib.request

R = pathlib.Path(__file__).resolve().parents[1]
out = R / 'outputs/source_contract_tswm_v1/metadata_v1'
assert not out.exists()
out.mkdir()
records = []
for i, url in enumerate(['https://api.datacite.org/dois/10.4121/21960932', 'https://doi.org/10.4121/21960932']):
    entry = {'requested_url': url}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'TSWorldModel-ReproducibilityAudit/1.0', 'Accept': 'application/vnd.api+json, application/json, text/html;q=0.8'})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read(1024 * 1024 + 1)
            assert len(data) <= 1024 * 1024, 'Metadata size bound exceeded'
            entry.update({'status': response.status, 'resolved_url': response.geturl(), 'content_type': response.headers.get('Content-Type'),
                          'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            (out / f'response_{i}.txt').write_bytes(data)
            try:
                obj = json.loads(data)
                attrs = obj.get('data', {}).get('attributes', {})
                entry['record'] = {k: attrs.get(k) for k in ['doi', 'url', 'titles', 'rightsList', 'publicationYear', 'relatedIdentifiers', 'contentUrl']}
            except (ValueError, AttributeError):
                pass
    except Exception as error:
        entry.update({'error_type': type(error).__name__, 'error': str(error)})
    records.append(entry)
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'Public metadata retrieval only; no signal/image download, no upstream changes',
          'records': records, 'source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
(out / 'metadata.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
