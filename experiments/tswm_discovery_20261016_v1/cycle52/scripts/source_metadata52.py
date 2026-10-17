"""Bounded public metadata listing; never fetches signal/image payloads."""
import datetime
import hashlib
import html
import json
import pathlib
import re
import urllib.request

R = pathlib.Path(__file__).resolve().parents[1]
out = R / 'outputs/source_metadata_v1'
assert not out.exists(), 'Version the output instead of overwriting evidence'
out.mkdir(parents=True)
records = []
for i, url in enumerate([
    'https://api.4tu.nl/v2/articles/21960932',
    'https://api.figshare.com/v2/articles/21960932',
    'https://data.4tu.nl/articles/_/21960932',
]):
    entry = {'requested_url': url}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'TSWorldModel-ReproducibilityAudit/1.0', 'Accept': 'application/json, text/html;q=0.8'})
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read(2 * 1024 * 1024 + 1)
            assert len(data) <= 2 * 1024 * 1024, 'Metadata size limit'
            entry.update(status=response.status, resolved_url=response.geturl(), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        (out / f'response_{i}.txt').write_bytes(data)
        text = data.decode('utf-8', errors='replace')
        try:
            obj = json.loads(data)
            entry['json_keys'] = list(obj) if isinstance(obj, dict) else None
            if isinstance(obj, dict):
                entry['title'] = obj.get('title')
                entry['files'] = obj.get('files')
                entry['description'] = obj.get('description')
                entry['license'] = obj.get('license')
                entry['doi'] = obj.get('doi')
        except ValueError:
            entry['links'] = list(dict.fromkeys(html.unescape(x) for x in re.findall(r'(?:href|src)=[\"\']([^\"\']+)', text)))
            entry['api_context'] = [text[max(0,m.start()-150):m.end()+250] for m in list(re.finditer(r'(?i)download|api/|\.zip|\.xlsx|TimeSeries|__NEXT_DATA__|uuid|portal', text))[:40]]
    except Exception as error:
        entry.update(error_type=type(error).__name__, error=str(error))
    records.append(entry)
result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope': 'public metadata only; no signal/image payload', 'records': records}
(out / 'metadata.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
