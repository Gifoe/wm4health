"""Inspect original public ZIP directories using strictly bounded HTTP ranges."""
import datetime
import hashlib
import io
import json
import pathlib
import re
import urllib.parse
import urllib.request
import zipfile

R = pathlib.Path(__file__).resolve().parents[1]
OUT = R / 'outputs/zip_inventory_v1'
assert not OUT.exists(), 'Do not overwrite prior acquisition evidence'
OUT.mkdir(parents=True)
landing = (R / 'outputs/source_metadata_v1/response_2.txt').read_text(encoding='utf-8')
items = re.findall(r'<li>\s*<span class="size">([\d,]+) bytes</span>.*?<span class="md5">MD5:<code>([0-9a-f]+)</code></span>.*?<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', landing, re.S)
assert len(items) == 2, [(a,c,d) for a,b,c,d in items]

class RangeReader(io.RawIOBase):
    def __init__(self, url, size, log):
        self.url, self.size, self.pos, self.log = url, size, 0, log
        self.transferred = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos
    def read(self, count=-1):
        if count is None or count < 0: count = self.size - self.pos
        count = min(count, self.size - self.pos)
        if count <= 0: return b''
        assert count <= 32 * 1024 * 1024, ('Range too large', count)
        assert self.transferred + count <= 96 * 1024 * 1024, 'Metadata transfer cap'
        start, end = self.pos, self.pos + count - 1
        req = urllib.request.Request(self.url, headers={'User-Agent':'TSWorldModel-ReproducibilityAudit/1.0', 'Range': f'bytes={start}-{end}', 'Accept-Encoding':'identity'})
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status != 206:
                raise RuntimeError(f'Range unsupported: status={response.status}; no full-body download attempted')
            expected = f'bytes {start}-{end}/{self.size}'
            assert response.headers.get('Content-Range') == expected, (response.headers.get('Content-Range'), expected)
            data = response.read(count + 1)
            assert len(data) == count, (len(data), count)
            self.log.append({'range':expected, 'resolved_url':response.geturl(), 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest(), 'etag':response.headers.get('ETag'), 'last_modified':response.headers.get('Last-Modified')})
        self.pos += len(data)
        self.transferred += len(data)
        return data

result = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_doi':'10.4121/21960932', 'scope':'ZIP directory metadata only; no signal/image member extraction', 'archives':[]}
for i, (size_text, md5, href, name) in enumerate(items):
    entry = {'name':name, 'declared_size':int(size_text.replace(',','')), 'declared_md5':md5, 'url':urllib.parse.urljoin('https://data.4tu.nl',href), 'http_ranges':[]}
    try:
        remote = RangeReader(entry['url'], entry['declared_size'], entry['http_ranges'])
        with zipfile.ZipFile(remote) as archive:
            files = [{'path':z.filename, 'size':z.file_size, 'compressed_size':z.compress_size, 'crc32':f'{z.CRC:08x}', 'compression':z.compress_type, 'header_offset':z.header_offset} for z in archive.infolist()]
            (OUT / f'archive_{i}_all_members.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
            entry.update(status='DIRECTORY_READ', member_count=len(files), metadata_bytes=remote.transferred, timeseries_candidates=[z for z in files if any(x in z['path'].lower() for x in ['timeseries','greenhouseclimate','weather.xlsx']) or z['path'].lower().endswith(('.xlsx','.csv','.txt','.pdf'))])
    except Exception as error:
        entry.update(status='INCOMPLETE', error_type=type(error).__name__, error=str(error))
    result['archives'].append(entry)
    (OUT / 'inventory.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(entry),flush=True)
