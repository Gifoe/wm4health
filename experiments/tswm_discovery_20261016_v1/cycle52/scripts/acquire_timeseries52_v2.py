"""Selective, verified XLSX/DOCX acquisition; no image member is downloaded."""
import datetime
import hashlib
import json
import pathlib
import struct
import urllib.request
import zlib

R = pathlib.Path(__file__).resolve().parents[1]
OUT = R / 'outputs/acquisition_v2'
DATA = R / 'data/greenhouse3_original_v1'
assert not OUT.exists(), 'Do not overwrite acquisition evidence'
OUT.mkdir(parents=True)
assert DATA.exists()
inventory = json.loads((R / 'outputs/zip_inventory_v1/inventory.json').read_text())
result = json.loads((R / 'outputs/acquisition_v1/receipt.json').read_text())
assert len(result['archives']) == 1 and result['archives'][0]['archive'] == '1st Experiment.zip'
for old_file in result['archives'][0]['files']:
    data = (R / old_file['local_path']).read_bytes()
    assert len(data) == old_file['bytes'] and hashlib.sha256(data).hexdigest() == old_file['sha256']
result['resume_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
result['repair'] = 'V1 guard rejected Images/ReadMe.docx before fetching second archive; exclude image subtree and allow root time-series readme. Verified first archive is reused without download.'
for i, archive in enumerate(inventory['archives']):
    if any(x['archive'] == archive['name'] for x in result['archives']): continue
    assert archive['status'] == 'DIRECTORY_READ'
    members = json.loads((R / f'outputs/zip_inventory_v1/archive_{i}_all_members.json').read_text())
    selected = [m for m in members if '/Images/' not in m['path'] and not pathlib.PurePosixPath(m['path']).name.startswith('~$') and pathlib.PurePosixPath(m['path']).name in {'GreenhouseClimate.xlsx','GreenhouseControls.xlsx','GreenhouseCrop.xlsx','Weather.xlsx','ReadMe.docx'}]
    assert len([m for m in selected if m['path'].endswith('/GreenhouseClimate.xlsx')]) == 6
    start = min(m['header_offset'] for m in selected)
    central_start = int(archive['http_ranges'][-1]['range'].split()[1].split('-')[0])
    end = central_start - 1
    # The contiguous tail region contains time-series material only, as checked
    # against the full central directory before any payload request.
    region_members = [m for m in members if start <= m['header_offset'] < central_start]
    assert all('timeseries' in m['path'].lower() or m['path'].endswith('/Weather.xlsx') or (m['path'].count('/') == 1 and m['path'].endswith('/ReadMe.docx')) for m in region_members)
    assert all(not m['path'].lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff','.bmp')) for m in region_members)
    count = end - start + 1
    assert 0 < count < 24 * 1024 * 1024
    req = urllib.request.Request(archive['url'], headers={'User-Agent':'TSWorldModel-ReproducibilityAudit/1.0','Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,timeout=45) as response:
        assert response.status == 206, 'Range unsupported; do not download whole body'
        content_range = f'bytes {start}-{end}/{archive["declared_size"]}'
        assert response.headers.get('Content-Range') == content_range
        assert response.headers.get('ETag') == archive['http_ranges'][-1]['etag'], 'Archive changed since inventory'
        raw = response.read(count + 1)
        assert len(raw) == count
        receipt = {'archive':archive['name'],'source_url':archive['url'],'http_range':content_range,'etag':response.headers.get('ETag'),'transferred_bytes':len(raw),'range_sha256':hashlib.sha256(raw).hexdigest(),'files':[]}
    for member in selected:
        offset = member['header_offset'] - start
        header = struct.unpack_from('<4s5H3I2H',raw,offset)
        signature,version,flags,compression,mtime,mdate,crc,sizec,sizeu,nname,nextra = header
        assert signature == b'PK\x03\x04' and not (flags & 1)
        name_raw = raw[offset+30:offset+30+nname]
        name = name_raw.decode('utf-8' if flags & 0x800 else 'cp437')
        assert name == member['path'] and compression == member['compression']
        payload_start = offset + 30 + nname + nextra
        compressed = raw[payload_start:payload_start+member['compressed_size']]
        assert len(compressed) == member['compressed_size']
        data = zlib.decompress(compressed,-15) if compression == 8 else compressed
        assert len(data) == member['size']
        assert f'{zlib.crc32(data)&0xffffffff:08x}' == member['crc32']
        parts = pathlib.PurePosixPath(name).parts
        series_idx = next((j for j,p in enumerate(parts) if p.lower().startswith('timeseries')),None)
        relative = pathlib.PurePosixPath(*parts[series_idx+1:]) if series_idx is not None else pathlib.PurePosixPath(parts[-1])
        assert not relative.is_absolute() and '..' not in relative.parts
        dest = DATA / f'experiment_{i+1}' / 'TimeSeries' / relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists()
        dest.write_bytes(data)
        receipt['files'].append({'archive_member':name,'local_path':dest.relative_to(R).as_posix(),'bytes':len(data),'zip_crc32':member['crc32'],'sha256':hashlib.sha256(data).hexdigest(),'layout_note':'Native bytes preserved; Weather.xlsx placed at loader root, original archive path retained here'})
    result['archives'].append(receipt)
    (OUT / 'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'archive':archive['name'],'status':'SELECTIVE_ACQUISITION_VERIFIED','files':len(receipt['files']),'transferred_bytes':count,'extracted_bytes':sum(x['bytes'] for x in receipt['files'])}),flush=True)
result['status']='COMPLETE_ORIGINAL_TIMESERIES_ONLY'
(OUT / 'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
