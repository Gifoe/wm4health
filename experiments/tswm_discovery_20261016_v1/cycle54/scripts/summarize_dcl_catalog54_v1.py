"""Curate public file metadata from the verified native Windows TLS receipt."""
import datetime, hashlib, json, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def main():
    source=OUT/'dcl_catalog_probe54_native_v2.json'; raw=json.loads(source.read_text(encoding='utf-8-sig'))
    api=next(x for x in raw['requests'] if '/api/records/' in x['url'])
    assert api['status']==200 and api['record_id']=='tdfs2-kx054'
    metadata=api['metadata']; entries=api['files']['entries']
    files=[{'key':k,'size_bytes':v['size'],'checksum':v['checksum'],'content_url':v['links']['content']} for k,v in sorted(entries.items())]
    assert len(files)>0 and all(f['size_bytes']>0 and f['checksum'].startswith('md5:') for f in files)
    obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PUBLIC_DCL_CATALOG_AVAILABLE_FROM_SERVER_WITH_DEFAULT_WINDOWS_TLS','source_record_id':api['record_id'],'source_url':api['url'],'title':metadata.get('title'),'publication_date':metadata.get('publication_date'),'rights':metadata.get('rights'),'version':metadata.get('version'),'files_count':len(files),'total_size_bytes':sum(f['size_bytes'] for f in files),'files':files,'signals_downloaded':False,'raw_receipt_sha256':sha(source),'script_sha256':sha(__file__),'transport_boundary':'Python urllib default CA lookup failed; unchanged Windows default certificate validation succeeded. No TLS validation bypass or global certificate/environment change.'}
    target=OUT/'dcl_catalog_summary54_v1.json'; assert not target.exists(); target.write_text(json.dumps(obj,indent=2,ensure_ascii=True),encoding='utf-8')
    text='# DCL public data: server access confirmed\n\n'
    text+=f"At {obj['utc']}, the public record API returned HTTP200 with {len(files)} file entries and {obj['total_size_bytes']:,} advertised bytes in total. These are catalog sizes, not downloaded signal bytes. No signals were downloaded. The exact filenames, provider MD5 checksums, content links and reported rights are retained in outputs/dcl_catalog_summary54_v1.json.\n\n"
    text+='The first Python urllib metadata probe failed to resolve the certificate issuer. A separate request through unchanged native Windows certificate validation succeeded for the record page and API. This establishes metadata availability from this server; it does not establish that every signal file has been downloaded, hash-verified or decoded. Both transport receipts are preserved. No insecure TLS fallback was used.\n\n'
    text+='The source/input boundary remains DCL_INFERENCE_CONTRACT54_V1.md. Future use should select only files that answer a discriminating mechanism question, freeze their advertised and actual hashes, and disclose the training/heldout protocol. Do not run the full Table1 sweep just to increase reproduction count. No DCL fit or forecast comparison is claimed here.\n\n'
    text+='Catalog summary SHA256: '+sha(target)+'\nRaw native TLS receipt SHA256: '+sha(source)+'\n'
    report=ROOT/'DCL_DATA_ACCESS54_V1.md'; assert not report.exists(); report.write_text(text,encoding='utf-8')
    print(json.dumps({k:obj[k] for k in ['status','files_count','total_size_bytes','signals_downloaded','rights']}),flush=True)
if __name__=='__main__': main()
