"""Isolate one unused text-encoder re-export; preserve all computational source."""
import pathlib,json,hashlib
R=pathlib.Path(__file__).resolve().parents[1];source=R/'source_archive/spf';target=R/'runtime_source/spf';assert not target.exists();target.mkdir(parents=True)
receipts=json.loads((source/'CACHE_RECEIPTS.json').read_text());files=[]
changed='emulator/models/pyramidal_flow/mmdit_modules/__init__.py'
for item in receipts['files']:
 p=source/item['path'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256'];dst=target/item['path'];dst.parent.mkdir(parents=True,exist_ok=True)
 if item['path']==changed:
  text=raw.decode('utf-8');line='from .modeling_text_encoder import SD3TextEncoderWithMask';assert text.count(line)==1;raw=text.replace(line,'# Text-encoder re-export unused by SPF core; omitted in the isolated adapter.').encode('utf-8')
 dst.write_bytes(raw);files.append({'path':item['path'],'sha256':hashlib.sha256(raw).hexdigest(),'original_sha256':item['sha256'],'import_only_change':item['path']==changed})
(target/'ADAPTER_MANIFEST.json').write_text(json.dumps({'files':files,'reason':'Unconditional package re-export imports an unused SD3 text encoder and Transformers; all numerical/core/scheduler/collator files remain unchanged.'},indent=2),encoding='utf-8')
print('Prepared import-only adapter, computational files unchanged',len(files),flush=True)
