"""Independent file-integrity/cadence checks; no fitting or effect claims."""
import pathlib,json,hashlib,collections,datetime
R=pathlib.Path(__file__).resolve().parents[1]
dst=R/'outputs/artifact_verification_v1.json'
assert not dst.exists(),'Preserve previous verification'
verified=[]
def verify(path,expected,size=None,git_blob=None):
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected,str(path)
    if size is not None:assert len(raw)==size
    if git_blob is not None:
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==git_blob
    verified.append({'path':path.relative_to(R).as_posix(),'sha256':expected,'bytes':len(raw)})
manifest=json.loads((R/'source_audit/manifest.json').read_text(encoding='utf-8'))
for x in manifest['files']:
    verify(R/'source_audit'/x['path'],x['sha256'],x['bytes'],x.get('git_blob'))
meta=json.loads((R/'public_metadata/receipts.json').read_text(encoding='utf-8'))
for x in meta['files']:verify(R/'public_metadata'/x['path'],x['sha256'],x['bytes'])
raw_receipts=json.loads((R/'outputs/elexon_probe_v2/receipts.json').read_text(encoding='utf-8'))
for x in raw_receipts:
    verify(R/'outputs/elexon_probe_v2'/x['file'],x['sha256'],x['bytes'])
    verify(R/'public_metadata/elexon_raw_local'/x['file'],x['sha256'],x['bytes'])
report_manifest=json.loads((R/'REPORT_MANIFEST.json').read_text(encoding='utf-8'))
for x in report_manifest['reports']:verify(R/x['path'],x['sha256'],x['bytes'])
def load(name):return json.loads((R/'outputs/elexon_probe_v2'/name).read_text(encoding='utf-8'))
F=load('windfor_20260101.json')['data'];A=load('fuelhh_wind_20260102.json')['data'];P=load('result.json')
def dt(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
stamps=sorted({dt(r['startTime']) for r in F})
assert len(F)==584 and len(A)==48 and len(stamps)==73
assert all((b-a).total_seconds()==3600 for a,b in zip(stamps,stamps[1:]))
for item in P['asof_checks']:
    o=dt(item['origin']);eligible=sorted({dt(x['startTime']) for x in A if dt(x['startTime'])>o and dt(x['startTime']).minute==0})
    chosen={dt(x['startTime']):x for x in item['selected']}
    assert sorted(chosen)==eligible
    for t,v in chosen.items():
        candidates=[x for x in F if dt(x['startTime'])==t and dt(x['publishTime'])<=o]
        assert dt(v['publishTime'])==max(dt(x['publishTime']) for x in candidates)
        assert v['generation']==next(x['generation'] for x in candidates if x['publishTime']==v['publishTime'])
assert report_manifest['accepted_research_directions']==0
result={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files_sha_verified':len(verified),'source_audit_files':len(manifest['files']),
    'source_audit_bytes':sum(x['bytes'] for x in manifest['files']),
    'reports_verified':len(report_manifest['reports']),'independent_hourly_asof_checks':3,
    'no_forecast_scoring_or_training':True,'accepted_research_directions':0,'files':verified}
dst.write_text(json.dumps(result,indent=2),encoding='utf-8')
marker='<!-- cycle36-index-start -->'
for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md',
             'hypothesis_space.md','research_ideas.md','final_research_direction.md']:
    path=R.parent/name;old=path.read_text(encoding='utf-8');assert marker not in old
    prefix=f'''{marker}
# 最新证据：C36，整体研究仍未完成

文献/源码和服务器一日发行数据资格审计已完成。当前observer-only提案与普通校准积分目标等价，关闭其新机制主张；没有新模型训练或方法效果证据，接受最终方向0。原生预报为小时粒度，24/48对齐不是缺半天。见[本项最新报告](cycle36/{name})与[总体状态](CURRENT_RESEARCH_STATUS.md)。下面内容均为各轮历史，旧成果保留。
<!-- cycle36-index-end -->

'''
    path.write_text(prefix+old,encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='files'}))
