"""Bounded primary-data qualification, no fitting or forecast scoring."""
import pathlib,json,hashlib,urllib.request,datetime,collections,math
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/elexon_probe_v1'
assert not OUT.exists(),'Preserve partial probe; never overwrite an existing audit'
OUT.mkdir(parents=True);receipts=[]
def fetch(name,url):
    request=urllib.request.Request(url,headers={'User-Agent':'TSWM-public-provenance-audit/1.0','Accept':'application/json'})
    with urllib.request.urlopen(request,timeout=45) as response:
        assert response.status==200
        raw=response.read(12*1024*1024+1);assert len(raw)<=12*1024*1024
    path=OUT/name;path.write_bytes(raw)
    receipts.append({'file':name,'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
                     'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    (OUT/'receipts.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    return json.loads(raw)
def stamp(x):
    t=datetime.datetime.fromisoformat(x.replace('Z','+00:00'))
    assert t.tzinfo is not None;return t.astimezone(datetime.timezone.utc)
schema=fetch('schema.json','https://data.elexon.co.uk/swagger/v1/swagger.json')
required_paths=['/datasets/WINDFOR','/datasets/FUELHH']
contracts={}
for suffix in required_paths:
    matches=[(k,v) for k,v in schema['paths'].items() if k.endswith(suffix)]
    assert len(matches)==1,(suffix,[k for k,v in matches])
    k,v=matches[0];contracts[suffix]={'schema_path':k,'get':v['get']}
forecast=fetch('windfor_20260101.json','https://data.elexon.co.uk/bmrs/api/v1/datasets/WINDFOR?publishDateTimeFrom=2026-01-01T00:00:00Z&publishDateTimeTo=2026-01-02T00:00:00Z&format=json')
actual=fetch('fuelhh_wind_20260102.json','https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH?settlementDateFrom=2026-01-02&settlementDateTo=2026-01-02&fuelType=WIND&format=json')
F=forecast['data'];A=actual['data'];assert F and A
for rows,ds in [(F,'WINDFOR'),(A,'FUELHH')]:
    for r in rows:
        assert r['dataset']==ds and all(x in r for x in ['publishTime','startTime','generation'])
        stamp(r['publishTime']);stamp(r['startTime']);assert math.isfinite(float(r['generation']))
for r in A:assert r['fuelType']=='WIND'
def describe(rows,extra_key=()):
    keys=[tuple(r[k] for k in ('publishTime','startTime')+extra_key) for r in rows]
    counts=collections.Counter(keys)
    return {'rows':len(rows),'fields':sorted(rows[0]),'duplicates':sum(n-1 for n in counts.values()),
            'publish_min':min(r['publishTime'] for r in rows),'publish_max':max(r['publishTime'] for r in rows),
            'start_min':min(r['startTime'] for r in rows),'start_max':max(r['startTime'] for r in rows),
            'unique_starts':len({r['startTime'] for r in rows})}
targets=sorted({stamp(r['startTime']) for r in A})
assert len(targets)==48 and all((b-a).total_seconds()==1800 for a,b in zip(targets,targets[1:]))
checks=[]
for text in ['2026-01-01T12:00:00Z','2026-01-01T18:00:00Z','2026-01-02T00:00:00Z']:
    origin=stamp(text);buckets=collections.defaultdict(list)
    for r in F:
        p=stamp(r['publishTime']);t=stamp(r['startTime'])
        if p<=origin<t and t in targets:buckets[t].append(r)
    producer={t:max(rows,key=lambda r:stamp(r['publishTime'])) for t,rows in buckets.items()}
    independent={}
    for t in targets:
        chosen=None;chosen_p=None
        for r in F:
            if stamp(r['startTime'])!=t:continue
            p=stamp(r['publishTime'])
            if p<=origin<t and (chosen_p is None or p>chosen_p):chosen=r;chosen_p=p
        if chosen is not None:independent[t]=chosen
    assert producer==independent
    chosen=[{'startTime':r['startTime'],'publishTime':r['publishTime'],'generation':r['generation']} for t,r in sorted(producer.items())]
    checks.append({'origin':text,'coverage':len(chosen),'target_half_hours':len(targets),'independent_asof_selection_equal':True,
                   'all_selected_published_by_origin':all(stamp(r['publishTime'])<=origin for r in chosen),'selected':chosen})
result={'status':'qualified_asof_fixture','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'forecast':describe(F),'actual':describe(A,('fuelType',)),'contracts':contracts,'asof_checks':checks,
 'receipts':receipts,'plan_sha256':hashlib.sha256((ROOT/'PROVENANCE_PROBE_PLAN.md').read_bytes()).hexdigest(),
 'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
 'no_actual_values_used_for_forecast_selection':True,'no_forecast_errors_or_model_scores_computed':True,
 'not_a_training_or_test_cohort':True,'accepted_research_directions':0,
 'limits':['official publishTime labels, not independently observed historical network receipts',
           'one issue day and one verification day, no system-level generalization evidence',
           'later settlement revisions/curtailment/fleet definition must be checked for any full experiment']}
(OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'forecast_rows':len(F),'actual_rows':len(A),
   'coverage':[x['coverage'] for x in checks],'forecast_scores_computed':False}),flush=True)
