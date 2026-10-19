"""Supplement to the preserved V2 probe: timing only, no resampling or scores."""
import pathlib,json,datetime,collections,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
P=ROOT/'outputs/elexon_probe_v2'
dst=ROOT/'outputs/cadence_audit_v1.json'
assert not dst.exists(),'Preserve existing audit'
def read(name):return json.loads((P/name).read_text(encoding='utf-8'))
def stamp(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
F=read('windfor_20260101.json')['data'];A=read('fuelhh_wind_20260102.json')['data'];R=read('result.json')
f=sorted({stamp(x['startTime']) for x in F});a=sorted({stamp(x['startTime']) for x in A})
fg=collections.Counter(int((y-x).total_seconds()) for x,y in zip(f,f[1:]))
ag=collections.Counter(int((y-x).total_seconds()) for x,y in zip(a,a[1:]))
assert fg=={3600:72} and ag=={1800:47}
assert all(x.minute==0 and x.second==0 for x in f)
checks=[]
for item in R['asof_checks']:
    origin=stamp(item['origin'])
    hourly=[t for t in a if t.minute==0 and t>origin]
    selected=[stamp(x['startTime']) for x in item['selected']]
    assert selected==hourly
    checks.append({'origin':item['origin'],'eligible_hourly_timestamps':len(hourly),
        'selected_timestamps':len(selected),'all_eligible_hourly_timestamps_covered':True,
        'first':selected[0].isoformat(),'last':selected[-1].isoformat()})
result={'status':'native_hourly_forecast_cadence_verified',
 'forecast_unique_timestamps':len(f),'forecast_adjacent_seconds':dict(fg),
 'outturn_unique_timestamps':len(a),'outturn_adjacent_seconds':dict(ag),'checks':checks,
 'interpretation':'24 of 48 half-hourly outturn timestamps are all 24 native hourly forecast timestamps; this is not a 12-hour horizon or evidence of missing hourly predictions.',
 'official_native_cadence_source':'https://bmrs.elexon.co.uk/wind-generation',
 'no_interpolation_or_aggregation_or_forecast_scoring':True,
 'limits':['Coincident timestamps do not establish equivalence of interval-average and point-value semantics.',
           'A full experiment must freeze a declared verification target and interval convention.'],
 'probe_result_sha256':hashlib.sha256((P/'result.json').read_bytes()).hexdigest(),
 'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dst.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
