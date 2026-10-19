"""Independent stdlib CSV/time/ZIP source reconstruction; not pandas intake code."""
import sys,pathlib,json,hashlib,csv,zipfile,io,datetime,calendar
R=pathlib.Path(__file__).resolve().parents[1];ROOT=R.parent
sys.path.insert(0,str(ROOT/'cycle29/scripts'));import runtime_profile
import numpy as np
A=R/'assets';O=R/'outputs';intake=json.loads((O/'intake.json').read_text())
for spec in intake['sources']:
 assert hashlib.sha256((A/spec['file']).read_bytes()).hexdigest()==spec['sha256']
def stamp(s,fmt=None):
 dt=datetime.datetime.strptime(s,fmt) if fmt else datetime.datetime.fromisoformat(s)
 return calendar.timegm(dt.timetuple())*10**9
with (A/'ETTh1.csv').open(newline='',encoding='utf-8-sig') as f:
 rows=list(csv.DictReader(f));et=np.array([stamp(x['date']) for x in rows],dtype=np.int64);ey=np.array([float(x['OT']) for x in rows])
observed={}
for half in ['a','b']:
 with zipfile.ZipFile(A/f'mpi_roof_2016{half}.zip') as z:
  name=next(n for n in z.namelist() if n.lower().endswith('.csv'))
  with z.open(name) as f:
   for row in csv.DictReader(io.TextIOWrapper(f,encoding='latin1')):
    dt=datetime.datetime.strptime(row['Date Time'],'%d.%m.%Y %H:%M:%S')
    if dt.minute!=0:continue
    ts=calendar.timegm(dt.timetuple())*10**9;assert ts not in observed
    value=float(row['T (degC)']);observed[ts]=np.nan if value==-9999 else value
jt=np.arange(min(observed),max(observed)+3600*10**9,3600*10**9,dtype=np.int64)
jy=np.array([observed.get(int(t),np.nan) for t in jt])
records=[]
for name,t,y in [('ett',et,ey),('jena',jt,jy)]:
 saved=np.load(A/f'{name}.npz');assert np.array_equal(t,saved['timestamps']), (name,'timestamps')
 assert np.array_equal(np.isnan(y),np.isnan(saved['y'])),(name,'missing mask')
 error=float(np.nanmax(np.abs(y-saved['y'])));assert error<2e-12,(name,'signal parser difference',error)
 records.append({'data':name,'source_rows_grid':len(y),'finite':int(np.isfinite(y).sum()),'max_signal_difference':error,'absolute_parser_tolerance':2e-12,'timestamp_difference_ns':0})
out={'status':'PASS','independent_parser':'stdlib csv/zipfile/datetime, not pandas intake','naive_timestamp_representation':'integer epoch coordinate without claiming physical UTC or timezone conversion','checks':records,'source_files_sha_verified':3,'v1_failure_preserved':'Exact bitwise float parsing comparison failed; V2 separately checks exact timestamps/missingness and an explicit 2e-12 absolute signal tolerance','test_scored':False}
(O/'raw_data_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
