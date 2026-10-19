"""Read fixed Task15 headers and archive documentation, no outcome scoring."""
import pathlib,zipfile,io,json,csv,hashlib
R=pathlib.Path(__file__).resolve().parents[1];D=R/'data_intake';out=D/'headers_inventory.json'
assert not out.exists()
items=[]
def describe(z,member,chain):
    b=z.read(member)
    if member.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(b)) as q:
            for child in q.namelist():
                if child.lower().endswith('.csv'):describe(q,child,chain+[member])
    elif member.lower().endswith('.csv'):
        text=b.decode('utf-8-sig');rows=csv.reader(io.StringIO(text));header=next(rows)
        first=next(rows,None);second=next(rows,None)
        items.append({'chain':chain+[member],'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
                      'header':header,'first_row':first,'second_row':second,'lines':len(text.splitlines())})
with zipfile.ZipFile(D/'GEFCom2014_original.zip') as outer:
    readme=outer.read('GEFCom2014 Data/READ ME_V2.txt')
    (D/'official_READ_ME_V2.txt').write_bytes(readme)
    for member in ['GEFCom2014 Data/GEFCom2014-S_V2.zip','GEFCom2014 Data/GEFCom2014-W_V2.zip']:
        with zipfile.ZipFile(io.BytesIO(outer.read(member))) as z:
            for name in z.namelist():
                if '/Task 15/' in name and (name.endswith('train15.csv') or name.lower().endswith('.zip')):
                    describe(z,name,[member])
out.write_text(json.dumps({'items':items,'official_readme_sha256':hashlib.sha256(readme).hexdigest()},indent=2),encoding='utf-8')
print(json.dumps({'items':items,'readme':readme.decode('utf-8-sig',errors='replace')[:7000]}))
