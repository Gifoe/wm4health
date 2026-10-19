"""Build a traceable CSV from the screened primary-source registry."""
import pathlib,json,csv,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];D=R/'literature-search-20261017-observation-interface';out=D/'search_manifest.json';assert not out.exists();papers=json.loads((D/'papers.json').read_text(encoding='utf-8'));assert len(papers)==5
with (D/'papers.csv').open('w',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(papers[0]));writer.writeheader();writer.writerows(papers)
names=['papers.md','papers.csv','papers.json','search-notes.md','idea-grounding.md'];manifest={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'mode':'quick','candidates_screened':9,'retained':5,'scope':'public prior-art screen, not a global novelty verdict','files':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in names},'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'accepted_final_research_directions':0};out.write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'literature_files':len(names),'retained_primary_sources':5,'accepted_final_directions':0}))
