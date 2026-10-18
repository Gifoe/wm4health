"""Read prior research decisions without treating broad citations as executions."""
import pathlib,json,hashlib,re,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def field(text,pattern,limit):
 lines=[re.sub(r'\s+',' ',x).strip() for x in text.splitlines() if re.search(pattern,x,re.I)]
 return '\n'.join(lines)[:limit]
rows=[]
for p in sorted(PROJECT.glob('cycle*'),key=lambda x:int(x.name[5:]) if x.name[5:].isdigit() else 9999):
 if not p.is_dir() or not p.name[5:].isdigit() or int(p.name[5:])>=71:continue
 files={}
 for name in ['final_research_direction.md','baseline_registry.md','negative_evidence.md','research_ideas.md']:
  f=p/name
  if f.exists():files[name]=dict(path=f.relative_to(PROJECT).as_posix(),sha256=sha(f),text=f.read_text(encoding='utf-8-sig'))
 final=files.get('final_research_direction.md',{}).get('text','');base=files.get('baseline_registry.md',{}).get('text','');neg=files.get('negative_evidence.md',{}).get('text','')
 rows.append(dict(cycle=int(p.name[5:]),sources={k:{kk:vv for kk,vv in v.items() if kk!='text'} for k,v in files.items()},question=field(final,r'Research Question|research.question|研究问题|RQ[0-9]?',600),decision=field(final,r'Proposed Idea|accepted|unachieved|unaccepted|拒绝|不成立|尚未|未接受|未确定|Goal|Next Experiment',1200),execution=base[:2200],negative=neg[:1600],fallback=final[:1000] if not field(final,r'Research Question|研究问题',50) else ''))
obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='HISTORICAL_COVERAGE_INDEX_NOT_NEW_EXPERIMENT',cycles=rows,cycle_count=len(rows),purpose='Prevent repeated acquisition/qualification and distinguish actual native execution from cited prior art. Entries are source excerpts; source files remain authoritative.',new_models_trained=0,accepted_final_directions=0)
dest=OUT/'HISTORICAL_COVERAGE71_V1.json'
with dest.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=True,indent=2)
print(json.dumps(dict(status=obj['status'],cycles=len(rows),sha256=sha(dest))))
