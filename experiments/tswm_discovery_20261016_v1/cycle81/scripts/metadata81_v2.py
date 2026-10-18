"""Post-opening contract audit; metadata only, no selection or fitting changes."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common81_v1 import *
import ast
def main():
 cohort=read(OUT/'COHORT81_V1.json');asset=next(a for a in read(PROJECT/'cycle80/outputs/ASSET_MANIFEST80_V1.json')['assets'] if a['relative'].endswith('strogatz_extended.json'));assert sha(asset['path'])==asset['sha256'];db=read(asset['path']);rows=[]
 for sid in cohort['development']:
  e=next(e for e in db if e['id']==sid);names=sorted({node.id for expr in e['eq'].split('|') for node in ast.walk(ast.parse(expr.strip().replace('^','**'),mode='eval')) if isinstance(node,ast.Name)});raw=[j for j,s in enumerate(e['solutions'][0]) if s.get('success',True)];row=dict(system=sid,equation_metadata=e['eq'],dimension=e['dim'],parameter_set=e['consts'][0] if e['consts'] else [],expression_identifiers=names,explicit_time_identifier='t' in names,successful_raw_solution_indices=raw,used_raw_solution_indices=raw[:2]);assert not row['explicit_time_identifier'];rows.append(row)
 dump(OUT/'METADATA_CONTRACT81_V1.json',dict(utc=utc(),status='NO_EXPLICIT_TIME_IDENTIFIER_IN_DEVELOPMENT_EQUATIONS',post_opening_metadata_audit=True,changes_to_cohort_or_estimator=False,rows=rows))
 print(json.dumps(read(OUT/'METADATA_CONTRACT81_V1.json')),flush=True)
if __name__=='__main__':main()
