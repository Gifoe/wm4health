"""Verify staged clean wrapper transformations and raw-cache qualification only."""
from native53 import *
import argparse
import ast

parser=argparse.ArgumentParser();parser.add_argument('--stage-root',type=pathlib.Path,required=True);args=parser.parse_args()
staged=args.stage_root/'cycle53'
stage=json.loads((staged/'outputs/FRESH_STAGE.json').read_text())
assert stage['status']=='STAGED_NOT_YET_EXECUTED' and stage['history_receipts_fabricated'] is False
assert stage['launcher_sha256']==sha(R/'scripts/fresh_reproduce53_v2.py')
assert stage['source_pin_sha256']==sha(R/'outputs/native_probe_v2/source_manifest.json')
assert stage['required_data_pin_sha256']==sha(R/'required_data_manifest_v1.json')
assert not (staged/'outputs/FRESH_RUN_STATE.json').exists()
syntax=0
for name,expected in stage['payload_sha256'].items():
    path=staged/name;assert sha(path)==expected,name
    if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'));syntax+=1
original=ast.parse((R/'scripts/train_native53_v7.py').read_text(encoding='utf-8'))
class Remove(ast.NodeTransformer):
    count=0
    def visit_If(self,node):
        condition=node.test
        if (isinstance(condition,ast.Compare) and isinstance(condition.left,ast.Name) and condition.left.id=='tag'
            and len(condition.ops)==1 and isinstance(condition.ops[0],ast.Eq) and len(condition.comparators)==1
            and isinstance(condition.comparators[0],ast.Constant) and condition.comparators[0].value in
            ['PatchTST__ae_concat__seed0','PatchTST__obs__seed3']):
            self.count+=1;return None
        return self.generic_visit(node)
transform=Remove();expected=transform.visit(original);assert transform.count==2
actual=ast.parse((staged/'scripts/train_clean53.py').read_text(encoding='utf-8'))
assert ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False)
cache=json.loads((staged/'outputs/input_cache_qualification_v1.json').read_text())
assert cache['status']=='PASS_EXISTING_NATIVE_CACHE_EQUALS_PINNED_RAW_INPUTS'
assert len(cache['existing_caches_checked'])==12
fresh_art=pathlib.Path(stage['storage_config']['artifact_root'])/'cycle53/native_v1'
assert not fresh_art.exists()
out=R/'outputs/FRESH_ENTRY_PREFLIGHT_V1.json';assert not out.exists()
dump(out,{'utc':now(),'status':'PASS_FRESH_STAGING_AST_AND_RAW_CACHE_ONLY','generated_python_files_parsed':syntax,
    'historical_metadata_checks_removed':2,'training_ast_otherwise_identical':True,
    'required_original_xlsx_files':26,'existing_native_cache_frames_equal_to_raw':12,
    'fresh_formal_training_updates':0,'fresh_gpu_model_probe_executed':False,'full_fresh_40_fit_pipeline_executed':False,
    'launcher_sha256':stage['launcher_sha256'],'source_pin_sha256':stage['source_pin_sha256'],
    'required_data_manifest_sha256':stage['required_data_pin_sha256'],'generated_payload_sha256':stage['payload_sha256'],
    'raw_cache_receipt_sha256':sha(staged/'outputs/input_cache_qualification_v1.json'),
    'scope':'Staging, exact AST transformation, published raw input identities, all 12 existing Parquet frames against native raw-XLSX reconstruction; not an end-to-end fresh training certification.'})
print(out.read_text(),flush=True)
