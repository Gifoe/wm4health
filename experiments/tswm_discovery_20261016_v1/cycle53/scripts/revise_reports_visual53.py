"""Archive pre-QA reports and point current evidence to the corrected figure layout."""
from native53 import *
import shutil
q=R/'outputs/plot_layout_revision_v1/qualification.json'
assert json.loads(q.read_text(encoding='utf-8-sig'))['status']=='PASS_LAYOUT_REVISION_WITH_NUMERIC_ARTIFACTS_UNCHANGED'
old=R/'outputs/final_reports_v1/manifest.json';manifest=json.loads(old.read_text(encoding='utf-8-sig'))
OUT=R/'outputs/final_reports_v2';assert not OUT.exists();OUT.mkdir()
archive=[]
for item in manifest['files']:
    source=R/item['path'];assert sha(source)==item['sha256']
    target=R/'outputs/final_reports_v1/reports_before_visual_qa'/item['path'];assert not target.exists()
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    archive.append({'original_path':item['path'],'archive_path':target.relative_to(R).as_posix(),'sha256':sha(target)})
    text=source.read_text(encoding='utf-8-sig').replace('render_native53_v4.py','render_native53_v5.py').replace('repair_plots53_v3.py','qualify_plot_layout53.py')
    source.write_text(text,encoding='utf-8')
latest=P/'CURRENT_RESEARCH_STATUS_V53_FINAL.md'
target=R/'outputs/final_reports_v1/reports_before_visual_qa/CURRENT_RESEARCH_STATUS_V53_FINAL.md'
shutil.copyfile(latest,target)
latest.write_text((R/'CURRENT_STATE.md').read_text(encoding='utf-8-sig'),encoding='utf-8')
dump(OUT/'archive_receipt.json',{'utc':now(),'files':archive,'old_final_manifest_sha256':sha(old),
    'additional_archived_status_sha256':sha(target),'scope':'Figure-source references only; numerical report content unchanged'})
note=R/'VISUAL_QA_NOTE.md'
note.write_text('''# Final figure layout

Final PNG/SVG figures use render_native53_v5.py after visual inspection found cropped contrast labels and vertical-label overlap in the previous fixed layout. Contrast labels use obs (observation concat), concat (frozen-AE concat), Gate (frozen-AE Gate) and Gate + DS (directional supervision). Panel spacing and axis-label clearance were increased. Values, seed SDs, block intervals and series are unchanged.

All 84 retained forecast groups and seven additional numerical files were hash-checked unchanged during this revision. Earlier native layout/PDF failures and all partial figures remain on the server. The final export uses PNG/SVG, with no PDF claim. Qualification is in outputs/plot_layout_revision_v1/qualification.json.
''',encoding='utf-8')
dump(OUT/'manifest.json',{'utc':now(),'status':'C53_COMPLETE_EVIDENCE_REPORTS_RESEARCH_GOAL_ACTIVE','accepted_final_directions':0,
    'previous_final_manifest_sha256':sha(old),'plot_qualification_sha256':sha(q),
    'files':[{'path':name,'sha256':sha(R/name)} for name in [*[x['path'] for x in manifest['files']],'VISUAL_QA_NOTE.md']]})
print(json.dumps({'status':'CURRENT_REPORT_REFERENCES_UPDATED_AFTER_VISUAL_QA','numerical_content_changed':False}),flush=True)
