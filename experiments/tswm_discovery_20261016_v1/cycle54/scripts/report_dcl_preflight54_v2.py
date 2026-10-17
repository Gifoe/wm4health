"""Close both DCL qualification runs with correct source NumPy and total counts."""
import hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    old=OUT/'dcl_preflight_v2/summary.json';current=OUT/'dcl_preflight_v3/summary.json';amend=OUT/'DCL_NUMPY_RUNTIME_AMENDMENT54_V1.json'
    a=json.loads(old.read_text());b=json.loads(current.read_text());c=json.loads(amend.read_text())
    assert a['discarded_native_optimizer_updates']==b['discarded_native_optimizer_updates']==32
    assert b['numpy']=='1.26.4' and b['all_DCL_preflight_updates_across_runs']==64 and c['base_numpy_source_file_unchanged']
    assert a['source_files_verified_unchanged']==b['source_files_verified_unchanged'] and len(b['source_files_verified_unchanged'])==53
    text='# DCL native core qualification: latest source-pinned NumPy run\n\n'
    text+='The latest run uses actual imported NumPy1.26.4, typeguard2.13.3, jaxtyping0.2.38 and existing Torch2.8.0+cu128 in the separate project venv. It passes the numerical and GPU checks. There are64 discarded native optimizer updates in total:32 initial qualification updates with inherited NumPy2.2.6, followed by32 with the source-required NumPy1.26.4. Both runs, weights and fixtures are preserved. These are environment qualification runs, not scientific replicates. Formal DCL fits remain0.\n\n'
    text+='| Latest independent comparison | Maximum absolute error |\n|---|---:|\n'
    for row in b['numerical_checks']:text+=f"|{row['device']} loss versus NumPy|{row['loss_abs_error']:.6g}|\n|{row['device']} reference gradient versus analytic NumPy derivative|{row['reference_gradient_max_abs_error']:.6g}|\n"
    text+=f"|Linear prediction versus NumPy|{b['linear_prediction_max_abs_error']:.6g}|\n|Reciprocal-MSE switching logits versus NumPy|{b['native_switching_logit_max_abs_error']:.6g}|\n\n"
    text+=f"The native fitter executed32 updates on {b['gpu']} in {b['fit_wall_seconds']:.6f}s for the small toy fixture. All2048 sampled pairs stayed within trial boundaries and advanced one point. Actual layers remain5->90->90->30->3; total parameters{b['parameter_count']}. Raw losses are finite and weights change, but this short toy run does not test recovery, forecasting or calibration. No heldout C54 data, released paper signal files, database service or external tracking was used.\n\n"
    text+='Runtime provenance is based on actual module paths. The earlier distribution-inventory dictionary can hide duplicate base/venv versions; it is retained as historical metadata and is not used to assert the active version. The base NumPy source hash is unchanged, and the newly installed copy resolves inside .venv_dcl. Neither global packages nor the active RP/JAX environment were changed. The earlier V1 report is a frozen32-update history, superseded for current totals by this report. The unexecuted qualifier v1, the initial executed v2 and source-NumPy executed v3 are retained separately.\n\n'
    text+='All53 native core/license/configuration source hashes match before and after both runs. Service implementations and the full paper sweep are outside the qualification scope. The switching-target input demonstration confirms retrospective identification; it establishes neither a causal forecast failure nor a new contribution. The immediate next scientific decision still waits for the16 fixed C54 fits and their guarded evaluation.\n\n'
    text+='Initial qualification SHA256: '+sha(old)+'\n\nLatest qualification SHA256: '+sha(current)+'\n\nRuntime amendment SHA256: '+sha(amend)+'\n'
    target=ROOT/'DCL_PREFLIGHT_REPORT54_V2.md';assert not target.exists();target.write_text(text,encoding='utf-8');print(json.dumps({'report_sha256':sha(target),'total_discarded_updates':64,'formal_fits':0}),flush=True)
if __name__=='__main__':main()
