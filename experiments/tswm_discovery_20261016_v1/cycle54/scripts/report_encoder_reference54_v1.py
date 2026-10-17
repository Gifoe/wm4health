"""Summarize the completed training-only encoder attribution without scoring models."""
import hashlib, json, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'

def main():
    path=OUT/'encoder_reference_audit_v1/summary.json'
    data=json.loads(path.read_text()); assert data['status']=='COMPLETE_FROZEN_ENCODER_REFERENCE_ATTRIBUTION'
    lines=['# Encoder-reference attribution at frozen native parameter points', '',
           'Training-only follow-up to TRAINING_OBJECTIVE_FINDING54_V1.md. No optimizer updates, latent-state labels, heldout observations or model selection were used.', '',
           '| Initialization | Contract | A relative gradient difference | Recognition relative difference | Recognition cosine |',
           '|---|---|---:|---:|---:|']
    for r in data['records']:
        assert r['checkpoint_unchanged'] and not r['latent_truth_read'] and not r['heldout_read'] and r['optimizer_updates']==0
        for m in r['modes']:
            lines.append(f"|{r['run']}|em={m['em']}|{m['A']['relative_difference_norm']:.6g}|{m['recognition']['relative_difference_norm']:.6g}|{m['recognition']['cosine']:.6g}|")
    lines += ['', 'Each relative difference is ||mean balanced B32 gradient - full N200 gradient|| / ||full N200 gradient||, at unchanged parameters. The25 balanced batches use each training sequence exactly four times. This is a finite-data reference comparison, not a population-gradient estimate or an independent learning replicate.', '',
              'The existing EM contract removes the A-block discrepancy at these points. Recognition updates remain reference dependent, but their True-contract cosine is approximately .994-.995: the observed difference is predominantly magnitude, not an encoder-gradient reversal. Adam can respond differently to changes in gradient magnitude than raw SGD; these frozen values establish neither training damage nor a remedy.', '',
              'The full native False loss and every gradient leaf exactly match the attributed per-target implementation. The True recognition gradient matches an explicit held-posterior M-step to about1e-16. These controls qualify the derivative comparison; they do not validate the approximate RPM likelihood or forecast performance. Complete per-leaf errors and unchanged checkpoint/source hashes are retained in the raw JSON.', '',
              'Nondecomposable contrastive-gradient dependence is established prior art; see LITERATURE_REFERENCE_BOUNDARY54_V1.md. This audit supports a narrower parameter-block attribution, not a new method or a field-wide missing-capability claim. The matched four-fit native True learning control remains pending; no extra model/optimizer/configuration is proposed.', '',
              'Raw summary SHA256: '+hashlib.sha256(path.read_bytes()).hexdigest(), '']
    target=ROOT/'ENCODER_REFERENCE_FINDING54_V1.md'; assert not target.exists(); target.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'runtime_seconds':data['runtime_seconds'],'optimizer_updates':0,'heldout_read':False}))
if __name__=='__main__': main()
