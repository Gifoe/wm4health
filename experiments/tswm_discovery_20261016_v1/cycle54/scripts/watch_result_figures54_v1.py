"""Wait for the existing risk analysis, then render immutable result figures once."""
import datetime, json, os, pathlib, subprocess, sys, time, traceback
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / 'outputs'
sys.stdout = (OUT / 'watch_result_figures54_v1.stdout.log').open('x', encoding='utf-8', buffering=1)
sys.stderr = (OUT / 'watch_result_figures54_v1.stderr.log').open('x', encoding='utf-8', buffering=1)

def state(status, **kw):
    value = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid': os.getpid(), 'status': status, **kw}
    (OUT / 'watch_result_figures54_v1.json').write_text(json.dumps(value, indent=2), encoding='utf-8')

try:
    state('WAITING_FOR_EXISTING_RISK_INTERPRETATION_NO_TRAINING_OR_SELECTION')
    while True:
        marker = OUT / 'watch_causal_risk54_v1.json'
        if marker.exists():
            try:
                risk_state = json.loads(marker.read_text())
            except json.JSONDecodeError:
                risk_state = {}
            if risk_state.get('status') == 'FAILED_PRESERVED':
                raise RuntimeError('Existing risk interpretation failed; preserve and diagnose')
        result = OUT / 'causal_risk_interpretation_v1/summary.json'
        if result.exists():
            try:
                ready = json.loads(result.read_text()).get('status') == 'COMPLETE_READ_ONLY_CAUSAL_RISK_INTERPRETATION_OF_ALL16_FROZEN_FITS'
            except json.JSONDecodeError:
                ready = False
            if ready:
                break
        time.sleep(30)
    state('RENDERING_FROM_COMPLETED_FROZEN_RESULTS')
    with (OUT / 'plot_results54_v1.stdout.log').open('xb') as stdout, (OUT / 'plot_results54_v1.stderr.log').open('xb') as stderr:
        subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'scripts/plot_results54_v1.py')], stdout=stdout, stderr=stderr, check=True)
    state('COMPLETE_RENDER_PENDING_ACTUAL_IMAGE_INSPECTION', optimizer_updates=0, model_selection=False)
except Exception:
    state('FAILED_PRESERVED', traceback=traceback.format_exc())
    raise
