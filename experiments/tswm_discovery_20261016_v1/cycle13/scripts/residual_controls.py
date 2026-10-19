"""Existing seasonal residual forecasting as an ordinary control, not a method."""
import runtime_profile
import datetime,hashlib,json,pathlib,sys,traceback
import acquisition_controls as base

ROOT=pathlib.Path(__file__).resolve().parents[1]
OriginalGRU=base.OrdinaryGRU

class ResidualGRU(OriginalGRU):
    def forward(self,x,fc):
        return x[:,-base.H:,2]+super().forward(x,fc)

def main():
    base.OUT=ROOT/'outputs/residual_controls';base.OUT.mkdir(parents=True,exist_ok=True)
    assert not (base.OUT/'results.json').exists()
    variant={'family':'ordinary seasonal residual GRU','source_loop':'scripts/acquisition_controls.py',
        'source_loop_sha256':hashlib.sha256(pathlib.Path(base.__file__).read_bytes()).hexdigest(),
        'wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'followup_plan_sha256':hashlib.sha256((ROOT/'RESIDUAL_CONTROL_PLAN.md').read_bytes()).hexdigest(),
        'skip':'last24forward-filled context values; observed inputs only','additional_parameters':0,
        'test_metrics_used_to_design_followup':False,'new_method_claimed':False,
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (base.OUT/'variant.json').write_text(json.dumps(variant,indent=2),encoding='utf-8')
    base.OrdinaryGRU=ResidualGRU
    class Tee:
        def __init__(self,original,log):self.original=original;self.log=log
        def write(self,value):self.original.write(value);self.log.write(value)
        def flush(self):self.original.flush();self.log.flush()
    with (base.OUT/'stdout.log').open('x',encoding='utf-8') as log:
        sys.stdout=Tee(sys.__stdout__,log)
        try:base.main()
        except BaseException:
            with (base.OUT/'stderr.log').open('a',encoding='utf-8') as err:traceback.print_exc(file=err)
            raise
        finally:sys.stdout.flush();sys.stdout=sys.__stdout__

if __name__=='__main__':main()
