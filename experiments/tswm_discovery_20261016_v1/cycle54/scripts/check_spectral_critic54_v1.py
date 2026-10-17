"""Analytic spectral-critic semantics fixture, not a trained SelfTICA reproduction."""
import datetime,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def objective(P,C,A):return float(np.trace(P@C@P.T@C)-2*np.trace(P@C@A.T))
def main():
    path=OUT/'SPECTRAL_CRITIC_SEMANTICS54_V1.json';assert not path.exists()
    A=np.diag([.9,.6]);C=np.diag([4.,.25]);Q=C-A@C@A.T;P=np.linalg.solve(C,A);converted=C@P
    assert np.min(np.linalg.eigvalsh(Q))>0 and max(abs(np.linalg.eigvals(A)))<1
    gradient=2*C@P@C-2*A@C;hessian=2*np.kron(C,C)
    assert np.max(abs(gradient))<1e-12 and np.min(np.linalg.eigvalsh(hessian))>0
    fd=np.zeros_like(P);step=1e-5
    for i in range(2):
        for j in range(2):
            delta=np.zeros_like(P);delta[i,j]=step;fd[i,j]=(objective(P+delta,C,A)-objective(P-delta,C,A))/(2*step)
    assert np.max(abs(fd-gradient))<1e-9
    conversion_error=float(np.max(abs(converted-A)));assert conversion_error<1e-14
    assert max(abs(np.linalg.eigvals(P)))>1
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_ANALYTIC_FIXED_FEATURE_SPECTRAL_CRITIC_SEMANTICS','true_A':A.tolist(),'stationary_covariance':C.tolist(),'positive_process_covariance':Q.tolist(),'fixed_feature_optimal_critic':P.tolist(),'covariance_converted_critic':converted.tolist(),'true_transition_spectral_radius':float(max(abs(np.linalg.eigvals(A)))),'raw_critic_spectral_radius':float(max(abs(np.linalg.eigvals(P)))),'conversion_max_abs_error':conversion_error,'analytic_gradient_max_abs':float(np.max(abs(gradient))),'finite_difference_gradient_max_abs_error':float(np.max(abs(fd-gradient))),'hessian_min_eigenvalue':float(np.min(np.linalg.eigvalsh(hessian))),'objective_at_optimal_critic':objective(P,C,A),'objective_at_naive_true_A_as_critic':objective(A,C,A),'native_mlcolvar_source_pin':'3a2549ffa63e47b30c390eebc84bed588c7d3868','paper_version':'2606.15495v3','interpretation':'This is the unregularized population bilinear L2 objective with fixed centered features. The scorer coefficient is not automatically the physical conditional-mean transition. Ordinary covariance conversion is known algebra, not a new method.','formal_data_reads':0,'C54_heldout_reads':0,'optimizer_updates':0,'SelfTICA_executed_fits':0,'accepted_directions':0,'script_sha256':sha(__file__),'review_note_sha256':sha(ROOT/'INFERENCE_AND_SPECTRAL_BOUNDARY54_V1.md')}
    path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
