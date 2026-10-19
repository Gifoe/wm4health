"""Critical mathematical and data-contract checks; no training regression tests."""
import json,pathlib,sys,unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from synthetic_data import split,pulse_truth,simulate as scalar_simulate
from confidence_tube import features,mean,simulate,family,regression_fit,tubes,shared_parameter_tube
from run_confidence_screen import TRUE,SIGMA

class Contracts(unittest.TestCase):
    def test_exact_pulse_truth_independent_recursion(self):
        cfg=json.loads((ROOT/'config.json').read_text());d=split('discovery',cfg)
        u=d['future'][:,:,0].astype(float);x0=d['history'][:,-1,0].astype(float)
        base=scalar_simulate(u,x0,cfg['state_coefficient'],cfg['action_coefficient'],d['noise'])
        for pos in [0,12,24]:
            alt=u.copy();alt[:,pos]+=.5
            changed=scalar_simulate(alt,x0,cfg['state_coefficient'],cfg['action_coefficient'],d['noise'])
            np.testing.assert_allclose(changed-base,pulse_truth(d,cfg,pos,.5)[:,:,0],rtol=2e-6,atol=1e-8)

    def test_splits_are_independent_and_reproducible(self):
        cfg=json.loads((ROOT/'config.json').read_text());a=split('train',cfg);b=split('validation',cfg)
        np.testing.assert_array_equal(a['history'],split('train',cfg)['history'])
        self.assertFalse(np.array_equal(a['history'][:len(b['history'])],b['history']))

    def test_likelihood_invariance_for_actions_and_rotations(self):
        rg=np.random.default_rng(902);e=rg.normal(size=(97,2));a=rg.uniform(-1,1,size=97)
        for theta in [-np.pi,-.7,.3,2.]:
            c=np.cos(theta*a);s=np.sin(theta*a)
            v=np.column_stack([c*e[:,0]-s*e[:,1],s*e[:,0]+c*e[:,1]])
            np.testing.assert_allclose(np.sum(e*e,axis=-1),np.sum(v*v,axis=-1),atol=1e-13)

    def test_exact_family_extrema_independent_dense_rollouts(self):
        rg=np.random.default_rng(903);x0=rg.normal(size=(8,2));e=rg.normal(scale=SIGMA,size=(8,19,2))
        f=simulate(TRUE,x0,e,0.);lo,hi=family(TRUE,f)
        # Dense independent simulation approximates analytic extrema from inside.
        values=np.array([simulate(TRUE,x0,e,1.,t)[:,-1,0]-f[:,-1,0] for t in np.linspace(-np.pi,np.pi,4097)])
        self.assertTrue(np.all(values>=lo-1e-12));self.assertTrue(np.all(values<=hi+1e-12))
        np.testing.assert_allclose(values.min(0),lo,atol=2e-6);np.testing.assert_allclose(values.max(0),hi,atol=2e-6)

    def test_confidence_set_boundary_is_contained(self):
        rg=np.random.default_rng(904);n=2048;x=rg.normal(size=(n,2));a=rg.uniform(-1,1,(n,1))
        y=mean(TRUE,x,a)+rg.normal(scale=SIGMA,size=(n,2));w,P=regression_fit(x,a,y)
        f=simulate(TRUE,rg.normal(size=(7,2)),rg.normal(scale=SIGMA,size=(7,32,2)),0.)
        t=tubes(w,P,SIGMA,f);lo,hi=family(w,f);s=shared_parameter_tube(w,P,SIGMA,f)
        lam,U=np.linalg.eigh(P);S=(U*np.sqrt(lam))@U.T
        # These models are selected on the EXACT confidence-set boundary,
        # independently of true coefficient coverage. Test outer-bound soundness.
        for _ in range(128):
            g=rg.normal(size=w.shape);g*=t['r']/np.linalg.norm(g)
            ww=w+S@g;ll,hh=family(ww,f)
            self.assertTrue(np.all(ll>=lo-t['paired_radius']-1e-10))
            self.assertTrue(np.all(hh<=hi+t['paired_radius']+1e-10))
            self.assertTrue(np.all(ll>=s['lo']-1e-10))
            self.assertTrue(np.all(hh<=s['hi']+1e-10))

if __name__=='__main__':unittest.main(verbosity=2)
