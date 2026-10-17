"""Qualify decorator-free official mathematical bodies; this is not the Diffrax trainer."""
from common59 import *
import ast,copy,types,numpy as np
os.environ.setdefault('JAX_PLATFORM_NAME','cpu')
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from scipy.integrate import solve_ivp

def extract(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[copy.deepcopy(n) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    for n in nodes:n.decorator_list=[]
    module=ast.Module(body=nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path), 'exec'),namespace)
    return {name:namespace[name] for name in names}

def main():
    source=read(OUT/'DOWNLOAD59_V2.json')
    for f in source['files']:assert sha(f['server_path'])==f['sha256']
    softpath=source_root()/'odesolver/utils/softjax_fct.py'
    soft=types.SimpleNamespace(**extract(softpath,['maximum','minimum','clip','penalty','abs'],{'jnp':jnp,'nn':jax.nn}))
    linear=extract(source_root()/'odemodel/eqn_linear.py',['get_ode'],{'jnp':jnp})['get_ode']
    nonlinear=extract(source_root()/'odemodel/eqn_nonlinear.py',['get_ode'],{'jnp':jnp,'softjax_fct':soft})['get_ode']
    # Independent NumPy expression, including the source's smooth clipping convention.
    def npclip(x,lo,hi,s):return lo+s*np.logaddexp(0,(x-lo)/s)+hi-s*np.logaddexp(0,(hi-x)/s)-x
    def numpy_nonlinear(H,u,c,p):
        a=H*np.tanh(H/c['abs_soft'])+c['abs_offset']
        mu=npclip(p['mu_sat']/(1+(a/p['a_sat'])**p['p_sat']),c['mu_min'],c['mu_max'],c['mu_soft'])
        r=npclip(p['r_mul']*(1+a/p['r_scl']),c['r_min'],c['r_max'],c['r_soft'])
        return (u-r*H)/(4*np.pi*1e-7*mu)
    c={'r_min':1.,'r_max':1000.,'r_soft':1.,'mu_min':100.,'mu_max':10000.,'mu_soft':100.,'abs_soft':.1,'abs_offset':.1}
    rng=np.random.default_rng(5901);errors=[]
    for _ in range(32):
        H=rng.uniform(-500,500,2);u=float(rng.uniform(-1e5,1e5))
        pl={'r_lin':rng.uniform(25,100,2),'mu_lin':rng.uniform(2000,5000,2)}
        l=np.asarray(linear(0.,jnp.asarray(H),{}, {k:jnp.asarray(v) for k,v in pl.items()},lambda _:jnp.asarray(u)))
        lr=(u-pl['r_lin']*H)/(4*np.pi*1e-7*pl['mu_lin'])
        pn={'r_mul':rng.uniform(1,350,2),'r_scl':rng.uniform(5,75,2),'a_sat':rng.uniform(10,150,2),'mu_sat':rng.uniform(2000,5000,2),'p_sat':rng.uniform(1,5,2)}
        n=np.asarray(nonlinear(0.,jnp.asarray(H),c,{k:jnp.asarray(v) for k,v in pn.items()},lambda _:jnp.asarray(u)))
        nr=numpy_nonlinear(H,u,c,pn)
        errors.append({'linear_scaled_max':float(np.max(abs(l-lr)/np.maximum(1,abs(lr)))),'nonlinear_scaled_max':float(np.max(abs(n-nr)/np.maximum(1,abs(nr))))})
    max_derivative=max(max(e.values()) for e in errors);assert max_derivative<1e-11
    p={'r_lin':jnp.array([50.]),'mu_lin':jnp.array([3000.])};u=12000.;h0=35.;ts=np.linspace(0,2e-4,101)
    fun=lambda t,y:np.asarray(linear(t,jnp.asarray(y),{},p,lambda _:jnp.asarray(u)))
    sol=solve_ivp(fun,(ts[0],ts[-1]),[h0],t_eval=ts,method='DOP853',rtol=1e-10,atol=1e-11)
    rate=50/(4*np.pi*1e-7*3000);truth=u/50+(h0-u/50)*np.exp(-rate*ts)
    error=float(np.max(abs(sol.y[0]-truth)));assert sol.success and error<1e-6
    result={'utc':utc(),'status':'PASS_EXTRACTED_OFFICIAL_EQUATION_CORES_AND_LINEAR_ANALYTIC_SOLVER','source_commit':source['source_commit'],'derivative_cases':32,'max_scaled_derivative_discrepancy':max_derivative,'linear_analytic_max_absolute_error':error,'jax_version':jax.__version__,'jax_x64':bool(jax.config.jax_enable_x64),'source_files_unchanged':True,'qualifier_sha256':sha(__file__),'scope':'Exact get_ode AST bodies with decorators removed and original softjax functions; independent NumPy math and SciPy DOP853 analytic check. Equinox/Diffrax/Optimistix unavailable, no import or training of the original complete model/Trainer. No real-data fit, no new method claim.','overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    for f in source['files']:assert sha(f['server_path'])==f['sha256']
    dump(OUT/'CORE_QUALIFICATION59_V1.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
