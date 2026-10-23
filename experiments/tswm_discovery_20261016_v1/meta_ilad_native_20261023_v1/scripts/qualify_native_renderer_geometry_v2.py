"""Independent soft-alpha geometry, original OBJ/camera and batching checks."""
import pathlib,sys,os,json,hashlib,datetime,traceback,time
import torch,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent));from native_renderer_loader_v1 import load
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'renderer_geometry_qualification_v2';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\renderer_geometry_v2');ART.mkdir(exist_ok=False)
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def independent_alpha(triangles,h,sigma):
 image=np.empty((h,h));threshold=np.log(1/.0001-1)*sigma
 for row in range(h):
  for col in range(h):
   q=np.array([(2*col+1-h)/h,(2*(h-1-row)+1-h)/h]);prod=1.
   for tri in triangles:
    xy=tri[:,:2]
    if np.any(q>xy.max(0)+np.sqrt(threshold)) or np.any(q<xy.min(0)-np.sqrt(threshold)):continue
    w=np.linalg.solve(np.vstack((xy.T,np.ones(3))),np.r_[q,1]);inside=bool(np.all(w>0)&np.all(w<1))
    distance=[]
    for a,b in zip(xy,np.roll(xy,-1,axis=0)):
     segment=b-a;t=np.clip(np.dot(q-a,segment)/np.dot(segment,segment),0.,1.);distance.append(np.sum((q-(a+t*segment))**2))
    d=min(distance)
    if not inside and d>=threshold:continue
    logit=(1 if inside else -1)*d/sigma;p=1/(1+np.exp(-np.clip(logit,-700,700)));prod*=1-p
   image[row,col]=1-prod
 return image
try:
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 native=load(ROOT);h=17
 tri=torch.tensor([[[[-.73,-.61,3.],[.81,-.47,3.],[-.11,.86,3.]],[[-.61,.31,4.],[.63,.15,4.],[.02,-.81,4.]]]],dtype=torch.float32,device='cuda')
 tex=torch.ones(1,2,1,3,device='cuda');geometry=[]
 for sigma in [.005,.00001]:
  got=native.soft_rasterize(tri,tex,h,[0,0,0],sigma_val=sigma)[:,-1].cpu().double().numpy()[0]
  expected=independent_alpha(tri.cpu().double().numpy()[0],h,sigma);err=float(np.max(np.abs(got-expected)));assert err<2e-5,(sigma,err)
  geometry.append(dict(sigma=sigma,pixels=h*h,max_absolute_alpha_error=err,frozen_tolerance=2e-5))
 gen=native.meta_source/'pendulum/gen_data';mesh=gen/'sampledata/sphere.obj';os.chdir(gen)
 obj=native.mesh_import(mesh);lines=mesh.read_text().splitlines()
 pv=torch.tensor([[float(x) for x in l.split()[1:4]] for l in lines if l.startswith('v ')],dtype=torch.float32)
 pf=torch.tensor([[int(x.split('/')[0])-1 for x in l.split()[1:]] for l in lines if l.startswith('f ')],dtype=torch.int64)
 assert obj.vertices.shape==(482,3) and obj.faces.shape==(960,3);assert torch.equal(obj.vertices,pv) and torch.equal(obj.faces,pf)
 decoder=native.PhysicsDecoder('cuda:0')
 v=decoder.vertices.cpu().double().numpy();camera=decoder.renderer.project_to_image(decoder.renderer.transform_to_camera_frame(decoder.vertices)).cpu().double().numpy()
 physical=v.copy();physical[:,:,2]+=8.;manual=physical.copy();manual[:,:,:2]/=physical[:,:,2,None]*np.tan(np.pi/6)
 camera_err=float(np.max(np.abs(camera-manual)));assert camera_err<1e-6,camera_err
 states=torch.stack([torch.linspace(-1.,1.,19),torch.zeros(19)],dim=1).reshape(19,1,2).cuda();start=time.perf_counter()
 image=decoder(states);torch.cuda.synchronize();elapsed=time.perf_counter()-start
 assert image.shape==(19,1,32,32) and torch.isfinite(image).all() and image.min()>=0 and image.max()<=1
 chunks=torch.cat([decoder(z) for z in states.split(5)],dim=0);torch.cuda.synchronize();batch_err=float((image-chunks).abs().max());assert batch_err==0,batch_err
 original3=decoder(states[-3:]);torch.cuda.synchronize();original3err=float((original3-image[-3:]).abs().max())
 original_cross=torch.cross
 def specified_cross(a,b,dim=None,**kwargs):return original_cross(a,b,dim=-1 if dim is None else dim,**kwargs)
 try:
  torch.cross=specified_cross;corrected3=decoder(states[-3:]);torch.cuda.synchronize()
 finally:torch.cross=original_cross
 corrected3err=float((corrected3-image[-3:]).abs().max());assert original3err>.1 and corrected3err==0,(original3err,corrected3err)
 out=ART/'fixtures.npz'
 with out.open('xb') as f:np.savez_compressed(f,states=states.cpu().numpy(),images=image.cpu().numpy(),camera_vertices=camera,mesh_vertices=pv.numpy(),mesh_faces=pf.numpy(),original_batch3=original3.cpu().numpy(),explicit_axis_batch3=corrected3.cpu().numpy())
 result=dict(status='PINNED_FORWARD_RENDERER_GEOMETRY_QUALIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),soft_alpha_checks=geometry,original_obj_function_ast_sha256=native.obj_import_ast_sha256,obj_vertices_exact=True,obj_faces_exact=True,obj_container='identity container for unmodified original no-material/no-normal parser',mesh_vertices=482,mesh_faces=960,camera_max_absolute_error=camera_err,rendered_frames=19,render_seconds=elapsed,full_vs_chunked_max_error=batch_err,qualified_native_batch_sizes=[19,5,4],native_batch3_max_error=original3err,explicit_axis_batch3_max_error=corrected3err,native_batch3_failure_preserved=True,finite_and_unit_interval=True,payload_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),full_dataset_fingerprint_qualified=False,physical_model_evaluations=0,new_fits=0,local_payload_sync=False)
 dump('RESULTS_V1.json',result);native.backend.module.close();print(json.dumps(result),flush=True)
except Exception as e:
 dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
