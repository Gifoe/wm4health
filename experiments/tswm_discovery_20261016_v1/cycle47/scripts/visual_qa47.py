"""Receipt for the two actual PNGs inspected in the calling conversation."""
from common47 import *
def main():
 folder=R/'outputs/figures_v1';m=read(folder/'manifest.json');expected={'clock_objective_screen.png':'ff3933567ba0a064b602d06e3313e066ae2307eba30e5eafe36b7d9ca0045b11','learned_mechanisms.png':'af332350d4791c459b90cade17a4d846c45fd9f06ca7aa282e5aa535340eb8ad'}
 for name,digest in expected.items():assert sha(folder/name)==digest,(name,'Inspection applies to different pixels')
 for ent in m['exports']:assert sha(folder/ent['path'])==ent['sha256']
 result={'status':'VISUALLY_INSPECTED_PASS','utc':now(),'inspected_png_sha256':expected,'checks':['Actual exported PNGs viewed in conversation','Axis labels, legends and captions legible','No overlapping panel titles or clipped content observed','Policy values and fixed-grid/conditional-fit limitations visible','True curves and both seed curves distinguished; full machine-readable numbers retained'],'exports':m['exports'],'source_sha256':sha(__file__)};assert not (folder/'visual_qa.json').exists();dump(folder/'visual_qa.json',result);print(json.dumps({'status':result['status'],'pngs':len(expected)}),flush=True)
if __name__=='__main__':main()
