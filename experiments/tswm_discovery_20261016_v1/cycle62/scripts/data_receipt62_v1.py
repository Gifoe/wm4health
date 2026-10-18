from native62 import *
df,p=load_data();q=read(OUT/'QUALIFICATION62_V1.json');pp=paths()
# Preserve a verified immutable source copy on the authorized data disk.
dest=pp['data_root']/'ETTh1.csv';b=p.read_bytes()
if dest.exists():assert dest.read_bytes()==b
else:
 with dest.open('xb') as f:f.write(b)
tr=q['train_boundary'];tv=q['tv_boundary']
dump(OUT/'DATA_RECEIPT62_V1.json',{'utc':utc(),'dataset':'ETTh1','source':read(ROOT/'PINNED_SOURCES62_V1.json')['ett'],'sha256':sha(dest),'git_blob':blob(b),'server_original_path':str(p),'server_verified_G_copy':str(dest),'shape':list(df.shape),'columns':df.columns.tolist(),'units':'Column units not inferred or newly verified; standardized point metrics only.','sampling':'1 hour, exact index spacing verified','boundaries':{'train':[0,tr],'validation_targets':[tr,tv],'test_targets':[tv,len(df)]},'timestamps':{'first':str(df.index[0]),'train_last':str(df.index[tr-1]),'validation_first':str(df.index[tr]),'validation_last':str(df.index[tv-1]),'test_first':str(df.index[tv]),'last':str(df.index[-1])},'previous_project_use':True,'new_confirmatory_domain':False})
print(json.dumps({'status':'G_DATA_COPY_VERIFIED','sha256':sha(dest),'rows':len(df),'columns':df.columns.tolist()}))
