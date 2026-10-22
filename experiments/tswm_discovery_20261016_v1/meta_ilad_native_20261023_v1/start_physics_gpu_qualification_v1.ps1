$ErrorActionPreference='Stop'
$r='D:\tswm_discovery_20261016_v1\meta_ilad_native_20261023_v1'
if(Test-Path -LiteralPath ($r+'\physics_gpu_execution_v1')){throw 'Existing GPU physics job; inspect terminal receipt.'}
$env:CONDA_PREFIX='E:\Anaconda\envs\persist_stable_251'
$env:PATH="$env:CONDA_PREFIX;$env:CONDA_PREFIX\Scripts;$env:CONDA_PREFIX\Library\bin;$env:SystemRoot\System32;$env:SystemRoot"
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe' -I -B -X utf8 -u ($r+'\scripts\capture_physics_gpu_v1.py')
exit $LASTEXITCODE
