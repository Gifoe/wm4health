$ErrorActionPreference='Stop'
$r='D:\tswm_discovery_20261016_v1\meta_ilad_native_20261023_v1'
if(Test-Path -LiteralPath ($r+'\renderer_kernel_v1_execution')){throw 'Existing kernel job; inspect terminal receipt.'}
$env:CONDA_PREFIX='E:\Anaconda\envs\persist_stable_251'
$env:PATH="$env:CONDA_PREFIX;$env:CONDA_PREFIX\Scripts;$env:CONDA_PREFIX\Library\bin;$env:SystemRoot\System32;$env:SystemRoot"
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:\tswm_discovery_20261016_v1\.venv\Scripts\python.exe' -I -B -X utf8 -u ($r+'\scripts\capture_renderer_stage_v1.py') kernel_v1
exit $LASTEXITCODE
