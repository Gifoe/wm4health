"""Explicitly hide CUDA devices for this CPU-only process on Windows."""
from common72_v2 import *
os.environ['CUDA_VISIBLE_DEVICES']='-1'
