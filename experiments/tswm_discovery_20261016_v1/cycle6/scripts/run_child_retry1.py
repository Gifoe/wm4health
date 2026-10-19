"""Retry after a confirmed pre-training adapter error, in a separate directory."""
import runtime_profile
import os
os.environ['TSWM_CHILD_RUN']='child_native_A_r1'
from run_child_guarded import main
if __name__=='__main__':main()
