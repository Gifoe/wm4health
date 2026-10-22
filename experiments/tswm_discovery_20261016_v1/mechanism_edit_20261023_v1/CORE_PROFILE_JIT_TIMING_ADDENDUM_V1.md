# Three discarded calls for meaningful JIT timing, before recovery launch

The first call starts with an empty Ninjax parameter tree; the next call has an initialized parameter/optimizer tree and may trigger another JIT trace. Therefore two calls cannot reliably identify steady execution cost. Before launching recovery, extend its limit to three discarded calls on the same TRAIN fixture, with a Python trace counter in the outer pure-step wrapper. The third-call timing is labeled warm only if the counter does not increase on that call. The full native computation and original model classes are unchanged.

This adds one qualification call, no scientific fit or tuning. The earlier failed call remains counted separately, yielding at most four total discarded optimizer calls if recovery completes. The600-second process-tree cap remains. Worker v2 was uploaded but not launched; v3 is the first recovery launch.
