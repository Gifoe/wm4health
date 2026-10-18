# Split scope before first model execution

Trajectory rows are disjoint; graph labels are not guaranteed disjoint across these original random simulations. Do not claim held-out graph-composition generalization from this baseline development split. The later mechanism experiment, if justified, needs a separately frozen structural split/independent data and matched controls. No reserved EVAL forecast or score is produced here.

The batch iterator is an explicit NumPy-shuffled stream of the unchanged native dataset objects, with its own recorded RNG state. It retains the source batching distribution while avoiding modern PyTorch's CPU-DataLoader-generator conflict with the source globalCUDA default tensor type. No gradient, acceptance rule, proposal, module forward or model loss body is altered. Checkpoint selection is on the fixed every50-epoch-plus-final grid.

Data acquisition switched to direct server HTTPS after the first local relay's source/server checksum disagreed. The failed payload remains untouched. The first direct loader failed on a removed private NumPy header API; a separately versioned loader uses the public version1 header reader and a fresh output directory. All four derived payloads then passed in-memory-versus-file SHA256. These are staging/API incidents, not scientific results; no original full-state dataset or result file was written locally.
