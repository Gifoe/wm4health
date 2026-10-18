# C86 controlled pretraining-sampler audit

Start with experimental_results.md and final_research_direction.md. This contains a sampler-method diagnostic and small paired factual-learning controls, not a reproduced foundation-model result.

Use Python3.10, Torch2.8, NumPy1.26, SciPy1.15 and Matplotlib in a task-local environment. Configure the project-parent STORAGE_LOCATION_V1.json with artifact_root/data_root/checkpoint_root/cache_root on the external disk. Put the pinned OpenFIM source tree under cache_root/cycle80/OpenFIM_cee2bb53_v1. Run sample86_v1.py, then capture86_v1.py (or train86_v1.py directly), then joint_dependence86_v1.py. The protocols were frozen at their documented stages. Fresh output/artifact/checkpoint folders are mandatory; existing evidence is never overwritten.

Run verify86_v1.py if public HTTPS is reachable from the server. The retained v2 verifier uses the independently verified public-source receipt described in PRIOR_ART_AND_REPAIRS86_V1.md, with unchanged scientific checks. Run verify_paired86_v1.py and verify_joint86_v1.py, then finalize86_v1.py. Scripts are in scripts/. All models use CPU two threads; the separate C71 campaign is not altered.

All raw populations, observations, checkpoints, predictions and logs remain on the server's external disk. Compact verified scores, source/protocol hashes, three figures and code are published. No local result-payload synchronization occurred.
