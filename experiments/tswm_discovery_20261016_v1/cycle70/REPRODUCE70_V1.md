# Reproduction scope

Use the pinned source revision and exact15 blob hashes in SOURCE_SELECTION70_V1.json; do not alter upstream code. Keep an isolated environment matching QUALIFICATION70_V1.json. CPU JAXx64,4-thread process affinity. Configure sibling projectSTORAGE_LOCATION_V1.json to place artifact_root on authorizedGstorage. Only authored source/control files are staged from local; no data/model/result files are synchronized locally.

For a clean run use fresh output and artifact directories, as writes are exclusive. Run qualify70_v1.py then replay70_v1.py via their capture/hiddenPowerShell launchers. Replay requires a closed qualification exit0 and every raw archive hash. No network or dataset download is needed for the scientific numerical worker. Qualification/replay do not train RP-GSSM's empirical-RPM objective; they exercise native controlled Gaussian chains and a known Gaussian likelihood. Read the named adapter/privilege boundaries before treating any result as a baseline.

