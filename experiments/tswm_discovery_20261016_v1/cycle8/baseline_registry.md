# Source, model and execution registry

| Object | Status | Evidence/scope |
|---|---|---|
| Causal Chambers protocols/data metadata | Pinned/read | 0c8a77ea14bd4437b1fdd79adc1aa91ac2342569;42small source/metadata files |
| wt_walks_v1 | Downloaded/verified/audited |46,471,335compressed bytes;16walks,10wave runs,2regime runs |
| wt_test_v1 | Downloaded/verified/audited |2,610,508compressed bytes;10calibration/characterization recordings |
| Ridge ARX/polynomial | Executed ordinary control | Two fitted information arms,six validation-selected penalties each |
| HGB sample-interval predictor | Executed ordinary control | History/known commands only,150iteration maximum |
| HGB next-RPM predictor | Executed stronger existing control | Two per-sensor fits per information arm,200iteration maximum; not compute matched |
| Measured persistence | Executed | No fitted parameters; same evaluation rows/roots |
| Float32 timer diagnostic | Executed |128regime sensor streams plus calibration; source-consistent interpretation, no physical latent labels |
| TAWM/HyperNSDE/TE-CDE/TESAR-CDE | Literature/protocol only | No paper metric reproduction or modified-native execution |
| CF-JEPA/EP-IDM/Controllable-Koopman | Fresh read-only scouting | Relevant assumptions/code links; no new benchmark execution or insufficiency claim |
| Earlier main/cycles2–7 | Preserved | No completed experiment restarted or relabeled as fresh |

Native dataset software remains MIT and numeric data CC BY4.0, with notices retained. Initial metadata bundle contains41files; firmware text was added afterwards at the same pinned commit, extending SOURCE_PIN to42files. Run stage_metadata once for the original snapshot, then fetch_firmware for the documented extension; do not compare an old bundle's pin against the expanded working pin as if unchanged.

Server direct S3 was refused; public archives were fetched locally then relayed over pinned SSH, rechecked against publisher MD5 and transport SHA256, and numerically analyzed on server. No RemoteLab/hardware command, new public publication or unrelated project edit occurred.
