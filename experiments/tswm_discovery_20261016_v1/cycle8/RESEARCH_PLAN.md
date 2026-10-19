# Cycle8: real physical clock and measurement qualification

The discovery goal remains active; no direction has been accepted. Preserve the
sealed main experiment and cycles2–7. This cycle changes the evidence domain from
low-dimensional simulation to prerecorded real wind-tunnel measurements.

## Authoritative provenance

Public Causal Chambers repository pinned to
`0c8a77ea14bd4437b1fdd79adc1aa91ac2342569`. The wt_walks_v1 and wt_test_v1 archives
match the publisher's MD5, and transport also verifies SHA256. Direct S3 access
from the server was refused; the public49MB archives were relayed through local
storage over pinned SSH and reverified on the server. All numerical experiments
run on the server. No hardware/RemoteLab interface is used.

The released variable description calls timestamps Unix time, but the actual
numbers are shifted clocks. Use only within-recording differences. The archive
contains16 actuator-walk recordings, one with1016rows and the other15 with10000,
plus10 wave runs and two320k regime runs. These verified counts supersede prose
assumptions about a fixed number of seeds. No window crosses a recording boundary.

## Competing explanations

- H1: timestamp/setting errors produce an apparent dynamics defect. Audit finite
  values, ordering, declared/recorded targets and command-to-measurement timing
  before assigning error to a learned transition.
- H2: useful elapsed-time information explains a substantial residual. Compare
  matched dynamics fits with no future time, actual future time, predicted time
  and past-only time; distinguish conditional prediction from deployable
  next-observation simulation. A new time-aware mechanism is unjustified if
  standard controls solve the problem.
- H3: tachometer reading blocks until the next pulse, making observation time
  endogenous to rotor state. **Rejected for the inspected firmware**: pulse
  interrupts maintain RPM, and getters just atomically read it. No direct
  evidence yet ties the archive to a particular flashed firmware version; this
  is a source-level falsifier of that proposed explanation, not hardware proof.
- H4: sensor-only intervention blocks show changed plant laws. Not established:
  they may change measurement scale/quantization or acquisition timing. Measured
  RPM/pressure are not privileged physical latent ground truth. Target labels
  and recorded settings are checked but no paired physical counterfactual is
  assumed.

## Current frozen control experiment

`clock_rpm_controls.py` saves a plan before fits. Train on actuator walks2–10,
validate11–12, test13–16; exclude truncated1 by inventory. All wave/regime runs
remain separate OOD groups and select no model. Ridge ARX with polynomial command
features and a standard HGB clock predictor are ordinary attribution controls,
not new methods or native WorldTS/TAWM reproductions. Report both one-step and
closed16-step autoregressive prediction, plus elapsed-time error. A16-step
episode-count horizon has varying physical duration; it is not a fixed-time
simulation certificate. Known future programmed commands are allowed; future
measured states and realized time intervals are not deployable inputs.

No numerical result alone establishes novelty. EndoClock2026 already audits
endogenous observation clocks; HyperNSDE2026 and LatentODE2019 already model
observation intensity; TAWM2025 conditions controlled dynamics on chosen time
steps; recent sensing-reliability work already separates internal and outcome
responses. A viable residual must differ from these and survive properly
specified existing baselines. Do not rename observation-intensity modeling or
ordinary time conditioning as a new mechanism.
