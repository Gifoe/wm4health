# Source-instance identity correction before diagnostic fitting

The factual shape gate stopped before any diagnostic fit or KDE training. Both released numeric CSVs have26544 rows, compatible with158 consecutive168-day simulation blocks, but only153 distinct county identifiers:148 occur once and five occur twice. No required factual numeric column has missing/nonfinite values. This is not evidence of153 versus158 different physical systems without inspecting the source blocks further.

Retain every original source row. The corrected discriminator uses the native source's consecutive168-row block as an instance, records both block index and county ID, verifies a constant county ID and chronological daily dates within each block, and checks duplicate-county blocks for exact equality versus differences. It makes no independence or seed-pairing claim for duplicate IDs. Later county-held-out evaluation must keep all blocks sharing an ID together.

The predeclared six readout features, days1..111 fitting/days112..167 diagnostic split and numerical solver remain unchanged. The native KDE still receives all158 source instances. Do not silently drop or average the repeated blocks, and do not report158 independent counties. Counterfactual targets and compartment values remain closed. The earlier failed finite/identity gate and source versions are preserved.
