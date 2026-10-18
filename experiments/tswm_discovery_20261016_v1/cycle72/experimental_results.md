# Executed evidence and interpretation

Native source and dataset payload hashes are in SOURCE_AUDIT72_V1.json and DATA_ACQUISITION72_V1.json. The constructed native contract test changes only CF compartment columns, leaves target/TRAIN exports unchanged, and observes those changes reaching the native test loader and whole-sequence KDE features. It accesses no actual benchmark CF values.

The factual discriminator uses days1..111 for fitting and112..167 for checking each block:8848 diagnostic points/setting. All six arms are fixed before scores and are reported below in original case-count units. They receive contemporaneous factual compartment states; these are conditional observation-decoding errors, not future forecasts or counterfactual prediction errors.

| Fixed diagnostic | Single-policy MSE | Multi-policy MSE |
|---|---:|---:|
| constant | 1562.8796 | 1255.2402 |
| stock_complement | 1602.8961 | 1362.8844 |
| minus_delta_S | 1143.9324 | 998.75072 |
| minus_delta_S_E | 1486.2259 | 1265.7647 |
| linear_levels | 4056.0702 | 4536.9887 |
| linear_levels_and_changes | 4564.7693 | 4880.1762 |

Stock complement fails to beat the constant on pooled MSE in either setting. Linear level/FIR readouts have lower median block errors in some comparisons but substantially worse pooled errors; the full per-block errors are retained. The minus-S-change control improves some aggregate errors but is far from a near-exact observation decoder. This does not establish that no nonlinear/delayed decoder could work. No model/hyperparameter was selected from CF labels, and no additional decoder was tuned after these scores.

Two exact native KDE training functions complete on all158 factual blocks using native preprocessing helper outputs. Their normalization and bandwidth selection use the source's factual training protocol. Training losses are in-sample and cannot establish generalization or mechanism consistency. Model arrays stay on the server; compact training receipts and hashes are published.
