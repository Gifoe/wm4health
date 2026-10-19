# C40: framework scope and source qualification

This cycle follows the user's correction: the research unit is a reusable time-series world-model framework. Individual datasets are evidence carriers, not optimization targets. WorldTS is the named reference (the user wrote “worldtfs”; the earlier request links decisionintelligence/WorldTS).

Working question: can a learned state support both future trajectory simulation and correction by newly arriving observations, without sacrificing useful predictive quality or repeatedly rebuilding the entire computation? This is a candidate question, not an established unsolved problem or an accepted method.

The framework interface, if justified, would be encode(history) -> state, transition(state, available conditions) -> predictive state, update(state, new observations) -> corrected state, decode(state, target times) -> future distribution. Separate weights per dataset are allowed. Actions, zero-shot transfer, multimodality and Loop are not compulsory.

Known overlap: autoregressive probabilistic models and state-space filters already provide coherent updating; FlowNP amortizes arbitrary conditional generation; SFM uses previous predictive samples as flow sources; ForcingDAS combines conditioning modes; PathFM combines trajectory generation, cache and few-step sampling. A pipeline containing these ingredients alone is not novel. No new architecture is authorized by evidence at intake.

Immediate executable work is an ordinary baseline qualification and screen using a fixed future tail while revealing actual prefixes. It cannot by itself establish a general world-model contribution. A positive residual would require a mechanism-level attribution and subsequently diverse simulation, partial-observation and shift tests. Stop a candidate if ordinary methods remove its motivating failure at comparable information and budget.

Data intake: audit named official ETTh1 OT and Jena station T (degC), not undocumented target_0. Preserve source bytes and hashes, verify timestamps, units and missingness. Only train-normalization. Do not score TEST. This is a component transfer / diagnostic, not reproduction of authors' headline results. Prior artifacts remain untouched.

Intake state: goal ACTIVE_UNACHIEVED, accepted final directions 0. No future training outcome is presumed.
