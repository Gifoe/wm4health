# C48 results — formal comparison pending

UTC 2026-10-17T13:19:45.026138+00:00. Frozen16-fit queue, 4 completed fits / 9000 completed-fit updates at this snapshot. Qualification24 discarded updates excluded. Actual server supervisor pid41632; staterunning. Training/evaluation pending as of this timestamp; consult live supervisor/status files. No EVAL performance or new-method claim is available.

Qualified known linear DDE maximum errors at h1/24,1/48,1/96day: [5.6675192778743444e-05, 1.7293041968069467e-05, 3.5496920468558457e-06]. Independent Heun maximum errors: [2.7755575615628914e-16, 6.106226635438361e-16, 1.1657341758564144e-15]. Neural delay gradient absolute discrepancies: [1.5356399484889741e-13, 1.9934433200278698e-13]. These are solver/gradient qualifications, not evidence of forecasting advantage.

Frozen real-process data: ETTh1 OT and Jena2016 temperature,96 observed hours to48 future hours. TRAIN-only scaling, chronological split boundaries, missing Jena hours retained and incomplete windows excluded. Reusing raw bytes is disclosed; this contract differs from C40 and its scores must not be merged into a controlled ranking.

Completed ett__fixed_dde__seed2026481: updates2000, selected250, VAL MSE0.1205108, capFalse. No TEST score.

Completed ett__gru__seed2026481: updates3000, selected1750, VAL MSE0.0954641, capFalse. No TEST score.

Completed ett__learned_dde__seed2026481: updates2000, selected250, VAL MSE0.1141349, capFalse. No TEST score.

Completed ett__node__seed2026481: updates2000, selected250, VAL MSE0.0957119, capFalse. No TEST score.

All16 selections must be sealed before test opening. Evaluation queue then performs VAL numerical/history diagnostics, locked forecasting and real-prefix continuation, followed by independent raw-source/window/NumPy-model/score verification. Real origins are temporally dependent; deterministic mean forecasts do not certify a stochastic world law.
