# Scaling scope before canonical model execution

Pool the first80% of each original TRAIN record within each physical system to fit one fixed u mean/SD and one fixed y mean/SD for that system. Do not normalize each F16 amplitude record separately: doing so would erase physically relevant amplitude differences. No window RevIN or test statistics are used. Other rules in DATA_PROTOCOL74_V1.md are unchanged. This clarification precedes numeric preprocessing and all new model scores.
