# C88 baseline registry

The three native neural cores originate from ALRhub/MTS3 commit ab971c0ec43df7664c4b727ca69d5b8dd0cab5da. Native files remain unchanged. Stock/lagged MTS3 initial state SHA256 is identical: ff13f90a0f617d35081e806a5803d2c266b445f15fc5079e22632e21e5432745. Only the manager macro-action slice differs via an in-memory AST transform; the final-window skip is preserved. This is an ordinary causal comparator, not a proposed novel architecture. The added conventionalGRU is independently authored (see GRU_CONTROL_ADDENDUM88_V1.md). Native acRKN is an existing causal recurrent Kalman network with official latent dimensions/widths and its reported parameter count. ARX4 is a source-trained ordinary linear baseline; its six ridge values are selected by factual VAL only. It is not compute matched to neural training.

| Model | Parameters | Duffing MSE | Reservoir MSE |
|---|---:|---:|---:|
| stock | 126932 | 0.24129893 | 0.080687342 |
| lagged | 126932 | 0.2196222 | 0.082494544 |
| acrkn | 66272 | 0.39343813 | 0.25755426 |
| gru | 128864 | 0.45503266 | 0.029322788 |
| ordinary ARX4 | 1168 | 0.77514475 | 0.21605579 |

H changed prospectively from the source mobileRobot75 to32 to match the declared transferred task. Adam lr=.001/batch16/forecast-only NLL is our shared controlled protocol. These are not the original paper's benchmark settings or convergence claims.
