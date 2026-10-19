# C48 development decision

Deprioritize the current explicit-delay route as a proposed new framework. It provides no stable advantage over the competent history-aware controls. On Jena, GRU is better than learned delay in both seeds for open-loop and updated forecasts. On ETT, learned versus GRU depends on seed; the weak scalar NODE has the lowest mean open-loop MSE among these arms. This does not rule out useful memory generally or establish NODE as a universal world model.

Adding explicit delays is not retained as the main new-method proposal. The broader state/evolution/update question remains open, but this screen supplies no necessary new mechanism.

The active separate C49 feasibility screen investigates observation-marginalized conditional stochastic-world learning versus complete-clock variational learning with a shared predictive core. Its measurement identity, scoring rule and conditional path learning are existing ingredients; a distinct useful learning interaction, competent comparisons and independent real-domain evidence remain to be established.
