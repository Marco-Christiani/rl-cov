---
topic: "RL for PA"
year: 2020
author: "Koker, Thomas E.; Koutmos, Dimitrios; Koutmos, Dimitrios"
title: "Cryptocurrency Trading Using Machine Learning"
journal: ""
doi: "10.3390/jrfm13080178"
---

<strong>
Koker, Thomas E., Dimitrios Koutmos, and Dimitrios Koutmos. "Cryptocurrency Trading Using Machine Learning" 13, no. 8 (August 10, 2020): 178. https://doi.org/10.3390/jrfm13080178.
</strong>

**Methodology:** Direct reinforcement (DR) learning model to make trading decisions that optimize risk-adjusted returns

**Network Architecture:** No neural network architecture is used. The DR model is based on estimating parameters of a nonlinear autoregressive model.

**Algorithms:** The DR model uses gradient ascent to optimize the Sortino ratio as the reward function. No modifications to standard RL algorithms are mentioned.

**Training and Testing Data:** Cryptocurrency price data from August 2015 to August 2019 is used (1447 data points). The first 1000 days are used for training, then 100 day test windows are simulated.

**Evaluation Metrics:** Cumulative returns, Sharpe ratio, Sortino ratio, maximum drawdown, and value-at-risk.

**Results:** The DR model outperforms buy-and-hold for risk-adjusted returns in most cryptocurrencies tested. It also reduces maximum drawdown and value-at-risk in most cases.

**Conclusions:** The DR model demonstrates viability of active cryptocurrency trading and machine learning for superior performance compared to passive approaches.

**Limitations:**

**Future Work:** Suggests integrating microstructure variables may further improve performance across more cryptocurrencies and market conditions.

