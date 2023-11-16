---
topic: "RL for PA"
year: 2018
author: "Xiong, Zhuoran; Liu, Xiao-Yang; Zhong, Shan; Yang, Hongyang; Walid, Anwar"
title: "<strong>Practical Deep Reinforcement Learning Approach for Stock Trading</strong>"
journal: "arXiv:1811.07522 [cs, q-fin, stat]"
doi: ""
---

<strong>
Xiong, Zhuoran, Xiao-Yang Liu, Shan Zhong, Hongyang Yang, and Anwar Walid. "Practical Deep Reinforcement Learning Approach for Stock Trading." arXiv:1811.07522 [Cs, q-Fin, Stat], December 1, 2018. http://arxiv.org/abs/1811.07522.
</strong>


**Methodology**: DDPG RL agent to make stock trading decisions.

**Network Architecture:** Not mentioned.

**Observation Space**: [p, h, b]: the prices of stocks, the amount of holdings of stocks, and the remaining balance.

**Action Space**: sell/hold/buy x N

**Rewards:** Raw change in portfolio value.

**Algorithms:** DDPG.

**Training and Testing Data:** Used 6 years of daily stock price data (2009-2014) for 30 Dow Jones stocks for training, 1 year (2015) for validation, and 2.75 years (2016-2018) for testing.

**Evaluation Metrics and Criteria:** Annualized return, annualized standard error, final portfolio value, and Sharpe ratio.

**Results:** DDPG strategy achieved higher annualized return (22.24%), final portfolio value ($19,791), and Sharpe ratio (1.79) compared to Dow Jones Industrial Average and min-variance portfolio allocation baseline methods.

**Conclusions:** The DDPG agent was able to learn an effective trading strategy that outperformed the baselines in maximizing return while balancing risk.

**Limitations:** This method will underperform benchmarks such as Sharpe portfolio.

**Suggestions for Future Work:** "Future work will be interesting to explore more sophisticated model [20], deal with larger scale data [21], observe intelligent behaviors [22], and incorporate prediction schemes [23]"

