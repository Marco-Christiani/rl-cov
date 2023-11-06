# Literature Review

<details>
<summary>
Ledoit, Olivier, and Michael Wolf. "Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection: Markowitz Meets Goldilocks." Review of Financial Studies, 2017. https://doi.org/10.5167/UZH-90273.
</summary>
**Methodology:** Authors develop a nonlinear shrinkage estimator for the covariance matrix that is tailored to Markowitz portfolio selection. The estimator has O(N) degrees of freedom and is proven to be asymptotically optimal.

**Network Architecture:** The paper does not involve deep learning or neural networks.

**Algorithms:** The authors derive an analytical formula for the optimal nonlinear shrinkage of the sample eigenvalues that minimizes the asymptotic limit of the portfolio loss function. For N dimensions we have N eigenvalues, thus N degrees of freedom.

**Training and Testing Data:** The method does not involve training data. It is evaluated on historical daily and monthly stock return data. The lookback window for estimation of the covariance matrix can be thought of as training data, in which case a variety are used in their robustness analysis.

**Evaluation Metrics and Criteria:** The proposed estimator is evaluated based on the out-of-sample standard deviation and Sharpe ratio of portfolio returns. A total of 11 approaches are compared.

**Results:** The nonlinear shrinkage estimator outperforms alternatives including linear shrinkage and sample covariance matrix in backtests.

**Conclusions:** The nonlinear shrinkage estimator with O(N) degrees of freedom is superior for portfolio selection compared to previous methods with O(1) or O(N^2) degrees of freedom.

**Acknowledged Limitations:** The method assumes no a priori knowledge about the orientation of the covariance matrix eigenvectors. Performance could potentially be further improved by incorporating such information.

**Suggestions for Future Work:** The authors suggest extending the nonlinear shrinkage approach to non-rotation equivariant situations and incorporating time-dependence in returns.
</details>

<details>
<summary>
Engle, Robert F., Olivier Ledoit, and Michael Wolf. "Large Dynamic Covariance Matrices." Journal of Business & Economic Statistics, 2019. https://doi.org/10.1080/07350015.2017.1345683.
</summary>
**Methodology:** Proposes combining two statistical methods - composite likelihood and nonlinear shrinkage - to improve estimation of the Dynamic Conditional Correlation (DCC) model for large covariance matrices.
</details>

<details>
<summary>
Mattera, Giulio, and Raffaele Mattera. "Shrinkage Estimation with Reinforcement Learning of Large Variance Matrices for Portfolio Selection." Intelligent Systems with Applications 17 (February 1, 2023): 200181. https://doi.org/10.1016/j.iswa.2023.200181.
</summary>
**Methodology:** Proposes a new shrinkage estimator for large covariance matrices based on deep reinforcement learning. The shrinkage intensity is optimized by a policy gradient agent to maximize the Sharpe ratio of the resulting minimum variance portfolio.

**Network Architecture:** Two architectures are used - a fully connected network for the Policy Gradient Agent (PGA) and a Gated Recurrent Unit (GRU) for the Recurrent Policy Gradient Agent (RPGA). The PGA has 3 hidden layers with 128, 64, and 8 nodes. The RPGA has 2 GRU layers with 256 and 128 nodes.  

**Algorithms:** Policy gradient algorithms are used to learn the optimal policy for selecting the shrinkage intensity. The PGA uses SGD with momentum while the RPGA uses Adam. The reward is the portfolio Sharpe ratio.

**Training and Testing Data:** 200 industry portfolio monthly returns from 1963-2022 (T=706 observations). Rolling window cross-validation is used with L=36 or 72 months for training and the rest for out-of-sample testing.

**Evaluation Metrics:** Out-of-sample Sharpe ratio and value-at-risk (VaR). Statistical tests are used to compare Sharpe ratios.

**Results:** The RPGA significantly outperforms existing methods, achieving a Sharpe ratio of 0.69 with L=36 vs 0.27-0.28 for others. It also has lower VaR. With L=72, RPGA Sharpe is 0.61 vs 0.28-0.33 for others.

**Conclusions:** The proposed RPGA shrinkage approach provides superior out-of-sample performance for minimum variance portfolios in high dimensions.

**Limitations:** The methods are demonstrated on a single dataset. Computational complexity and training time are not analyzed.

**Future Work:** Apply the framework to other covariance-based analyses and datasets. Consider computational optimizations.
</details>

<details>
<summary>
Wu, Mu-En, Jia-Hao Syu, Jerry Chun-Wei Lin, and Jan-Ming Ho. "Portfolio Management System in Equity Market Neutral Using Reinforcement Learning." Applied Intelligence 51, no. 11 (November 1, 2021): 8119–31. https://doi.org/10.1007/s10489-021-02262-0.
</summary>

**RL Allocation Variant:** asset weight assignment

**Reward functions:** Return, Sharpe

**Performance metrics:** Return, Sharpe, MDD, Profit Factor

**Methodology:** Equity market neutral portfolio constructed by training one long and one short RL model.

**Features:** OHLC

**Network Architecture:** Two neural network architectures - a CNN and an RNN. The CNN uses convolutional layers, dense layers, and a softmax output layer. The RNN uses an LSTM layer followed by dense and softmax layers. Details like number of layers, neurons, etc are provided in Tables 1 and 2.    

**Algorithms:** No specifics. The CNN and RNN serving as the policy networks in the RL framework. The paper also proposes a novel reward function based on the Sharpe ratio.

**Train/Test Data:** The dataset consists of daily OHLC stock price data. The TW50 stock dataset from Aug 2015 - Jul 2017 is used for training, and Aug 2017 - Jul 2019 for testing.

**Evaluation Metrics:** Total return, Sharpe ratio, maximum drawdown, and profit factor are used to evaluate the performance.

**Results:** The proposed Sharpe ratio reward function outperforms the return-based reward, giving 39% higher returns and 13.7% lower drawdown. The CNN model outperforms RNN in returns and Sharpe ratio. The PMS outperforms benchmarks on TW50 and traditional stock datasets.

**Conclusions:** The PMS with CNN and novel Sharpe ratio reward is an effective portfolio management system with good profitability and low risk. It can support decision making for resource allocation in stock trading.

**Limitations:** No explicit limitations are acknowledged, but the performance on the financial dataset was inferior to benchmarks.

**Future Work:** No concrete future work directions are suggested.
</details>
