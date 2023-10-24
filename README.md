RL for covariance shrinkage factor estimation.

[(Mattera 2023) Shrinkage estimation with reinforcement learning of large variance matrices for portfolio selection](https://doi.org/10.1016/j.iswa.2023.200181)

[(Wu et al. 2021) Portfolio management system in equity market neutral using reinforcement learning](https://doi.org/10.1007/s10489-021-02262-0)

[(Lu et al. 2022) Improved Estimation of the Covariance Matrix using Reinforcement Learning](https://dx.doi.org/10.2139/ssrn.4081502)



## Tasks

- Currently have an environment, but it only uses the reweight prices, not all
  the prices in between
    - This is good for sim speed but the obs space is too small
    - Plan is to wrap this environment:
        - Wrapper contains all prices
        - Calls wrapped sim, passing only the prices for reweight periods
    - This gives us access to larger obs space
        - Could calculate a returns series to calculate better rewards
- Using default ray model which is fine for testing but need to implement multi
  head model one env is done.
- spock configs
- integrate the time utils for mixed unit handling that I wrote

## Notes

## Targeted Journals

- Journal of Financial and Quantitative Analysis
- Journal of Empirical Finance
- Quantitative Finance
- The Journal of Finance and Data Science
- We can also consider AI journals, e.g., 'Expert Systems with Applications'.

### Metrics

- [ ] Concentration ratio
- [x] Normalized Herfindahl index

## Log

I just have to wrap an existing env:

- Action space for me is value of shrinkage factor
- Use shrinkage factor to calculate covariance matrix
- Get new portfolio weights
- Pass as actions to super().step() method

FinRL [env_portfolio.py](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_portfolio_allocation/env_portfolio.py)

Has problems:

- transaction cost not used
- I would have to modify the existing env, it expects the output of an RL policy
  network so its doing softmax normalization etc.

Reference portfolio envs:

- [env_portfolio.py](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_portfolio_allocation/env_portfolio.py)
- [env_stocktrading_np.py](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading_np.py)

```
(RolloutWorker pid=586260) For your custom (single agent) gym.Env classes:                                                                                                                                            │
(RolloutWorker pid=586260) 3.1) Either wrap your old Env class via the provided `from gymnasium.wrappers import                                                                                                       │
(RolloutWorker pid=586260)      EnvCompatibility` wrapper class.                                                                                                                                                      │
(RolloutWorker pid=586260) 3.2) Alternatively to 3.1:                                                                                                                                                                 │
(RolloutWorker pid=586260)  - Change your `reset()` method to have the call signature 'def reset(self, *,                                                                                                             │
(RolloutWorker pid=586260)    seed=None, options=None)'                                                                                                                                                               │
(RolloutWorker pid=586260)  - Return an additional info dict (empty dict should be fine) from your `reset()`                                                                                                          │
(RolloutWorker pid=586260)    method.                                                                                                                                                                                 │
(RolloutWorker pid=586260)  - Return an additional `truncated` flag from your `step()` method (between `done` and                                                                                                     │
(RolloutWorker pid=586260)    `info`). This flag should indicate, whether the episode was terminated prematurely                                                                                                      │
(RolloutWorker pid=586260)    due to some time constraint or other kind of horizon setting.                                                                                                                           │
```
