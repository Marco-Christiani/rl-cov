RL for covariance shrinkage factor estimation.

[Literature Review](research/LiteratureReview.md)

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
- integrate the time utils for mixed unit handling that I wrote
- Custom model selects action directly, use DDPG
  - Rllib default for PPO does odd sampling that is typically out of bounds
  - **Action space should be symmetric around 0, need to fix this**
- Implement minimum reweight delta (i.e. change in 1% shouldnt trigger trades)
- Feature engineering
- Use minvar weights, GMV, risk ratio measurment (ideally close to 1)

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

## Training methodology

[Curriculum learning](https://github.com/ray-project/ray/blob/master/rllib/examples/curriculum_learning.py)

[Evaluation callback](https://github.com/ray-project/ray/blob/master/rllib/examples/parallel_evaluation_and_training.py)

[Load and eval w/ Tune training](https://github.com/ray-project/ray/blob/master/rllib/examples/sb2rllib_rllib_example.py)

## Softwex
