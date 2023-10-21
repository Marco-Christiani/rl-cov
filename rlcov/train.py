import pandas as pd
import ray
from omegaconf import OmegaConf
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

import utils
from rayenv import RayTradingEnv

context = ray.init(include_dashboard=True)
print(context.dashboard_url)


def main():
    config = OmegaConf.create({
        "tickers": [
            'ADAUSD',
            'BTCUSD',
            'CRVUSD',
            'ETHUSD',
            'FTTUSD',
            'LTCUSD',
            'XRPUSD',
        ],
        "start_date": '2021-01-01',
        "end_date": '2023-01-01',
        "warmup": 24 * 7 * 4,
        "rebalance_freq": 24 * 7,
        "data_freq": 1,
        "freq_unit": "h",
        "env": {
            "init_cash": 10000,
            "txn_cost": 0.001,
        }
    })

    data = utils.load_local_data(list(config.tickers), config.start_date, config.end_date,
                                 freq=f'{config.data_freq}{config.freq_unit}')
    close_df = pd.DataFrame({sym: data[sym].close for sym in list(config.tickers)})

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=16, num_envs_per_worker=4)
        .resources(num_gpus=1)
        .environment(env=RayTradingEnv, env_config={
            'open_prices': close_df.values,
            'close_prices': close_df.values,
            **config.env
        }).build()
    )
    print('Training')
    for i in range(1000):
        result = algo.train()
        print(pretty_print(result))
    print('done')


if __name__ == '__main__':
    main()
