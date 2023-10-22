import pandas as pd
import ray
from omegaconf import OmegaConf
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

from . import utils

context = ray.init(include_dashboard=True, dashboard_host='0.0.0.0')
print(context.dashboard_url)

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CustomCallbacks(DefaultCallbacks):
    def on_episode_step(self, worker, base_env, episode, **kwargs):
        # Get the last info dict for the primary agent (assuming a single agent setup)
        last_info = episode.last_info_for(episode.agent_keys[0])

        keys_to_log = ["portfolio_value", "shared_cash", "one_step_return", "position_values"]
        for key in keys_to_log:
            if key in last_info:
                episode.custom_metrics[key] = last_info[key]
                print(f"{key}: {last_info[key]}")

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_unwrapped()[0] # get reference to the env object

        ## if I cant access the attribute directly, get it from info dict??
        final_value = getattr(env, "portfolio_value", None)
        if final_value is None or final_value == 0:
            last_info = episode.last_info_for(episode.agent_keys[0])
            final_value = last_info.get("portfolio_value", None)

        # Assuming env.initial_cash is accessible directly
        initial_cash = env.initial_cash

        # Calculate the total return
        total_return = final_value - initial_cash if final_value is not None else 0

        # Log the values
        episode.custom_metrics["final_value"] = final_value
        episode.custom_metrics["total_return"] = total_return
        print(f"final_value: {final_value}")
        print(f"total_return: {total_return}")


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
        "env": "RayTradingEnv",
        "start_date": '2021-01-01',
        "end_date": '2023-01-01',
        "warmup": 24 * 7 * 4,
        "rebalance_freq": 24 * 7,
        "data_freq": 1,
        "freq_unit": "h",
        "init_cash": 100_000,
        "txn_cost": 1e-3,
    })

    data = utils.load_local_data(list(config.tickers), config.start_date, config.end_date,
                                 freq=f'{config.data_freq}{config.freq_unit}')
    close_df = pd.DataFrame({sym: data[sym].close for sym in list(config.tickers)})

    if close_df.isna().any().any():
        msg = 'Close prices contain NaNs'
        msg += f'\n{close_df[close_df.isna().any(axis=1)]}'
        raise ValueError(msg)

    # check for zeros or negative prices
    if (close_df <= 0).any().any():
        msg = 'Close prices contain zeros or negative values'
        msg += f'\n{close_df[close_df <= 0]}'
        raise ValueError(msg)

    if config.env == 'RayTradingEnv':
        from . import rayenv
        env_cls = rayenv.RayTradingEnv
    elif config.env == 'ShrinkEnv':
        from . import shrinkenv
        env_cls = shrinkenv.ShrinkEnv
    else:
        raise ValueError(f'Unknown env: {config.env}')

    algo = (
        PPOConfig()
        .callbacks(CustomCallbacks)
        .rollouts(num_envs_per_worker=4)
        .resources(num_gpus=1)
        .environment(env=env_cls, env_config={
            'open_prices': close_df.values,
            'close_prices': close_df.values,
            **config
        }).build()
    )
    print('Training')
    for i in range(1000):
        result = algo.train()
        print(pretty_print(result))
    print('done')


if __name__ == '__main__':
    main()
