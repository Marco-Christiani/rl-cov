import pandas as pd
import ray
from omegaconf import OmegaConf
from ray import logger
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.tune.logger import pretty_print

from . import utils

context = ray.init(include_dashboard=True, dashboard_host='0.0.0.0')

from ray.rllib.algorithms.callbacks import DefaultCallbacks

from pprint import pformat


class CustomCallbacks(DefaultCallbacks):
    def on_episode_step(self, worker, base_env, episode: EpisodeV2, **kwargs):
        agents = episode.get_agents()
        # Get the last info dict for the primary agent (assuming a single agent setup)
        last_info = episode.last_info_for(agents[0])

        # Normalized Herfindahl–Hirschman Index
        position_pct = last_info.get("position_pct", None)
        if position_pct is not None:
            n = len(position_pct)
            episode.custom_metrics["hhi"] = (n * (position_pct ** 2).sum() - 1) / (n - 1)

        keys_to_log = ["next_portfolio_value", "shared_cash", "one_step_return"]
        for key in keys_to_log:
            if key in last_info:
                episode.custom_metrics[key] = last_info[key]
                logger.info(f"{key}: {last_info[key]}")

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        logger.info(f"type(base_env): {type(base_env)}")
        env = base_env.get_sub_environments()[episode.env_id]
        if isinstance(env, FlattenObservation):
            env = env.env
        logger.info(f"type(env): {type(env)}")
        agents = episode.get_agents()
        agent_id = agents[0]
        last_info = episode.last_info_for(agent_id)

        initial_cash = env.init_cash
        final_value = last_info.get("next_portfolio_value", None)
        total_return = (final_value - initial_cash) / initial_cash if final_value is not None else None

        position_pct = last_info.get("position_pct", None)
        n = len(position_pct)

        metrics_to_log = {
            "final_shared_cash": last_info.get("shared_cash", None),
            "final_hhi": (n * (position_pct ** 2).sum() - 1) / (n - 1),  # Normalized Herfindahl–Hirschman Index
            "final_portfolio_value": final_value,
            "final_total_return": total_return,
            "total_return": total_return
        }
        formatted_metrics = pformat(metrics_to_log, indent=4)
        logger.info(formatted_metrics)
        episode.custom_metrics = {**episode.custom_metrics, **metrics_to_log}


from ray.tune.registry import register_env
from gymnasium.wrappers import FlattenObservation


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
        "env": "ShrinkEnv",
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

    def env_creator(env_config):
        return FlattenObservation(env_cls(env_config))

    register_env(env_cls.__name__, env_creator)

    model_config = {
        "use_transformer": False,
        "use_lstm": True,
        "lstm_cell_size": 256,
        "lstm_use_prev_action": False,
        "lstm_use_prev_reward": False,
        # "max_seq_len": config['rebalance_freq'],
        "fcnet_hiddens": [256, 256]
    }
    env_config = {
        'open_prices': close_df.values,
        'close_prices': close_df.values,
        **config
    }
    algo = (
        PPOConfig()
        .callbacks(CustomCallbacks)
        .training(gamma=0.99, lr=0.0005, clip_param=0.2, model=model_config)
        .resources(num_gpus=1)
        .rollouts(num_envs_per_worker=4)
        .environment(env=env_cls.__name__, env_config=env_config)
    ).build()

    print('Training')
    for i in range(1000):
        result = algo.train()
        print(pretty_print(result))
    print('done')


if __name__ == '__main__':
    main()
