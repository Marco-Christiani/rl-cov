from pprint import pformat

import hydra
import numpy as np
import pandas as pd
import ray
from gymnasium.wrappers import FlattenObservation
from omegaconf import DictConfig
from omegaconf import OmegaConf
from ray import logger
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.tune.logger import pretty_print

from rlcov import utils
from rlcov.envs import get_wrapped_env

context = ray.init(include_dashboard=True, dashboard_host='0.0.0.0')

OmegaConf.register_new_resolver("eval", lambda s: eval(s))


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

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        env = base_env.get_sub_environments()[episode.env_id]
        if isinstance(env, FlattenObservation):
            env = env.env
        agents = episode.get_agents()
        agent_id = agents[0]
        last_info = episode.last_info_for(agent_id)

        initial_cash = env.init_cash
        final_value = last_info.get("next_portfolio_value", None)
        total_return = (final_value - initial_cash) / initial_cash if final_value is not None else None

        position_pct = last_info.get("position_pct", None)
        n = len(position_pct)

        metrics_to_log = {
            f'final_{k}': v
            for k, v in last_info.items()
            if isinstance(v, (int, float, str, np.number))
        }
        # Normalized Herfindahl–Hirschman Index
        metrics_to_log["final_hhi"] = (n * (position_pct ** 2).sum() - 1) / (n - 1)
        metrics_to_log["final_total_return"] = total_return

        formatted_metrics = pformat(metrics_to_log, indent=4)
        logger.info(formatted_metrics)
        print(pformat(dict(last_info.items())))

        for k, v in metrics_to_log.items():
            episode.custom_metrics[k] = v


@hydra.main(config_name="config", config_path="conf")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    logger.info(f"config: {OmegaConf.to_yaml(config)}")
    data = utils.load_local_data(list(config.env_config.tickers), config.env_config.start_date,
                                 config.env_config.end_date,
                                 freq=f'{config.env_config.data_freq}{config.env_config.freq_unit}')
    close_df = pd.DataFrame({sym: data[sym].close for sym in list(config.env_config.tickers)})

    if close_df.isna().any().any():
        msg = 'Close prices contain NaNs'
        msg += f'\n{close_df[close_df.isna().any(axis=1)]}'
        raise ValueError(msg)

    # check for zeros or negative prices
    if (close_df <= 0).any().any():
        msg = 'Close prices contain zeros or negative values'
        msg += f'\n{close_df[close_df <= 0]}'
        raise ValueError(msg)

    model_config = OmegaConf.to_container(config.model_config)
    env_config = {
        'open_prices': close_df.values,  # use close for both for now
        'close_prices': close_df.values,
        'timestamps': close_df.index,
        **OmegaConf.to_container(config.env_config)
    }
    if config.smoke_test:
        logger.info('Smoke testing env')
        env = get_wrapped_env(config.env_config.env)(env_config=env_config)
        logger.info('resetting env')
        obs, *_ = env.reset()
        logger.info(f'obs.shape: {obs.shape}')
        logger.info('stepping env')
        obs, *_ = env.step(np.array([1.0] * env.num_assets) / env.num_assets)
        logger.info('done smoke testing env')
        return

    algo = (
        PPOConfig()
        .callbacks(CustomCallbacks).exploration()
        .training(gamma=0.99, lr=0.0005, clip_param=0.2, model=model_config)
        .resources(num_gpus=1)
        .rollouts(num_envs_per_worker=4)
        .environment(env=config.env_config.env, env_config=env_config)
    ).build()

    logger.info('Training')
    for _ in range(5):
        result = algo.train()
        logger.info(pretty_print(result))
    save_path = algo.save('models')
    logger.info(f'saved model to {save_path}')


if __name__ == '__main__':
    main()
