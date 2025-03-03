import logging
from pprint import pformat

import hydra
import numpy as np
import ray
from gymnasium.wrappers import FlattenObservation
from omegaconf import DictConfig
from omegaconf import OmegaConf
from ray import air
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.tune.logger import pretty_print

from rlcov.envs import get_wrapped_env
from rlcov.utils import load_close_from_config

logger = logging.getLogger(__name__)
# context = ray.init(include_dashboard=True, address='auto', ignore_reinit_error=True, num_gpus=1, local_mode=True)
context = ray.init()


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

        print(pformat(dict(last_info.items())))

        for k, v in metrics_to_log.items():
            episode.custom_metrics[k] = v


@hydra.main(config_name="config", config_path="conf")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    logger.info(f"config: {OmegaConf.to_yaml(config)}")
    close_df = load_close_from_config(config)
    logger.info(f'close_df.shape: {close_df.shape}')

    import subprocess
    import os

    def get_git_commit_hash():
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except subprocess.CalledProcessError:
            return "N/A"

    def get_git_diff():
        try:

            return subprocess.check_output(['git', 'diff']).decode('utf-8').strip()
        except subprocess.CalledProcessError:
            return "N/A"

    diff = get_git_diff()
    git_hash = get_git_commit_hash()

    with open(os.path.join(os.getcwd(), f'diff-{git_hash}.txt'), 'w') as diff_file:
        diff_file.write(diff)

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

    if config.algorithm.lower() == 'ddpg':
        from ray.rllib.algorithms.ddpg import DDPGConfig as AlgoConfig
    elif config.algorithm.lower() == 'ppo':
        from ray.rllib.algorithms.ppo import PPOConfig as AlgoConfig
    else:
        raise ValueError(f'Unknown algorithm: {config.algorithm}')

    algo_config = AlgoConfig()

    enable_learner_api = True
    if config.use_custom_model:
        from rlcov import models  # noqa
        model_config = {
            "custom_model": config.custom_model_config.name,
            "max_seq_len": 10,
        }
        enable_learner_api = False

    algo_config = (
        algo_config
        .rl_module(_enable_rl_module_api=enable_learner_api)
        .framework('torch')
        .callbacks(CustomCallbacks)
        .training(
            model=model_config,
            _enable_learner_api=enable_learner_api,
            **OmegaConf.to_container(config.training_args),
        )
        .resources(num_cpus_per_worker=1, num_gpus_per_worker=1)
        # .resources(**OmegaConf.to_container(config.resources_args))
        # .rollouts(**OmegaConf.to_container(config.rollouts_args))
        # .resources(**OmegaConf.to_container(config.resources_args))
        # .rollouts(**OmegaConf.to_container(config.rollouts_args))
        .environment(env=config.env_config.env, env_config=env_config)
    )

    # algo = algo_config.build()
    import gymnasium
    from ray.tune.registry import register_env

    import ray

    import ray.air
    from ray.rllib.algorithms.ppo import PPOConfig

    from ray.rllib.policy.policy import PolicySpec
    from ray.tune.stopper import MaximumIterationStopper

    # Create an RLlib config using multi-agent PPO on mobile-env's small scenario.
    # config = (
    #     PPOConfig()
    #     .environment(env="CartPole-v1")
    #     .framework("torch")
    #     # RLlib needs +1 CPU than configured below (for the driver/traininer?)
    #     .resources(num_cpus_per_worker=1, num_gpus_per_worker=1/16)
    #     .rollouts(num_rollout_workers=15)
    # )

    # Create the Trainer/Tuner and define how long to train
    tuner = ray.tune.Tuner(
        "PPO",
        param_space=algo_config.to_dict(),
    )

    # Run training and save the result
    result_grid = tuner.fit()
    logger.info(ray.get_gpu_ids())

    logger.info('BYE')
    # logger.info('Training')
    # for i in range(1, config.training_iterations + 1):
    #     result = algo.train()
    #     logger.info(pretty_print(result))
    #     if i % config.checkpoint_freq == 0:
    #         checkpoint_path = algo.save('models')
    #         logger.info(f'saved model to {checkpoint_path}')
    # algo.export_policy_model

    # results = tune.run("PPO", config=algo_config.to_dict())
    # logger.info(pretty_print(results))


if __name__ == '__main__':
    main()
