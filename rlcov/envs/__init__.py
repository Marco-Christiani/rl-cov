from gymnasium.wrappers import FlattenObservation
from ray.tune import register_env

from rlcov.envs import rayenv
from rlcov.envs import shrinkenv


def rte_factory(env_config: dict):
    return FlattenObservation(rayenv.RayTradingEnv(env_config))


register_env("RayTradingEnv", rte_factory)


def se_factory(env_config: dict):
    return FlattenObservation(shrinkenv.ShrinkEnv(env_config))


register_env("ShrinkEnv", se_factory)


def get_wrapped_env(env_name: str):
    if env_name == "RayTradingEnv":
        return rte_factory
    elif env_name == "ShrinkEnv":
        return se_factory
