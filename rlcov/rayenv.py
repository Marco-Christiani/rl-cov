import gymnasium as gym
import numpy as np
from ray.rllib.env import EnvContext

from env import TradingEnv


class RayTradingEnv(TradingEnv, gym.Env):
    def __init__(self, config: EnvContext):
        super().__init__(config["open_prices"], config["close_prices"], config["init_cash"], config["txn_cost"])
        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_assets,),
            dtype=np.float32
        )
