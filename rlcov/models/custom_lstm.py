import torch.nn as nn
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override


class TorchMLP(nn.Module):
    def __init__(self, input_dim, layers, output_dim, activation=nn.Tanh):
        super(TorchMLP, self).__init__()
        layers = [input_dim] + layers + [output_dim]
        mlp_layers = []
        for i in range(len(layers) - 2):
            mlp_layers.extend([nn.Linear(layers[i], layers[i + 1]), activation()])
        mlp_layers.append(nn.Linear(layers[-2], layers[-1]))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp(x)


class TorchMLPEncoder(nn.Module):
    def __init__(self, input_dim, layers):
        super(TorchMLPEncoder, self).__init__()
        self.net = TorchMLP(input_dim, layers[:-1], layers[-1])

    def forward(self, x):
        return self.net(x)


class TorchLSTMEncoder(nn.Module):
    def __init__(self, input_dim, mlp_layers, lstm_hidden_size):
        super(TorchLSTMEncoder, self).__init__()
        self.tokenizer = TorchMLPEncoder(input_dim, mlp_layers)
        self.lstm = nn.LSTM(mlp_layers[-1], lstm_hidden_size, batch_first=True)

    def forward(self, x, state=None):
        x = self.tokenizer(x)
        x, state = self.lstm(x, state)
        return x, state


class TorchStatefulActorCriticEncoder(nn.Module):
    def __init__(self, input_dim, mlp_layers, lstm_hidden_size):
        super(TorchStatefulActorCriticEncoder, self).__init__()
        self.actor_encoder = TorchLSTMEncoder(input_dim, mlp_layers, lstm_hidden_size)
        self.critic_encoder = TorchLSTMEncoder(input_dim, mlp_layers, lstm_hidden_size)

    def forward(self, x, state=None):
        actor_out, actor_state = self.actor_encoder(x, state)
        critic_out, critic_state = self.critic_encoder(x, state)
        return actor_out, critic_out, actor_state, critic_state


class TorchMLPHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TorchMLPHead, self).__init__()
        self.net = TorchMLP(input_dim, [], output_dim)

    def forward(self, x):
        return self.net(x)


class LSTMComplex(RecurrentNetwork, nn.Module):
    def __init__(self, action_space, obs_space, num_outputs, model_config, name: str = "LSTMComplex"):
        nn.Module.__init__(self)
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name
        )
        self.encoder = TorchStatefulActorCriticEncoder(obs_space, [64, 64], 32)
        self.pi = TorchMLPHead(32, action_space)
        self.vf = TorchMLPHead(32, 1)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        actor_out, critic_out, actor_state, critic_state = self.encoder(inputs, state)
        pi_out = self.pi(actor_out)
        vf_out = self.vf(critic_out)
        return pi_out, vf_out, actor_state, critic_state

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self.vf(self.encoder._features), [-1])

    @override(ModelV2)
    def get_initial_state(self):
        # Return initial state for both actor and critic LSTM
        linear = next(self.pi.net.mlp.children())
        h_actor = linear.weight.new(1, 32).zero_().squeeze(0)
        c_actor = linear.weight.new(1, 32).zero_().squeeze(0)
        h_critic = linear.weight.new(1, 32).zero_().squeeze(0)
        c_critic = linear.weight.new(1, 32).zero_().squeeze(0)
        return [h_actor, c_actor, h_critic, c_critic]

    def import_from_h5(self, h5_file: str) -> None:
        pass


import torch
import torch.nn as nn


def test_ppo_torch_rl_module():
    import gymnasium as gym
    n_actions = 1
    obs_dim = 4
    batch_size = 5
    seq_len = 8
    obs_space = gym.spaces.Box(low=0, high=1, shape=(obs_dim,))
    action_space = gym.spaces.Box(
        low=-torch.inf,
        high=torch.inf,
        shape=(n_actions,)
    )
    model = LSTMComplex(action_space, obs_space, n_actions, {})

    obs = torch.rand(batch_size, seq_len, obs_dim)
    pi_out, vf_out, actor_state, critic_state = model(obs)

    assert pi_out.shape == (batch_size, seq_len, n_actions), f"Unexpected shape: {pi_out.shape}"
    assert vf_out.shape == (batch_size, seq_len, 1), f"Unexpected shape: {vf_out.shape}"
    assert actor_state[0].shape == (1, batch_size, 32), f"Unexpected shape: {actor_state[0].shape}"
    assert actor_state[1].shape == (1, batch_size, 32), f"Unexpected shape: {actor_state[1].shape}"
    assert critic_state[0].shape == (1, batch_size, 32), f"Unexpected shape: {critic_state[0].shape}"
    assert critic_state[1].shape == (1, batch_size, 32), f"Unexpected shape: {critic_state[1].shape}"

    print("All tests passed!")


if __name__ == '__main__':
    test_ppo_torch_rl_module()
