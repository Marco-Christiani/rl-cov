from ray.rllib.models import ModelCatalog

from rlcov.models.custom_lstm import LSTMComplex
from rlcov.models.torchrnn import TorchRNN1
from rlcov.models.torchrnn import TorchRNN2

ModelCatalog.register_custom_model("TorchRNN1", TorchRNN1)
ModelCatalog.register_custom_model("TorchRNN2", TorchRNN2)
ModelCatalog.register_custom_model("LSTMComplex", LSTMComplex)
