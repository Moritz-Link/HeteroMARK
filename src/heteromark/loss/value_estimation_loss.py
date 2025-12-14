import torch.nn as nn
from tensordict import TensorDict
from torchrl.objectives.utils import _reduce, distance_loss


class LossValueEstimationModule(nn.Module):
    def __init__(
        self,
        value_network: nn.Module,
        loss_critic_type: str,
        state_value_key: str = "state_value",
        value_target_key: str = "value_target",
        critic_coeff: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.value_network = value_network
        self.loss_critic_type = loss_critic_type
        self.state_value_key = state_value_key
        self.value_target_key = value_target_key
        self.critic_coeff = critic_coeff
        self.reduction = reduction

    def forward(self, tensordict: TensorDict):
        target_return = tensordict.get(self.value_target_key)

        # Compute new State_values
        state_value_td = self.value_network(tensordict)
        state_value = state_value_td.get(self.state_value_key)

        if state_value is None:
            raise KeyError(
                f"the key {self.state_value_key} was not found in the critic output tensordict. "
                f"Make sure that the 'value_key' passed to PPO is accurate."
            )

        loss_value = distance_loss(
            target_return,
            state_value,
            loss_function=self.loss_critic_type,
        )

        # Reduce Loss Value
        loss_value = _reduce(loss_value, reduction=self.reduction).squeeze(-1)

        # Apply Critic Coefficient
        loss_value = loss_value * self.critic_coeff
        return loss_value
