"""PPO Algorithm for multi-agent reinforcement learning.

This module implements the PPO (Proximal Policy Optimization) algorithm
for multi-agent systems. Unlike HAPPO, PPO does not use factor-based
gradient scaling.
"""

import torch
from tensordict import TensorDictBase

from heteromark.algorithm.algorithm_base import AlgorithmBase


class PpoAlgorithm(AlgorithmBase):
    """PPO Algorithm for multi-agent training.

    This class implements the PPO (Proximal Policy Optimization) algorithm
    for multi-agent systems. PPO updates all agents independently without
    factor-based coordination.
    """

    def __init__(
        self,
        agent_groups: dict[str, list],
        policy_modules: dict[str, any],
        sample_log_prob_key: str = "action_log_prob",
        action_aggregation: str = "sum",
        fixed_order: bool = False,
        advantage_key: str = "advantage",
        device: torch.device = None,
        functional: bool = True,
    ):
        """Initialize the PPO algorithm.

        Args:
            agent_groups (dict): Dictionary mapping agent group names to lists of agents.
            policy_modules (dict): Dictionary mapping agent group names to their policy modules.
            sample_log_prob_key (str, optional): Key for log probabilities. Defaults to "action_log_prob".
            action_aggregation (str, optional): Method to aggregate action log probs. Defaults to "sum".
            fixed_order (bool, optional): If True, use fixed agent order. Defaults to False.
            advantage_key (str, optional): Key for advantage values. Defaults to "advantage".
            device (torch.device, optional): Device for tensor operations. Defaults to None (CPU).
            functional (bool, optional): Whether to use functional mode. Defaults to True.
        """
        super().__init__(
            agent_groups=agent_groups,
            policy_modules=policy_modules,
            sample_log_prob_key=sample_log_prob_key,
            action_aggregation=action_aggregation,
            fixed_order=fixed_order,
            advantage_key=advantage_key,
            device=device,
            functional=functional,
        )

    def prepare_rollout(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Prepare tensordict for rollout.

        This method can be overridden by subclasses to modify the tensordict
        before rollout (e.g., adding exploration noise).

        For PPO, no modifications are made.

        Args:
            tensordict (TensorDictBase): Input tensordict for rollout.


        """

        pass

    def post_update(
        self, tensordict: TensorDictBase, agent_group: str
    ) -> TensorDictBase:
        """
        Update factors or tensordict entries after update of agent_group.

        For PPO, no updates are necessary.


        Args:
            tensordict (TensorDictBase): Input tensordict for rollout.
            agent_group (str): Name of the agent group that was just updated.
        """

        pass

    def prepare_training(
        self, tensordict: TensorDictBase, agent_group: str
    ) -> TensorDictBase:
        """Prepare tensordict for training.

        This method can be overridden by subclasses to modify the tensordict
        before training (e.g., normalizing advantages).

        Args:
            tensordict (TensorDictBase): Input tensordict for training.
        """

        pass

    def pre_update(
        self, tensordict: TensorDictBase, agent_group: str
    ) -> TensorDictBase:
        """
        Update factors or tensordict entries before update of agent_group.


        Args:
            tensordict (TensorDictBase): Input tensordict for rollout.
            agent_group (str): Name of the agent group that was just updated.
        """

        tensordict[(agent_group, "advantage")] = (
            tensordict["advantage"].clone().unsqueeze(-1)
        )
