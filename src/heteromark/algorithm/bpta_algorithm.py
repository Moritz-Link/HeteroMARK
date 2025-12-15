import torch
from tensordict import TensorDictBase

from heteromark.algorithm.algorithm_base import AlgorithmBase


class BptaAlgorithm(AlgorithmBase):
    """BPTA Algorithm for managing factor updates in heterogeneous agent training.

    This class implements the BPTA (Backpropagation Through Agents) algorithm, which extends
    algorithm's factor management system, which accounts for updates made by previous
    agents in the training sequence.

    Based on: https://github.com/LiZhYun/BackPropagationThroughAgents
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
        """Initialize the HAPPO algorithm.

        Args:
            agent_groups (Dict[str, List]): Dictionary mapping agent group names to lists of agents.
                Example: {"job_selector": [agents], "agv_selector": [agents]}
            policy_modules (Dict[str, any]): Dictionary mapping agent group names to their policy modules.
                Example: {"job_selector": policy_module, "agv_selector": policy_module}
            sample_log_prob_key (str, optional): Key for log probabilities in tensordict.
                Defaults to "action_log_prob".
            action_aggregation (str, optional): Method to aggregate action log probs.
                Options: "sum", "mean", "prod". Defaults to "sum".
            fixed_order (bool, optional): If True, agents are updated in fixed order.
                If False, random permutation is used. Defaults to False.
            advantage_key (str, optional): Key for advantage values. Defaults to "advantage".
            device (torch.device, optional): Device for tensor operations. Defaults to None.
            functional (bool, optional): Whether to use functional mode for networks. Defaults to True.
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

        # Initialize factor as None (will be set based on tensordict shape)
        self.factor = None

        # Return actual agent group names in the determined order

    def reset_factor(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Reset the factor tensor to ones based on tensordict shape.

        The factor shape is typically (episode_length, n_rollout_threads, 1) for EP state
        or matches the batch dimensions of the tensordict.

        Args:
            tensordict (TensorDictBase): Input tensordict to determine factor shape.

        Returns:
            torch.Tensor: Factor tensor initialized to ones.
        """
        # Get batch shape from tensordict
        pass

    def set_adv_as_factor(self, tensordict: TensorDictBase):
        """Set the factor to advantage if shapes and dimensions match.

        Args:
            tensordict (TensorDictBase): Input tensordict containing advantage.

        Raises:
            ValueError: If advantage shape/dimensions don't match expected factor shape.
        """
        pass

    def update_factor(
        self, tensordict: TensorDictBase, agent_group: str
    ) -> torch.Tensor:
        """Update the factor tensor based on the ratio of new to old action log probabilities.

        This implements the HAPPO update logic:
        factor = factor * exp(new_log_probs - old_log_probs)

        The factor accounts for updates made by the current agent when training subsequent agents.

        Args:
            tensordict (TensorDictBase): Input tensordict containing observations, actions, and old log probs.
            agent_group (str): Name of the agent group being updated.
            actor_network_params: Parameters for the actor network (if functional mode).

        Returns:
            torch.Tensor: Updated factor tensor.
        """

        pass

    def get_factor(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Get the current factor tensor.

        If factor hasn't been initialized yet, it will be reset based on tensordict shape.

        Args:
            tensordict (TensorDictBase): Input tensordict for shape reference.

        Returns:
            torch.Tensor: Current factor tensor.
        """
        if self.factor is None:
            self.reset_factor(tensordict)
        return self.factor

    def get_factor_for_agent(self, agent_group: str = None) -> torch.Tensor:
        """Get the factor tensor for a specific agent group.

        This returns the current factor that should be used when training the specified agent.

        Args:
            agent_group (str): Name of the agent group.

        Returns:
            torch.Tensor: Factor tensor to use for this agent's update.
        """
        pass

    def prepare_rollout(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Prepare tensordict for rollout.

        This method can be overridden by subclasses to modify the tensordict
        before rollout (e.g., adding exploration noise).

        Set the Advantage of the tensordict as the factor.

        Args:
            tensordict (TensorDictBase): Input tensordict for rollout.


        """

        pass

    def post_update(
        self, tensordict: TensorDictBase, agent_group: str
    ) -> TensorDictBase:
        """
        Update factors or tensordict entries after update of agent_group.

        For HAPPO; update the factor based on the latest policy update.


        Args:
            tensordict (TensorDictBase): Input tensordict for rollout.
            agent_group (str): Name of the agent group that was just updated.
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

        pass
