import copy

import torch
from tensordict import TensorDictBase
from torchrl.objectives.utils import _maybe_get_or_select

from heteromark.algorithm.algorithm_base import AlgorithmBase


class HappoAlgorithm(AlgorithmBase):
    """HAPPO Algorithm for managing factor updates in heterogeneous agent training.

    This class implements the HAPPO (Heterogeneous-Agent Proximal Policy Optimization)
    algorithm's factor management system, which accounts for updates made by previous
    agents in the training sequence.

    Based on: https://github.com/PKU-MARL/HARL/blob/main/harl/runners/on_policy_ha_runner.py
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
        batch_shape = tensordict.batch_size

        # Create factor with shape matching batch dimensions plus feature dimension
        if len(batch_shape) == 1:
            # Single batch dimension: (batch_size, 1)
            factor_shape = (batch_shape[0], 1)
        elif len(batch_shape) == 2:
            # Two batch dimensions: (episode_length, n_rollout_threads, 1)
            factor_shape = (*batch_shape, 1)
        else:
            # General case: append 1 as feature dimension
            factor_shape = (*batch_shape, 1)
        # If factor_shape is 2D (e.g. (256, 1)), expand it to 3D (256, 1, 1)
        # if len(factor_shape) == 2:
        #     factor_shape = (*factor_shape, 1)
        self.factor = torch.ones(factor_shape, dtype=torch.float32, device=self.device)
        return self.factor

    def set_adv_as_factor(self, tensordict: TensorDictBase):
        """Set the factor to advantage if shapes and dimensions match.

        Args:
            tensordict (TensorDictBase): Input tensordict containing advantage.

        Raises:
            ValueError: If advantage shape/dimensions don't match expected factor shape.
        """
        adv = tensordict[self.advantage_key]

        # Compare with current factor if it exists
        if self.factor is not None:
            if (
                adv.shape != self.factor.shape
                or adv.ndim != self.factor.ndim
                or adv.dtype != self.factor.dtype
                or adv.device != self.factor.device
            ):
                raise ValueError(
                    f"Advantage properties don't match factor. "
                    f"Advantage: shape={adv.shape}, ndim={adv.ndim}, dtype={adv.dtype}, device={adv.device}. "
                    f"Factor: shape={self.factor.shape}, ndim={self.factor.ndim}, dtype={self.factor.dtype}, device={self.factor.device}."
                )

                self.factor = copy.deepcopy(adv)
        adv = tensordict[self.advantage_key]
        self.factor = copy.deepcopy(adv)
        tensordict["factor"] = self.factor

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
        if self.factor is None:
            raise RuntimeError(
                "Factor not initialized. Call reset_factor() or get_factor() first."
            )

        if agent_group not in self.policy_modules:
            raise ValueError(
                f"Agent group '{agent_group}' not found in policy_modules."
            )
        # TODO: Thinking about how to update only factors with alive agent information!?
        # Get the policy module for this agent
        actor_network = self.policy_modules[agent_group]

        # Get old log probabilities from tensordict
        # adv_shape = tensordict[(agent_group, "advantage")].shape
        adv_shape = tensordict[("advantage")].shape

        # Action from current agent
        # Now the shape is 256,1
        prev_log_prob = _maybe_get_or_select(
            tensordict,
            (agent_group, self.sample_log_prob_key),
            adv_shape,
        )
        self.action_key = (agent_group, "action")
        # Compute new log probabilities using the updated policy
        # Use no_grad() because we don't need gradients for factor computation
        # This prevents errors when reusing tensordicts that were already used in backward passes
        with torch.no_grad():
            new_log_prob, _, _ = self._get_cur_log_prob(tensordict, actor_network)

        # Detach old log probs to ensure no gradients flow
        prev_log_prob = prev_log_prob.detach()
        new_log_prob = new_log_prob.detach()

        # Compute importance weights: exp(new_log_prob - old_log_prob)
        # Apply action aggregation (sum, mean, or prod over action dimension)
        imp_weights = torch.exp((new_log_prob - prev_log_prob).unsqueeze(-1))
        truncated_mask = tensordict[(agent_group, self.truncated_key)]

        # Mask importance weights: where truncated is True, use 1.0 (no factor update)
        # where truncated is False, use computed imp_weights
        imp_weights = torch.where(
            truncated_mask, torch.ones_like(imp_weights), imp_weights
        )

        # Multiply all importance weights along the agent dimension (dim=1)
        imp_weights = torch.prod(imp_weights, dim=1, keepdim=False)
        # Reshape importance weights to match factor shape if needed
        if imp_weights.shape != self.factor.shape:
            raise Warning(
                f"Importance weights shape {imp_weights.shape} does not match factor shape {self.factor.shape}. Attempting to reshape."
            )
            # Assuming imp_weights is flattened: (episode_length * n_threads, 1)
            # Reshape to: (episode_length, n_threads, 1)
            target_shape = self.factor.shape
            if len(target_shape) == 2:
                # Factor is (batch_size, 1), imp_weights should match
                imp_weights = imp_weights.reshape(target_shape)
            elif len(target_shape) == 3:
                # Factor is (episode_length, n_threads, 1)
                # Compute episode_length and n_threads from total size
                total_size = imp_weights.shape[0]
                episode_length = target_shape[0]
                n_threads = target_shape[1]
                if episode_length * n_threads == total_size:
                    imp_weights = imp_weights.reshape(episode_length, n_threads, 1)

        # Update factor: factor *= importance_weights (detach to ensure no gradients)
        self.factor = self.factor * imp_weights.detach()

        return self.factor

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
        if self.factor is None:
            raise RuntimeError(
                "Factor not initialized. Call reset_factor() or get_factor() first."
            )

        # Return a copy to prevent accidental modification
        return (
            self.factor.clone()
            if isinstance(self.factor, torch.Tensor)
            else torch.from_numpy(self.factor.copy()).to(self.device)
        )

    def prepare_rollout(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Prepare tensordict for rollout.

        This method can be overridden by subclasses to modify the tensordict
        before rollout (e.g., adding exploration noise).

        Set the Advantage of the tensordict as the factor.

        Args:
            tensordict (TensorDictBase): Input tensordict for rollout.


        """

        self.set_adv_as_factor(tensordict)

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

        self.update_factor(tensordict, agent_group)
        tensordict["factor"] = self.get_factor(tensordict)

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
        tensordict[(agent_group, "factor")] = tensordict["factor"].clone().unsqueeze(-1)
