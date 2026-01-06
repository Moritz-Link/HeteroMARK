"""Base class for multi-agent reinforcement learning algorithms.

This module provides an abstract base class that defines the interface for
heterogeneous multi-agent RL algorithms like HAPPO, PPO, etc.
"""

import contextlib
import warnings
from abc import ABC, abstractmethod

import torch
from tensordict import TensorDictBase, is_tensor_collection
from tensordict.nn import (
    CompositeDistribution,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    set_composite_lp_aggregate,
)
from tensordict.utils import NestedKey
from torchrl.objectives.utils import _maybe_get_or_select, _sum_td_features


class AlgorithmBase(ABC):
    """Abstract base class for multi-agent RL algorithms.

    This class defines the common interface and shared functionality for
    heterogeneous multi-agent algorithms. Subclasses must implement
    agent ordering, factor management, and update logic.

    Attributes:
        agent_groups (dict): Dictionary mapping agent group names to lists of agents.
        policy_modules (dict): Dictionary mapping agent group names to their policy modules.
        sample_log_prob_key (str): Key for log probabilities in tensordict.
        action_aggregation (str): Method to aggregate action log probs ("sum", "mean", "prod").
        fixed_order (bool): If True, agents are updated in fixed order.
        device (torch.device): Device for tensor operations.
        functional (bool): Whether to use functional mode for networks.
        advantage_key (str): Key for advantage values in tensordict.
        truncated_key (str): Key for truncation flags in tensordict.
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
        """Initialize the algorithm base class.

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
        self.agent_groups = agent_groups
        self.agents = [agent for group in agent_groups.values() for agent in group]
        self.agent_update_type = "group-wise"
        self.policy_modules = policy_modules
        self.sample_log_prob_key = sample_log_prob_key
        self.action_aggregation = action_aggregation
        self.fixed_order = fixed_order
        self.device = device if device is not None else torch.device("cpu")
        self.functional = functional
        self.advantage_key = advantage_key
        self.truncated_key = "truncated"

        # Extract agent group names
        self.group_names = list(agent_groups.keys())
        self.num_groups = len(self.group_names)
        self.num_agents = len(self.agents)

        # Current agent order (will be updated in get_agent_order)
        self.current_agent_order = None

    @abstractmethod
    def prepare_rollout(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Prepare tensordict for rollout.

        This method can be overridden by subclasses to modify the tensordict
        before rollout (e.g., adding exploration noise).

        Args:
            tensordict (TensorDictBase): Input tensordict for rollout.
        """

        pass

    @abstractmethod
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

    @abstractmethod
    def post_update(
        self, tensordict: TensorDictBase, agent_group: str
    ) -> TensorDictBase:
        """
        Update factors or tensordict entries after update of agent_group.


        Args:
            tensordict (TensorDictBase): Input tensordict for rollout.
            agent_group (str): Name of the agent group that was just updated.
        """

        pass

    @abstractmethod
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

    def _get_dist_of_actor(self, tensordict: TensorDictBase, actor_network: any):
        """Get the distribution of the actor network.

        This method can be overridden by subclasses to customize how the
        distribution is obtained from the actor network.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        if isinstance(
            actor_network,
            (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
        ) or hasattr(actor_network, "get_dist"):
            context = contextlib.nullcontext()

            with context:
                dist = actor_network.get_dist(tensordict)

            return dist
        else:
            raise NotImplementedError(
                "Only probabilistic modules from tensordict.nn are currently supported. "
                "If you need to implement a custom logic to retrieve the log-probs, "
                "please augment the algorithm class."
            )

    def _get_cur_log_prob(self, tensordict: TensorDictBase, actor_network: any):
        """Get current log probabilities from the actor network.

        Adapted from PPOLoss._get_cur_log_prob to work with specific agent policies.

        Args:
            tensordict (TensorDictBase): Input tensordict containing observations and actions.
            actor_network: The actor network (policy module) for the current agent.

        Returns:
            tuple: (log_prob, dist, is_composite) - log probabilities, distribution, and composite flag.
        """
        if isinstance(
            actor_network,
            (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
        ) or hasattr(actor_network, "get_dist"):
            context = contextlib.nullcontext()

            with context:
                dist = actor_network.get_dist(tensordict)

            is_composite = isinstance(dist, CompositeDistribution)

            if is_composite:
                action = tensordict.select(
                    *(
                        (self.action_key,)
                        if isinstance(self.action_key, NestedKey)
                        else (self.action_key,)
                        if isinstance(self.action_key, str)
                        else self.action_key
                    )
                )
            else:
                action = _maybe_get_or_select(tensordict, self.action_key)

            if action.requires_grad:
                raise RuntimeError(
                    f"tensordict stored {self.action_key} requires grad."
                )
            log_prob = dist.log_prob(action)
        else:
            raise NotImplementedError(
                "Only probabilistic modules from tensordict.nn are currently supported. "
                "If you need to implement a custom logic to retrieve the log-probs, "
                "please augment the algorithm class."
            )
        return log_prob, dist, is_composite

    def _log_weight(
        self,
        tensordict: TensorDictBase,
        actor_network: any,
        actor_network_params: any = None,
        adv_shape: torch.Size = None,
    ):
        """Compute log weight (importance weight) for the policy update.

        Adapted from PPOLoss._log_weight to work with specific agent policies.

        Args:
            tensordict (TensorDictBase): Input tensordict containing observations, actions, and old log probs.
            actor_network: The actor network (policy module) for the current agent.
            actor_network_params: Parameters for the actor network (if functional mode).
            adv_shape (torch.Size, optional): Shape of the advantage tensor.

        Returns:
            tuple: (log_weight, dist, kl_approx) - log importance weight, distribution, and KL approximation.
        """
        if adv_shape is None:
            adv_shape = tensordict.batch_size

        prev_log_prob = _maybe_get_or_select(
            tensordict,
            self.sample_log_prob_key,
            adv_shape,
        )
        if prev_log_prob is None:
            raise KeyError(
                f"Couldn't find the log-prob {self.sample_log_prob_key} in the input data."
            )
        if prev_log_prob.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.sample_log_prob_key} requires grad."
            )

        log_prob, dist, is_composite = self._get_cur_log_prob(
            tensordict, actor_network, actor_network_params
        )

        if is_composite:
            with set_composite_lp_aggregate(False):
                if not is_tensor_collection(prev_log_prob):
                    warnings.warn(
                        "You are using a composite distribution, yet your log-probability is a tensor. "
                        "Make sure you have called tensordict.nn.set_composite_lp_aggregate(False).set() at "
                        "the beginning of your script to get a proper composite log-prob.",
                        category=UserWarning,
                    )

                    if is_tensor_collection(log_prob):
                        log_prob = _sum_td_features(log_prob)
                        log_prob.view_as(prev_log_prob)
                if log_prob.batch_size != adv_shape:
                    log_prob.batch_size = adv_shape

        log_weight = (log_prob - prev_log_prob).unsqueeze(-1)
        if is_tensor_collection(log_weight):
            log_weight = _sum_td_features(log_weight)
            log_weight = log_weight.view(adv_shape).unsqueeze(-1)

        kl_approx = (prev_log_prob - log_prob).unsqueeze(-1)
        if is_tensor_collection(kl_approx):
            kl_approx = _sum_td_features(kl_approx)

        return log_weight, dist, kl_approx

    def get_agent_order(self) -> list[str]:
        """Generate agent order for training.

        Returns a random permutation of agent groups if fixed_order is False,
        otherwise returns the groups in their original order.

        Returns:
            List[str]: List of agent group names in training order.
        """
        if self.fixed_order:
            self.current_agent_order = list(range(self.num_groups))
            return [self.group_names[i] for i in self.current_agent_order]
        else:
            if self.agent_update_type == "agent-wise":
                self.current_agent_order = list(torch.randperm(self.num_agents).numpy())
                return [self.agents[i] for i in self.current_agent_order]
            if self.agent_update_type == "group-wise":
                self.current_agent_order = list(torch.randperm(self.num_groups).numpy())
                return [self.group_names[i] for i in self.current_agent_order]
