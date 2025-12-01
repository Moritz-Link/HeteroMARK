import torch
import numpy as np
import warnings
import contextlib
from typing import Dict, List
from tensordict import TensorDictBase, is_tensor_collection
from tensordict.nn import (
    CompositeDistribution,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    set_composite_lp_aggregate,
)
from tensordict.utils import NestedKey
from torch import distributions as d
from torchrl.objectives.utils import _maybe_get_or_select, _sum_td_features


class HappoAlgorithm:
    """HAPPO Algorithm for managing factor updates in heterogeneous agent training.
    
    This class implements the HAPPO (Heterogeneous-Agent Proximal Policy Optimization)
    algorithm's factor management system, which accounts for updates made by previous
    agents in the training sequence.
    
    Based on: https://github.com/PKU-MARL/HARL/blob/main/harl/runners/on_policy_ha_runner.py
    """
    
    def __init__(self, agent_groups: Dict[str, List], policy_modules: Dict[str, any],
                 sample_log_prob_key: str = "action_log_prob",
                 action_aggregation: str = "sum", fixed_order: bool = False, 
                 advantage_key: str = "advantage",
                 device: torch.device = None, functional: bool = True):
        """Initialize the HAPPO algorithm.
        
        Args:
            agent_groups (Dict[str, List]): Dictionary mapping agent group names to lists of agents.
                Example: {"job_selector": [agents], "agv_selector": [agents]}
            policy_modules (Dict[str, any]): Dictionary mapping agent group names to their policy modules.
                Example: {"job_selector": policy_module, "agv_selector": policy_module}
            action_key (str, optional): Key for action in tensordict. Defaults to "action".
            sample_log_prob_key (str, optional): Key for log probabilities in tensordict. 
                Defaults to "action_log_prob".
            action_aggregation (str, optional): Method to aggregate action log probs. 
                Options: "sum", "mean", "prod". Defaults to "sum".
            fixed_order (bool, optional): If True, agents are updated in fixed order. 
                If False, random permutation is used. Defaults to False.
            device (torch.device, optional): Device for tensor operations. Defaults to None.
            functional (bool, optional): Whether to use functional mode for networks. Defaults to True.
        """
        self.agent_groups = agent_groups
        self.policy_modules = policy_modules
        self.sample_log_prob_key = sample_log_prob_key
        self.action_aggregation = action_aggregation
        self.fixed_order = fixed_order
        self.device = device if device is not None else torch.device("cpu")
        self.functional = functional
        
        # Extract agent group names
        self.group_names = list(agent_groups.keys())
        self.num_groups = len(self.group_names)
        
        # Initialize factor as None (will be set based on tensordict shape)
        self.factor = None
        
        # Current agent order (will be updated in get_agent_order)
        self.current_agent_order = None
    
    def get_agent_order(self) -> List[str]:
        """Generate agent order for training.
        
        Returns a random permutation of agent groups if fixed_order is False,
        otherwise returns the groups in their original order.
        
        Returns:
            List[str]: List of agent group names in training order.
        """
        if self.fixed_order:
            self.current_agent_order = list(range(self.num_groups))
        else:
            self.current_agent_order = list(torch.randperm(self.num_groups).numpy())
        
        # Return actual agent group names in the determined order
        return [self.group_names[i] for i in self.current_agent_order]
    
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
        if len(factor_shape) == 2:
            factor_shape = (*factor_shape, 1)
        self.factor = torch.ones(factor_shape, dtype=torch.float32, device=self.device)
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
    
    def _get_cur_log_prob(self, tensordict: TensorDictBase, actor_network: any):
        """Get current log probabilities from the actor network.
        
        Adapted from PPOLoss._get_cur_log_prob to work with specific agent policies.
        
        Args:
            tensordict (TensorDictBase): Input tensordict containing observations and actions.
            actor_network: The actor network (policy module) for the current agent.
            actor_network_params: Parameters for the actor network (if functional mode).
        
        Returns:
            tuple: (log_prob, dist, is_composite) - log probabilities, distribution, and composite flag.
        """
        if isinstance(
            actor_network,
            (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
        ) or hasattr(actor_network, "get_dist"):
            # Use functional mode if parameters are provided
 
            context = contextlib.nullcontext()
            
            with context:
                dist = actor_network.get_dist(tensordict)
            
            is_composite = isinstance(dist, CompositeDistribution)

            if is_composite:
                action = tensordict.select(
                    *(
                        (self.action_key,)
                        if isinstance(self.action_key, NestedKey)
                        else (self.action_key,) if isinstance(self.action_key, str) else self.action_key
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
                "please augment the HappoAlgorithm class."
            )
        return log_prob, dist, is_composite
    
    def _log_weight(self, tensordict: TensorDictBase, actor_network: any,
                    actor_network_params: any = None, adv_shape: torch.Size = None):
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
    
    def update_factor(self, 
                     tensordict: TensorDictBase,
                     agent_group: str) -> torch.Tensor:
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
            raise RuntimeError("Factor not initialized. Call reset_factor() or get_factor() first.")
        
        if agent_group not in self.policy_modules:
            raise ValueError(f"Agent group '{agent_group}' not found in policy_modules.")
        
        # Get the policy module for this agent
        actor_network = self.policy_modules[agent_group]
        
        # Get old log probabilities from tensordict
        adv_shape = tensordict[(agent_group,"advantage")].shape 
        
        # Action from current agent
        # Now the shape is 256,1 
        prev_log_prob = _maybe_get_or_select(
            tensordict,
            (agent_group,self.sample_log_prob_key),
            adv_shape,
        )
        self.action_key = (agent_group, "action")
        # Compute new log probabilities using the updated policy
        # Use no_grad() because we don't need gradients for factor computation
        # This prevents errors when reusing tensordicts that were already used in backward passes
        with torch.no_grad():
            new_log_prob, _, _ = self._get_cur_log_prob(
                tensordict, actor_network
            )
        
        # Detach old log probs to ensure no gradients flow
        prev_log_prob = prev_log_prob.detach()
        new_log_prob = new_log_prob.detach()
        
        # Compute importance weights: exp(new_log_prob - old_log_prob)
        # Apply action aggregation (sum, mean, or prod over action dimension)
        imp_weights = torch.exp((new_log_prob - prev_log_prob).unsqueeze(-1))

        
        # Reshape importance weights to match factor shape if needed
        if imp_weights.shape != self.factor.shape:
            raise Warning(f"Importance weights shape {imp_weights.shape} does not match factor shape {self.factor.shape}. Attempting to reshape.")
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
    
    def get_factor_for_agent(self, agent_group: str) -> torch.Tensor:
        """Get the factor tensor for a specific agent group.
        
        This returns the current factor that should be used when training the specified agent.
        The factor accounts for all updates made by agents that were trained before this one
        in the current training iteration.
        
        Args:
            agent_group (str): Name of the agent group.
        
        Returns:
            torch.Tensor: Factor tensor to use for this agent's update.
        """
        if self.factor is None:
            raise RuntimeError("Factor not initialized. Call reset_factor() or get_factor() first.")
        
        # Return a copy to prevent accidental modification
        return self.factor.clone() if isinstance(self.factor, torch.Tensor) else torch.from_numpy(self.factor.copy()).to(self.device)
