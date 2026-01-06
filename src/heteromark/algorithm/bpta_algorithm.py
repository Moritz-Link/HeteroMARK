import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.objectives.utils import _maybe_get_or_select

from heteromark.algorithm.algorithm_base import AlgorithmBase
from heteromark.algorithm.utils import generate_mask_from_order
from heteromark.loss.bpta_loss import BptaActionLoss


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
        self.action_loss = BptaActionLoss()

        # Return actual agent group names in the determined order

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
                self.current_agent_order = torch.randperm(self.num_agents).tolist()
                self.agent_order = list(
                    reversed([self.agents[i] for i in self.current_agent_order])
                )
                return self.agent_order

            elif self.agent_update_type == "group-wise":
                self.current_agent_order = torch.randperm(self.num_groups).tolist()
                self.agent_order = list(
                    reversed([self.group_names[i] for i in self.current_agent_order])
                )
                return self.agent_order

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

    def _get_factor_for_agent(self, agent_group: str = None) -> torch.Tensor:
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

        self.factor = torch.ones((self.num_groups, tensordict.batch_size, 1))
        self.action_gradients = torch.zeros(
            (self.num_groups, self.num_groups, tensordict.batch_size)
        )
        self.execution_mask = generate_mask_from_order(
            self.agent_order, ego_exclusive=False
        )
        ## set factor to ones for all agents num_agents, ep_length , n_rollout_threads
        ## Set action  grad to zeros for each agent to each other : n,n,ep_length, n_rollout_threads, action_shape
        ## ? exection mask batchall

    def prepare_training(
        self, tensordict: TensorDictBase, agent_group: str
    ) -> TensorDictBase:
        """Prepare tensordict for training.

        This method can be overridden by subclasses to modify the tensordict
        before training (e.g., normalizing advantages).

        Args:
            tensordict (TensorDictBase): Input tensordict for training.
        """
        ## update the factor
        ## use other agents and compute action_grad_per_agent
        ## update action_grad

        prod_agent_factor, cated_other_agents_factors = self._get_factor_for_agent(
            agent_group
        )
        tensordict.set("group_factor", prod_agent_factor)
        action_shape = self._get_action_shape(tensordict, agent_group)
        action_grad_per_agent = np.zeros(
            (
                tensordict.batch_size,
                1,
                action_shape,  # assuming action shape is known in this scope
            ),
            dtype=np.float32,
        )

        ## get all updated agents before this agent TODO: only beta
        updated_agents_idx = []
        action_grad_per_agent = self._get_action_grad_of_updated_agents(
            action_grad_per_agent,
            cated_other_agents_factors,
            updated_agents_idx,
            self.group_names.index(agent_group),
            tensordict.batch_size,
        )

        self.action_grad_copy = action_grad_per_agent.copy()  # Basiert auf dem Factor
        tensordict.set("action_grad_per_agent", action_grad_per_agent)

        # old_actions = tensordict.get(agent_group).get("action")  #
        # old_actions.requires_grad = True
        # old_action_wo = old_actions.detach()

        execution_masks_agent = self.execution_mask[
            :, :, self.group_names.index(agent_group)
        ]

        # must be toech with grad all actions?
        # Need ALL the actions?
        self.one_hot_actions = self._get_one_hot_actions(tensordict, agent_group)
        self.old_one_hot_actions = self.one_hot_actions.copy().detach()
        self.one_hot_actions.requires_grad = (
            True  # TODO: add this to the evalute functions?
        )
        # TODO prüfen -> weglassen ? Das ist damit die Agenten über die vorherugen Aktionen bescheid wissen.
        # Aber ohnne kan ich keine Gradienten miteinbeziehen?
        masked_actions = (
            self.one_hot_actions * execution_masks_agent.unsqueeze(-1)
        ).view(*self.one_hot_actions.shape[:-2], -1)

        # https://github.com/LiZhYun/BackPropagationThroughAgents/blob/main/bta/algorithms/bta/algorithm/r_actor_critic.py#L19
        # https://github.com/LiZhYun/BackPropagationThroughAgents/blob/main/bta/algorithms/utils/act.py#L10

        # Hier muss rsample rein
        # https://github.com/LiZhYun/BackPropagationThroughAgents/blob/main/bta/algorithms/utils/act.py#L10

        # Hier old_one_hot_actions ohne gradienten !
        # in masked_actions
        # actor_features = actor_features + self.action_base(torch.cat([masked_actions, id_feat], dim=1))
        # part of gradient graph
        self.old_log_prob = (
            # Here use old_one_hot_actions
            _maybe_get_or_select(  # replace by _rsample_action_log_probs
                tensordict,
                (agent_group, self.sample_log_prob_key),
                tensordict[("advantage")].shape,
            )
        )

        # Then generate updated tensorDict and return for training each agent of group

    def _get_action_grad_of_updated_agents(
        self,
        action_grad_per_agent: torch.Tensor,
        cated_other_agents_factors: torch.Tensor,
        updated_agents_idx: list[int],
        updated_agent: int,
        bs: int = None,
    ):
        # TODO: self.clip_param
        self.clip_param = 0.2
        for agent_id in updated_agents_idx:
            ## compute action grad for this agent
            ## multiply with the cated_other_agents_factors

            multiplier = np.concatenate(
                [
                    cated_other_agents_factors[:agent_id],
                    cated_other_agents_factors[agent_id + 1 :],
                ],
                0,
            )
            multiplier = (
                np.ones((bs, 1, 1), dtype=np.float32)
                if multiplier is None
                else np.prod(multiplier, 0)
            )
            multiplier = np.clip(
                multiplier, 1 - self.clip_param / 2, 1 + self.clip_param / 2
            )
            assert (
                multiplier.shape == self.action_gradients[updated_agent][agent_id].shape
            )
            action_grad_per_agent += (
                self.action_gradients[updated_agent][agent_id] * multiplier
            )

        return action_grad_per_agent

    def _get_factor_for_agent(self, agent_group: str = None) -> torch.Tensor:
        """Get the factor tensor for a specific agent group.

        This returns the current factor that should be used when training the specified agent.

        Args:
            agent_group (str): Name of the agent group.
        """

        group_index = self.group_names.index(agent_group)
        cated_other_agents_factors = np.concatenate(
            [self.factor[:group_index], self.factor[group_index + 1 :]], 0
        )
        prod_factors = np.prod(cated_other_agents_factors, 0)
        print(f' "Agent Group: {agent_group}, Factor Shape: {prod_factors.shape}" ')
        return prod_factors, cated_other_agents_factors

    def _get_action_shape(self, tensordict: TensorDictBase, agent_group: str) -> tuple:
        """Get the action shape for a specific agent group.

        Args:
            agent_group (str): Name of the agent group.
        """
        group_dict = tensordict.get(agent_group)
        action = group_dict.get("action")
        return action.shape

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

        # get new_actions_log_probs
        # https://github.com/LiZhYun/BackPropagationThroughAgents/blob/main/bta/runner/temporal/base_runner.py 349
        new_log_prob, _, _ = self._get_cur_log_prob(  # TODO: _rsample_action_log_probs
            tensordict, self.policy_modules[agent_group]
        )

        # adv batch
        advantage = tensordict[self.advantage_key]

        # BptaActionLoss
        action_loss = self.action_loss(new_log_prob, self.old_log_prob, advantage)

        # update action_grad :TODO -> Numpy? - ohne grad
        # one_hot_actions with grad?
        self._update_action_gradients()
        self.action_gradients[self.group_names.index(agent_group)] = (
            self.action_gradients[self.group_names.index(agent_group)]
        )

        # For other_agent each agent in agents (agent_groups)
        # update action_grad[current_agent][other_agent] with gradients from action_loss

        # update factor with this loss -> Factor contains gradients!
        # TODO To numpy? Aber dann verschwinden die Gradienten!? -> Oder im Optimizer?
        self.factor[self.group_names.index(agent_group)] = (
            self.factor[self.group_names.index(agent_group)]
            * torch.exp(new_log_prob - self.old_log_prob.detach())
        ).detach()

    def _update_action_gradients(self) -> None:
        """
        Updates the action gradients based on the actions and gradients of the updated agent for subsequent agents.

        Args:
            one_hot_actions (torch.Tensor): One-hot encoded actions for the current agent group.

        """

        for agent_id, agent in enumerate(self.agent_order):
            self.one_hot_actions.grad[:, agent_id]
            # reshape

        return None

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
