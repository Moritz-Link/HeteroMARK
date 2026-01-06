import contextlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase, is_tensor_collection
from tensordict.nn import (
    CompositeDistribution,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictParams,
    composite_lp_aggregate,
)
from tensordict.utils import NestedKey
from torchrl._utils import _standardize
from torchrl.objectives import PPOLoss
from torchrl.objectives.utils import (
    ValueEstimators,
    _reduce,
    _sum_td_features,
)


class ClipBptaLoss(PPOLoss):
    """Clipped BPTA (Backpropagation Through Agents) loss.

    This class extends ClipPPOLoss with HAPPO-specific functionality, including
    a factor tensor that can be updated to modulate the policy gradient.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage) * factor

    Args:
        actor_network (ProbabilisticTensorDictSequential): policy operator.
        critic_network (ValueOperator): value operator.

    Keyword Args:
        clip_epsilon (scalar, optional): weight clipping threshold in the clipped PPO loss equation.
            default: 0.2
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coeff: (scalar | Mapping[NestedKey, scalar], optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coeff (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        normalize_advantage_exclude_dims (Tuple[int], optional): dimensions to exclude from the advantage
            standardization. Default: ().
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. Default: ``"mean"``.
        clip_value (bool or float, optional): If a ``float`` is provided, it will be used to compute a clipped
            version of the value prediction. Defaults to ``False``.
        device (torch.device, optional): device of the buffers. Defaults to ``None``.
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            advantage (NestedKey): The input tensordict key where the advantage is expected.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is expected.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            sample_log_prob (NestedKey or list of nested keys): The input tensordict key where the
               sample log probability is expected.
               Defaults to ``"sample_log_prob"`` when :func:`~tensordict.nn.composite_lp_aggregate` returns `True`,
                `"action_log_prob"`  otherwise.
            action (NestedKey or list of nested keys): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            reward (NestedKey or list of nested keys): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey or list of nested keys): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey or list of nested keys): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        advantage: NestedKey = "advantage"
        factor: NestedKey = "factor"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        sample_log_prob: NestedKey | list[NestedKey] | None = None
        action: NestedKey | list[NestedKey] = "action"
        reward: NestedKey | list[NestedKey] = "reward"
        done: NestedKey | list[NestedKey] = "done"
        terminated: NestedKey | list[NestedKey] = "terminated"

        def __post_init__(self):
            if self.sample_log_prob is None:
                if composite_lp_aggregate(nowarn=True):
                    self.sample_log_prob = "sample_log_prob"
                else:
                    self.sample_log_prob = "action_log_prob"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys
    default_value_estimator = ValueEstimators.GAE

    actor_network: ProbabilisticTensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coeff: float | Mapping[NestedKey, float] | None = None,
        critic_coeff: float | None = None,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = False,
        normalize_advantage_exclude_dims: tuple[int] = (),
        gamma: float | None = None,
        separate_losses: bool = False,
        reduction: str | None = None,
        clip_value: bool | float | None = None,
        device: torch.device | None = None,
        **kwargs,
    ):
        # Define clipping of the value loss
        if isinstance(clip_value, bool):
            clip_value = clip_epsilon if clip_value else None

        super().__init__(
            actor_network,
            critic_network,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coeff=entropy_coeff,
            critic_coeff=critic_coeff,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            normalize_advantage_exclude_dims=normalize_advantage_exclude_dims,
            gamma=gamma,
            separate_losses=separate_losses,
            reduction=reduction,
            clip_value=clip_value,
            device=device,
            **kwargs,
        )
        if device is None:
            try:
                device = next(self.parameters()).device
            except (AttributeError, StopIteration):
                device = getattr(
                    torch, "get_default_device", lambda: torch.device("cpu")
                )()
        self.register_buffer("clip_epsilon", torch.tensor(clip_epsilon, device=device))

        # Initialize factor tensor as ones
        self.register_buffer("factor", torch.ones(1, device=device))

    @property
    def _clip_bounds(self):
        return (
            (-self.clip_epsilon).log1p(),
            self.clip_epsilon.log1p(),
        )

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective", "clip_fraction"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            if self.clip_value:
                keys.append("value_clip_fraction")
            keys.append("ESS")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # https://github.com/LiZhYun/BackPropagationThroughAgents/blob/main/bta/runner/temporal/base_runner.py
        # Log Prob also with rsample
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(
            self.tensor_keys.advantage, None, as_padded_tensor=True
        )
        # Validate that a TensorDict nested under the configured factor key exists
        if not hasattr(self.tensor_keys, "factor"):
            raise RuntimeError(
                "No factor key configured on self.tensor_keys; expected attribute 'factor' "
                "pointing to a nested key like ('agent', 'factor')."
            )

        factor_key = self.tensor_keys.factor  # TODO: Den Key noch anpassen
        factor_td = tensordict["group_factor"].detach()

        if factor_td is None:
            raise RuntimeError(
                f"Missing TensorDict under key {factor_key!r} in the input tensordict."
            )

        # Ensure the nested TensorDict contains the actual 'factor' tensor

        if advantage is None:
            if self.critic_network is None:
                raise RuntimeError(
                    "Critic network is not specified, cannot compute advantage within forward."
                )
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            if advantage.numel() > tensordict.batch_size.numel() and not len(
                self.normalize_advantage_exclude_dims
            ):
                warnings.warn(
                    "You requested advantage normalization and the advantage key has more dimensions"
                    " than the tensordict batch. Make sure to pass `normalize_advantage_exclude_dims` "
                    "if you want to keep any dimension independent while computing normalization statistics. "
                    "If you are working in multi-agent/multi-objective settings this is highly suggested."
                )
            advantage = _standardize(advantage, self.normalize_advantage_exclude_dims)
        # https://github.com/LiZhYun/BackPropagationThroughAgents/blob/main/bta/algorithms/bta/t_policy.py#L176
        log_weight, dist, kl_approx = self._log_weight(
            tensordict,
            adv_shape=advantage.shape[
                :-1
            ],  # TODO: werden hier die werte direkt upgedated?
        )
        factor_td = torch.clamp(
            factor_td,
            1.0 - self._clip_bounds / 2,
            1.0 + self._clip_bounds / 2,
        )
        # ESS for logging
        with torch.no_grad():
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]
        # Set this always new for each agent but update the grad in algorithm
        # In algorithm
        action_grad = tensordict["action_grad_per_agent"]  # Gradient of action log prob
        train_actions = None  # Actions of this agent

        # TODO when to use log_weight.exp() and log_weight!
        assert log_weight.exp().shape == advantage.shape, (
            f"{log_weight.exp().shape}, {advantage.shape}"
        )
        assert factor_td.shape == advantage.shape, (
            f"{factor_td.shape}, {advantage.shape}"
        )
        surr1 = (
            log_weight.exp() * advantage * factor_td
            + log_weight.exp().detach() * action_grad * train_actions
        )

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
        detached_log_weight_clip = log_weight_clip.detach()

        surr2 = (
            log_weight_clip * advantage * factor_td
            + detached_log_weight_clip * action_grad * train_actions
        )
        # Apply factor to the gains (HAPPO modification)
        # Final policy  policy_action_loss
        gain = torch.stack([surr1, surr2], -1).min(dim=-1).values
        gain = gain * factor_td
        td_out = TensorDict({"loss_objective": -gain})
        td_out.set("clip_fraction", clip_fraction)
        td_out.set("kl_approx", kl_approx.detach().mean())  # for logging

        if self.entropy_bonus:
            entropy = self._get_entropy(dist, adv_shape=advantage.shape[:-1])
            if is_tensor_collection(entropy):
                td_out.set("composite_entropy", entropy.detach())
                td_out.set("entropy", _sum_td_features(entropy).detach().mean())
            else:
                td_out.set("entropy", entropy.detach().mean())
            td_out.set("loss_entropy", self._weighted_loss_entropy(entropy))
        if self._has_critic:
            loss_critic, value_clip_fraction, explained_variance = self.loss_critic(
                tensordict
            )
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)
            if explained_variance is not None:
                td_out.set("explained_variance", explained_variance)

        td_out.set("ESS", _reduce(ess, self.reduction) / batch)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
        )
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
            "critic_network_params",
            "target_actor_network_params",
            "target_critic_network_params",
        )
        return td_out

    def _get_cur_log_prob(self, tensordict):
        if isinstance(
            self.actor_network,
            (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
        ) or hasattr(self.actor_network, "get_dist"):
            # assert tensordict['log_probs'].requires_grad
            # assert tensordict['logits'].requires_grad
            with (
                self.actor_network_params.to_module(self.actor_network)
                if self.functional
                else contextlib.nullcontext()
            ):
                dist = self.actor_network.get_dist(tensordict)
            is_composite = isinstance(dist, CompositeDistribution)
            in_dict_action = tensordict.get(self.tensor_keys.action)
            action = self._rsample_action_log_probs(in_dict_action)

            if action.requires_grad:
                raise Warning(
                    f"tensordict stored {self.tensor_keys.action} requires grad."
                )
            log_prob = dist.log_prob(action)
        else:
            raise NotImplementedError(
                "Only probabilistic modules from tensordict.nn are currently supported. "
                "If you need to implement a custom logic to retrieve the log-probs (to compute "
                "the PPO objective) or the distribution (for the PPO entropy), please augment "
                f"the {type(self).__class__} by implementing your own logic in _get_cur_log_prob."
            )

        return log_prob, dist, is_composite

    def _rsample_action_log_probs(self, action) -> tuple:
        """
        Sample actions and compute log probabilities.
        Apply Reparameterization trick for  actions.

        Inputs:
            action: Actions taken by the agent. -> Batch

        Returns:
            train_actions: Sampled actions after reparameterization.
            action_log_probs: Log probabilities of the sampled actions.

        """
        # action_logits: Distribution with the logits! -> dist

        dist = self._get_dist_of_actor()

        if self.continuous_action:
            train_actions_soft = dist.mean
            train_actions = action - train_actions_soft.detach() + train_actions_soft

        elif self.discrete_action:
            train_actions_soft = dist.rsample(
                hard=False, tau=self.tau
            )  # TODO: Define tau
            train_actions_soft_ = train_actions_soft.gather(1, action.long())
            index = action
            train_actions_hard = torch.zeros_like(
                train_actions_soft, memory_format=torch.legacy_contiguous_format
            ).scatter_(-1, index.long(), 1.0)
            train_actions_soft = torch.zeros_like(
                train_actions_soft, memory_format=torch.legacy_contiguous_format
            ).scatter_(-1, index.long(), train_actions_soft_)
            train_actions = (
                train_actions_hard - train_actions_soft.detach() + train_actions_soft
            )

        return train_actions


class BptaActionLoss(torch.nn.Module):
    """BPTA action loss module.

    This module computes the action loss for BPTA (Backpropagation Through Agents) algorithms.
    It calculates the log probabilities of actions taken by agents using the reparameterization trick.

    Args:
        continuous_action (bool): Indicates if the action space is continuous.
        discrete_action (bool): Indicates if the action space is discrete.
        tau (float, optional): Temperature parameter for Gumbel-Softmax sampling in discrete action spaces.
            Defaults to 1.0.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, new_log_prob, old_log_prob, advantage) -> torch.Tensor:
        """Compute the action loss.

        Args:
            new_log_prob (torch.Tensor): Log probabilities of the current actions.
            old_log_prob (torch.Tensor): Log probabilities of the previous actions.
            advantage (torch.Tensor): Advantage values for the actions.
        Returns:
            torch.Tensor: The computed action loss.
        """
        # Action loss
        action_loss = torch.sum(
            torch.prod(
                torch.exp(new_log_prob - old_log_prob.detach()),
                -1,
                keepdim=True,
            )
            * advantage,
            dim=-1,
            keepdim=True,
        )
        return action_loss
