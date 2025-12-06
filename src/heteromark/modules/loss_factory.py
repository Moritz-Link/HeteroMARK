from abc import ABC, abstractmethod

from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from heteromark.loss.happo_loss import ClipHAPPOLoss


class BaseLossFactory(ABC):
    """Abstract base class for loss factories."""

    @abstractmethod
    def create(
        self, config: dict, policy_modules: dict, value_modules: dict
    ) -> tuple[dict, dict]:
        """Create and return loss modules and advantage estimators.

        Args:
            config: Configuration dictionary for loss creation
            policy_modules: Dictionary of policy modules by agent group
            value_modules: Dictionary of value modules by agent group

        Returns:
            Tuple of (loss_modules_dict, advantage_modules_dict)
        """
        pass


class LossFactory(BaseLossFactory):
    """Factory for creating loss modules and advantage estimators."""

    def __init__(self, loss_type: str = "happo"):
        """Initialize loss factory.

        Args:
            loss_type: Type of loss function (default: "happo")
        """
        self.loss_type = loss_type

    def create(
        self, config: dict, policy_modules: dict, value_modules: dict
    ) -> tuple[dict, dict]:
        """Create loss modules based on configuration.

        Args:
            config: Configuration dictionary containing loss specs
            policy_modules: Dictionary of policy modules by agent group
            value_modules: Dictionary of value modules by agent group

        Returns:
            Tuple of (loss_modules, advantage_modules) dictionaries by agent group
        """
        if self.loss_type == "happo":
            return self._create_happo_loss(config, policy_modules, value_modules)
        elif self.loss_type == "ppo":
            return self._create_ppo_loss(config, policy_modules, value_modules)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _create_happo_loss(
        self, config: dict, policy_modules: dict, value_modules: dict
    ) -> tuple[dict, dict]:
        """Create HAPPO loss modules.

        Args:
            config: Configuration with loss parameters
            policy_modules: Dictionary of policy modules
            value_modules: Dictionary of value modules

        Returns:
            Tuple of (loss_modules_dict, advantage_modules_dict)
        """
        loss_modules = {}
        advantage_modules = {}

        # Loss hyperparameters
        clip_epsilon = config.get("clip_epsilon", 0.2)
        entropy_coeff = config.get("entropy_coeff", 0.01)
        critic_coeff = config.get("critic_coeff", 1.0)
        gamma = config.get("gamma", 0.99)
        lmbda = config.get("lmbda", 0.95)

        for agent_group in policy_modules.keys():
            # Create GAE advantage estimator
            advantage_module = GAE(
                gamma=gamma,
                lmbda=lmbda,
                value_network=value_modules[agent_group],
                average_gae=False,
            )

            # Create HAPPO loss module
            loss_module = ClipHAPPOLoss(
                actor_network=policy_modules[agent_group],
                critic_network=value_modules[agent_group],
                clip_epsilon=clip_epsilon,
                entropy_coeff=entropy_coeff,
                critic_coeff=critic_coeff,
                normalize_advantage=config.get("normalize_advantage", True),
            )

            # Set appropriate keys for multi-agent setup
            loss_module.set_keys(
                advantage=(agent_group, "advantage"),
                value_target=(agent_group, "value_target"),
                value=(agent_group, "state_value"),
                sample_log_prob=(agent_group, "action_log_prob"),
                action=(agent_group, "action"),
                reward=(agent_group, "reward"),
                done=(agent_group, "done"),
                terminated=(agent_group, "terminated"),
                factor=(agent_group, "factor"),
            )

            loss_modules[agent_group] = loss_module
            advantage_modules[agent_group] = advantage_module

        return loss_modules, advantage_modules

    def _create_ppo_loss(
        self, config: dict, policy_modules: dict, value_modules: dict
    ) -> tuple[dict, dict]:
        """Create PPO loss modules.

        Args:
            config: Configuration with loss parameters
            policy_modules: Dictionary of policy modules
            value_modules: Dictionary of value modules

        Returns:
            Tuple of (loss_modules_dict, advantage_modules_dict)
        """
        loss_modules = {}
        advantage_modules = {}

        # Loss hyperparameters
        clip_epsilon = config.get("clip_epsilon", 0.2)
        entropy_coeff = config.get("entropy_coeff", 0.01)
        critic_coeff = config.get("critic_coeff", 1.0)
        gamma = config.get("gamma", 0.99)
        lmbda = config.get("lmbda", 0.95)

        for agent_group in policy_modules.keys():
            # Create GAE advantage estimator
            # TODO: ADvantage Module immer raus -> Nur CTDE
            advantage_module = GAE(
                gamma=gamma,
                lmbda=lmbda,
                value_network=value_modules[agent_group],
                average_gae=False,
            )
            # Create PPO loss module
            loss_module = ClipPPOLoss(
                actor_network=policy_modules[agent_group],
                critic_network=value_modules[agent_group],
                clip_epsilon=clip_epsilon,
                entropy_coeff=entropy_coeff,
                critic_coeff=critic_coeff,
                normalize_advantage=config.get("normalize_advantage", True),
            )

            # Set appropriate keys for multi-agent setup
            loss_module.set_keys(
                advantage=(agent_group, "advantage"),
                value_target=(agent_group, "value_target"),
                value=(agent_group, "state_value"),
                sample_log_prob=(agent_group, "action_log_prob"),
                action=(agent_group, "action"),
                reward=(agent_group, "reward"),
                done=(agent_group, "done"),
                terminated=(agent_group, "terminated"),
            )

            loss_modules[agent_group] = loss_module
            advantage_modules[agent_group] = advantage_module

        return loss_modules, advantage_modules


# class enum LossTypes:
#     "ppo": Clip


def get_dummy_loss_modules_from_factory(policy_modules, value_modules):
    loss_factory = LossFactory("ppo")
    config = {}
    loss_modules, advantage_modules = loss_factory.create(
        config=config, policy_modules=policy_modules, value_modules=value_modules
    )

    return loss_modules, advantage_modules


if __name__ == "__main__":
    from heteromark.modules.environment_factory import get_dummy_env_from_factory
    from heteromark.modules.policy_factory import get_dummy_policy_from_factory

    env = get_dummy_env_from_factory()
    policy_modules, value_modules = get_dummy_policy_from_factory(env)

    loss_factory = LossFactory("ppo")
    config = {}
    loss_modules, advantage_modules = loss_factory.create(
        config=config, policy_modules=policy_modules, value_modules=value_modules
    )
    print("=== Loss Modules ===")
