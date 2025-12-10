from abc import ABC, abstractmethod

from torch.optim import SGD, Adam, RMSprop


class BaseOptimizerFactory(ABC):
    """Abstract base class for optimizer factories."""

    @abstractmethod
    def create(self, config: dict, loss_modules: dict) -> dict:
        """Create and return optimizers.

        Args:
            config: Configuration dictionary for optimizer creation
            loss_modules: Dictionary of loss modules by agent group

        Returns:
            Dictionary of optimizers by agent group
        """
        pass


class OptimizerFactory(BaseOptimizerFactory):
    """Factory for creating optimizers."""

    def __init__(self, optimizer_type: str = "adam"):
        """Initialize optimizer factory.

        Args:
            optimizer_type: Type of optimizer (default: "adam")
        """
        self.optimizer_type = optimizer_type

    def create(self, config: dict, loss_modules: dict) -> dict:
        """Create optimizers based on configuration.

        Args:
            config: Configuration dictionary containing optimizer specs
            loss_modules: Dictionary of loss modules by agent group

        Returns:
            Dictionary of optimizers by agent group
        """
        if self.optimizer_type == "adam":
            return self._create_adam_optimizers(config, loss_modules)
        elif self.optimizer_type == "sgd":
            return self._create_sgd_optimizers(config, loss_modules)
        elif self.optimizer_type == "rmsprop":
            return self._create_rmsprop_optimizers(config, loss_modules)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def _create_adam_optimizers(self, config: dict, loss_modules: dict) -> dict:
        """Create Adam optimizers.

        Args:
            config: Configuration with optimizer parameters
            loss_modules: Dictionary of loss modules

        Returns:
            Dictionary of Adam optimizers by agent group
        """
        optimizers = {}
        lr = config.get("learning_rate", 3e-4)
        weight_decay = config.get("weight_decay", 0.0)
        eps = config.get("eps", 1e-8)

        for agent_group, loss_module in loss_modules.items():
            optimizer = Adam(
                loss_module.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
            )
            optimizers[agent_group] = optimizer

        return optimizers

    def _create_sgd_optimizers(self, config: dict, loss_modules: dict) -> dict:
        """Create SGD optimizers.

        Args:
            config: Configuration with optimizer parameters
            loss_modules: Dictionary of loss modules

        Returns:
            Dictionary of SGD optimizers by agent group
        """
        optimizers = {}
        lr = config.get("learning_rate", 1e-2)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 0.0)

        for agent_group, loss_module in loss_modules.items():
            optimizer = SGD(
                loss_module.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
            optimizers[agent_group] = optimizer

        return optimizers

    def _create_rmsprop_optimizers(self, config: dict, loss_modules: dict) -> dict:
        """Create RMSprop optimizers.

        Args:
            config: Configuration with optimizer parameters
            loss_modules: Dictionary of loss modules

        Returns:
            Dictionary of RMSprop optimizers by agent group
        """
        optimizers = {}
        lr = config.get("learning_rate", 1e-3)
        alpha = config.get("alpha", 0.99)
        eps = config.get("eps", 1e-8)
        weight_decay = config.get("weight_decay", 0.0)

        for agent_group, loss_module in loss_modules.items():
            optimizer = RMSprop(
                loss_module.parameters(),
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
            )
            optimizers[agent_group] = optimizer

        return optimizers


def get_dummy_loss_from_factory(loss_modules):
    optmizer_factory = OptimizerFactory("adam")
    config = {}
    optimizer = optmizer_factory.create(loss_modules=loss_modules, config=config)
    return optimizer


import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../../conf", config_name="dummy_config")
def test(config: DictConfig):
    from heteromark.modules.environment_factory import EnvironmentFactory
    from heteromark.modules.loss_factory import LossFactory
    from heteromark.modules.policy_factory import PolicyFactory

    env_factory = EnvironmentFactory(env_type=config.env.env_type)
    env = env_factory.create(config.env)
    # env = env_factory._apply_transforms(env)
    print(" === Environment created :", env, "===")

    policy_factory = PolicyFactory(policy_type=config.components.policy.policy_type)
    policy_modules, value_modules = policy_factory.create(config.components.policy, env)

    print(" === Policy and Value modules created ===")
    loss_factory = LossFactory(config.components.loss.loss_type)

    loss_modules, advantage_modules = loss_factory.create(
        config=config.components.loss,
        policy_modules=policy_modules,
        value_modules=value_modules,
    )
    print("=== Loss Modules ===")

    optmizer_factory = OptimizerFactory(config.components.optimizer.optimizer_type)
    optimizer = optmizer_factory.create(
        loss_modules=loss_modules, config=config.components.optimizer
    )
    print("=== Optimizers ===")


if __name__ == "__main__":
    test()

    # from heteromark.modules.environment_factory import get_dummy_env_from_factory
    # from heteromark.modules.loss_factory import get_dummy_loss_modules_from_factory
    # from heteromark.modules.policy_factory import get_dummy_policy_from_factory

    # env = get_dummy_env_from_factory()
    # policy_modules, value_modules = get_dummy_policy_from_factory(env)
    # loss_modules, _ = get_dummy_loss_modules_from_factory(policy_modules, value_modules)

    # optmizer_factory = OptimizerFactory("adam")
    # config = {}
    # optimizer = optmizer_factory.create(loss_modules=loss_modules, config=config)
    # print("=== Optimizers ===")
