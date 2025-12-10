from abc import ABC, abstractmethod
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import (
    Compose,
    ParallelEnv,
    RewardSum,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.pettingzoo import PettingZooWrapper

from heteromark.environment import (
    create_dummy_parallel_pz_env,
    create_parallel_pz_env,
)

STANDARD_ENV_TRANSFORMS = Compose(
    StepCounter(),
)


class BaseEnvironmentFactory(ABC):
    """Abstract base class for environment factories."""

    @abstractmethod
    def create(self, config: dict) -> Any:
        """Create and return an environment.

        Args:
            config: Configuration dictionary for environment creation

        Returns:
            Created environment instance
        """
        pass


class EnvironmentFactory(BaseEnvironmentFactory):
    """Factory for creating TorchRL-compatible environments."""

    def __init__(self, env_type: str = "smac"):
        """Initialize environment factory.

        Args:
            env_type: Type of environment to create (default: "smac")
        """
        self.env_type = env_type

    def create(self, config: dict) -> Any:
        """Create environment based on configuration.

        Args:
            config: Configuration dictionary containing environment specs

        Returns:
            Environment instance (potentially wrapped with TransformedEnv)
        """
        if self.env_type == "smac":
            return self._create_smac_env(config)
        elif self.env_type == "custom":
            return self._create_custom_env(config)
        else:
            raise ValueError(f"Unknown environment type: {self.env_type}")

    def _create_smac_env(self, config: dict) -> Any:
        """Create SMAC environment.

        Args:
            config: Configuration dictionary with SMAC specs

        Returns:
            SMAC environment
        """
        # TODO Adapt to generate real environments NOT DUMMY
        if config.get("use_dummy", False):
            env = create_dummy_parallel_pz_env()
            # env = create_parallel_pz_env(config)
            use_mask = False
            env = PettingZooWrapper(
                env=env,
                return_state=False,
                group_map=None,
                use_mask=use_mask,
                done_on_any=False,
            )
            env = self._apply_transforms(env)
            return env
        else:
            env = create_parallel_pz_env(OmegaConf.to_container(config, resolve=True))

        # Apply transformations if specified
        if config.get("transforms"):
            env = self._apply_transforms(env, config["transforms"])

        # Wrap in parallel env if specified
        if config.get("num_parallel_envs", 1) > 1:
            env = ParallelEnv(
                num_workers=config["num_parallel_envs"], create_env_fn=lambda: env
            )

        return env

    def _create_custom_env(self, config: dict) -> Any:
        """Create custom environment from config.

        Args:
            config: Configuration dictionary with custom environment parameters

        Returns:
            Custom environment instance
        """
        # Placeholder for custom environment creation
        raise NotImplementedError("Custom environment creation not yet implemented")

    def _apply_transforms(self, env: Any) -> TransformedEnv:
        """Apply transformations to environment.

        Args:
            env: Base environment
            transforms: List of transform specifications

        Returns:
            Transformed environment
        """
        # This would apply various TorchRL transforms
        # For now, return the environment as-is

        for agent_group in env.group_map.keys():
            STANDARD_ENV_TRANSFORMS.append(
                RewardSum([(agent_group, "reward")], ["reward"])
            )
            self.reward_key = ((agent_group, "reward"), "reward")
            # STANDARD_ENV_TRANSFORMS.append(
            #     SqueezeTransform(dim=0, in_keys=["reward"], out_keys=["reward"])
            # )
            # STANDARD_ENV_TRANSFORMS.append(
            #     SqueezeTransform(dim=-1, in_keys=("reward"), out_keys=("reward"))
            # )
            break

        env = TransformedEnv(env, STANDARD_ENV_TRANSFORMS)
        return env


def get_dummy_env_from_factory():
    env_factory = EnvironmentFactory(env_type="smac")
    config = {
        "map_name": "10gen_terran",
        "distributed_config": {
            "n_units": 5,
            "n_enemies": 5,
            # Additional configuration...
        },
        "use_dummy": True,
        "num_parallel_envs": 2,
        "transforms": [],
    }

    env = env_factory.create(config)

    return env


@hydra.main(version_base=None, config_path="../../../conf", config_name="dummy_config")
def test(config: DictConfig):
    env_factory = EnvironmentFactory(env_type=config.env.env_type)
    env = env_factory.create(config.env)
    env = env_factory._apply_transforms(env)
    print(env.group_map)
    print(" === Environment created :", env, "===")
    return env


if __name__ == "__main__":
    env = test()
    # print(env.group_map)

    # env_factory = EnvironmentFactory(env_type="smac")
    # config = {
    #     "map_name": "10gen_terran",
    #     "distributed_config": {
    #         "n_units": 5,
    #         "n_enemies": 5,
    #         # Additional configuration...
    #     },
    #     "use_dummy": True,
    #     "num_parallel_envs": 1,
    #     "transforms": [],
    # }

    # env = env_factory.create(config)
    # print(" === Environment created:", env, "===")
    # print(" === Start: Apply Transforms === ")
    # env = env_factory._apply_transforms(env)

    # print(" === Finished Applying Transforms === ")
    # print(env.group_map)
