from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
from torch import nn
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential
from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
    MLP,
)


class BasePolicyFactory(ABC):
    """Abstract base class for policy factories."""

    @abstractmethod
    def create(self, config: dict, env: Any) -> tuple[Any, Any]:
        """Create and return policy and value modules.

        Args:
            config: Configuration dictionary for policy creation
            env: Environment instance to extract observation/action specs

        Returns:
            Tuple of (policy_module, value_module)
        """
        pass


class PolicyFactory(BasePolicyFactory):
    """Factory for creating actor-critic policy modules."""

    def __init__(self, policy_type: str = "mlp"):
        """Initialize policy factory.

        Args:
            policy_type: Type of policy network architecture (default: "mlp")
        """
        self.policy_type = policy_type

    def create(self, config: dict, env: Any) -> tuple[Any, Any]:
        """Create policy and value modules based on configuration.

        Args:
            config: Configuration dictionary containing network specs
            env: Environment instance

        Returns:
            Tuple of (policy_module, value_module) dictionaries by agent group
        """
        if self.policy_type == "mlp":
            return self._create_mlp_policy(config, env)
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")

    def _create_mlp_policy(self, config: dict, env: Any) -> tuple[Dict, Dict]:
        """Create MLP-based policy and value networks.

        Args:
            config: Configuration with network architecture specs
            env: Environment instance to get observation/action specs

        Returns:
            Tuple of (policy_modules_dict, value_modules_dict)
        """
        device = torch.device(config.get("device", "cpu"))
        hidden_sizes = config.get("hidden_sizes", [64, 64])
        activation_class = getattr(nn, config.get("activation", "Tanh"))

        policy_modules = {}
        value_modules = {}

        # Get agent groups from environment
        group_map = getattr(env, "group_map", {"default": env.agents if hasattr(env, "agents") else []})

        for agent_group in group_map.keys():
            # Extract observation and action specs for this agent group
            obs_spec = self._get_observation_spec(env, agent_group)
            action_spec = self._get_action_spec(env, agent_group)
            obs_dim = obs_spec[agent_group]["observation"]["observation"].shape[-1]
            action_dim = action_spec[agent_group]["action"].shape[-1]
            #obs_dim = obs_spec.shape[-1] if hasattr(obs_spec, "shape") else config.get("obs_dim", 64)
            #action_dim = action_spec.shape[-1] if hasattr(action_spec, "shape") else config.get("action_dim", 4)

            # Create actor network
            actor_net = MLP(
                in_features=obs_dim,
                out_features=action_dim,
                num_cells=hidden_sizes,
                activation_class=activation_class,
                device=device,
            )

            # Wrap in TensorDictModule
            policy_module = TensorDictModule(
                actor_net,
                in_keys=[(agent_group, "observation")],
                out_keys=[(agent_group, "action")],
            )

            # Create critic network
            critic_net = MLP(
                in_features=obs_dim,
                out_features=1,
                num_cells=hidden_sizes,
                activation_class=activation_class,
                device=device,
            )

            value_module = ValueOperator(
                module=critic_net,
                in_keys=[(agent_group, "observation")],
                out_keys=[(agent_group, "state_value")],
            )

            policy_modules[agent_group] = policy_module
            value_modules[agent_group] = value_module

        return policy_modules, value_modules

    def _get_observation_spec(self, env: Any, agent_group: str) -> Any:
        """Extract observation spec for an agent group.

        Args:
            env: Environment instance
            agent_group: Name of agent group

        Returns:
            Observation spec
        """
        if hasattr(env, "observation_spec"):
            obs_spec = env.observation_spec
            if isinstance(obs_spec, dict) and agent_group in obs_spec:
                return obs_spec[agent_group]
            return obs_spec
        return None

    def _get_action_spec(self, env: Any, agent_group: str) -> Any:
        """Extract action spec for an agent group.

        Args:
            env: Environment instance
            agent_group: Name of agent group

        Returns:
            Action spec
        """
        if hasattr(env, "action_spec"):
            action_spec = env.action_spec
            if isinstance(action_spec, dict) and agent_group in action_spec:
                return action_spec[agent_group]
            return action_spec
        return None




if __name__ == "__main__":
    # Example usage
    from heteromark.modules.environment_factory import EnvironmentFactory

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

    policy_factory = PolicyFactory(policy_type="mlp")
    config = {
        "hidden_sizes": [128, 128],
        "activation": "ReLU",
        "device": "cpu",
    }
    policy_modules, value_modules = policy_factory.create(config, env)

    print("Policy Modules:", policy_modules)
    print("Value Modules:", value_modules)
    print(" === Policy and Value modules created ===")