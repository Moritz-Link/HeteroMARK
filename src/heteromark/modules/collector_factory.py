from abc import ABC, abstractmethod
from typing import Any

import torch
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector


class BaseCollectorFactory(ABC):
    """Abstract base class for collector factories."""

    @abstractmethod
    def create(self, config: dict, env: Any, policy_modules: dict) -> Any:
        """Create and return a data collector.

        Args:
            config: Configuration dictionary for collector creation
            env: Environment instance
            policy_modules: Dictionary of policy modules by agent group

        Returns:
            Data collector instance
        """
        pass


class CollectorFactory(BaseCollectorFactory):
    """Factory for creating TorchRL data collectors."""

    def __init__(self, collector_type: str = "sync"):
        """Initialize collector factory.

        Args:
            collector_type: Type of collector (default: "sync")
        """
        self.collector_type = collector_type

    def create(self, config: dict, env: Any, policy_modules: dict) -> Any:
        """Create collector based on configuration.

        Args:
            config: Configuration dictionary containing collector specs
            env: Environment instance
            policy_modules: Dictionary of policy modules

        Returns:
            Collector instance
        """
        if self.collector_type == "sync":
            return self._create_sync_collector(config, env, policy_modules)
        elif self.collector_type == "multi_sync":
            return self._create_multi_sync_collector(config, env, policy_modules)
        else:
            raise ValueError(f"Unknown collector type: {self.collector_type}")

    def _create_sync_collector(
        self, config: dict, env: Any, policy_modules: dict
    ) -> SyncDataCollector:
        """Create synchronous data collector.

        Args:
            config: Configuration with collector parameters
            env: Environment instance
            policy_modules: Dictionary of policy modules

        Returns:
            SyncDataCollector instance
        """
        frames_per_batch = config.get("frames_per_batch", 1000)
        total_frames = config.get("total_frames", 1000000)
        device = torch.device(config.get("device", "cpu"))

        # Create a combined policy module if there are multiple agents
        if len(policy_modules) > 1:
            # For multi-agent, we need to handle policy execution differently
            # This is a simplified approach - you may need custom logic
            policy = self._create_multi_agent_policy(policy_modules)
        else:
            policy = list(policy_modules.values())[0]

        collector = SyncDataCollector(
            create_env_fn=lambda: env,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
            storing_device=device,
        )

        return collector

    def _create_multi_sync_collector(
        self, config: dict, env: Any, policy_modules: dict
    ) -> MultiSyncDataCollector:
        """Create multi-process synchronous data collector.

        Args:
            config: Configuration with collector parameters
            env: Environment instance
            policy_modules: Dictionary of policy modules

        Returns:
            MultiSyncDataCollector instance
        """
        frames_per_batch = config.get("frames_per_batch", 1000)
        total_frames = config.get("total_frames", 1000000)
        num_collectors = config.get("num_collectors", 4)
        device = torch.device(config.get("device", "cpu"))

        # Create a combined policy module if there are multiple agents
        if len(policy_modules) > 1:
            policy = self._create_multi_agent_policy(policy_modules)
        else:
            policy = list(policy_modules.values())[0]

        collector = MultiSyncDataCollector(
            create_env_fn=[lambda: env] * num_collectors,
            policy=policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            device=device,
        )

        return collector

    def _create_multi_agent_policy(self, policy_modules: dict) -> Any:
        """Create a combined policy for multi-agent scenarios.

        Args:
            policy_modules: Dictionary of policy modules by agent group

        Returns:
            Combined policy module
        """
        # This is a simplified implementation
        # In practice, you'd need a custom TensorDictModule that handles
        # multiple agent groups and routes observations to the right policies
        from tensordict.nn import TensorDictSequential

        # Create a sequential module that applies all policies
        modules = list(policy_modules.values())
        if len(modules) == 1:
            return modules[0]

        # For true multi-agent support, you'd need custom logic here
        # This is a placeholder that would need to be adapted to your specific use case
        return TensorDictSequential(*modules)


def get_dummy_collector(env, policy_modules):
    config = {}
    collector_factory = CollectorFactory("sync")
    collector_factory.create(config=None, env=env, policy_modules=policy_modules)


if __name__ == "__main__":
    from heteromark.modules.environment_factory import get_dummy_env_from_factory
    from heteromark.modules.policy_factory import get_dummy_policy_from_factory

    env = get_dummy_env_from_factory()
    policy_modules, _ = get_dummy_policy_from_factory(env)
    collector_factory = CollectorFactory("sync")
    collector_factory.create(config=None, env=None, policy_modules=None)
