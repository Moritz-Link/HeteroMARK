"""
Factory-based training script using TorchRL and modular components.

This script demonstrates a clean, modular approach to training multi-agent
reinforcement learning systems using the factory design pattern.
"""

import torch
from tqdm import tqdm
from typing import Dict, Any
from omegaconf import DictConfig

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from heteromark.modules import (
    EnvironmentFactory,
    PolicyFactory,
    LossFactory,
    OptimizerFactory,
    CollectorFactory,
    ReplayBufferFactory,
)
from heteromark.algorithm.happo_algorithm import HappoAlgorithm


class ComponentFactory:
    """Factory for creating all training components."""

    def __init__(self, config: DictConfig):
        """Initialize component factory with configuration.

        Args:
            config: Configuration object containing all parameters
        """
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

        # Initialize factories
        self.env_factory = EnvironmentFactory(
            env_type=config.get("env_type", "smac")
        )
        self.policy_factory = PolicyFactory(
            policy_type=config.get("policy_type", "mlp")
        )
        self.loss_factory = LossFactory(
            loss_type=config.get("loss_type", "happo")
        )
        self.optimizer_factory = OptimizerFactory(
            optimizer_type=config.get("optimizer_type", "adam")
        )
        self.collector_factory = CollectorFactory(
            collector_type=config.get("collector_type", "sync")
        )
        self.buffer_factory = ReplayBufferFactory(
            buffer_type=config.get("buffer_type", "tensor")
        )

    def create_components(self) -> Dict[str, Any]:
        """Create all training components using factories.

        Returns:
            Dictionary containing all initialized components
        """
        components = {}

        # Create environment
        components["env"] = self.env_factory.create(self.config.environment)

        # Create policy and value networks
        components["policy_modules"], components["value_modules"] = self.policy_factory.create(
            self.config.policy, components["env"]
        )

        # Create loss modules and advantage estimators
        components["loss_modules"], components["advantage_modules"] = self.loss_factory.create(
            self.config.loss, components["policy_modules"], components["value_modules"]
        )

        # Create optimizers
        components["optimizers"] = self.optimizer_factory.create(
            self.config.optimizer, components["loss_modules"]
        )

        # Create replay buffers
        buffer_config = {
            **self.config.replay_buffer,
            "agent_groups": list(components["policy_modules"].keys()),
        }
        components["replay_buffers"] = self.buffer_factory.create(buffer_config)

        # Create data collector
        components["collector"] = self.collector_factory.create(
            self.config.collector, components["env"], components["policy_modules"]
        )

        # Initialize HAPPO algorithm if using HAPPO loss
        if self.config.get("loss_type") == "happo":
            components["happo_algorithm"] = HappoAlgorithm(
                agent_groups=components["env"].group_map,
                policy_modules=components["policy_modules"],
                sample_log_prob_key="action_log_prob",
                device=self.device,
            )
        else:
            components["happo_algorithm"] = None

        components["device"] = self.device

        return components


def process_batch(tensordict_data: Any, agent_group: str, device: torch.device) -> Any:
    """Process batch for specific agent group.

    Args:
        tensordict_data: Raw tensordict from collector
        agent_group: Name of agent group to process
        device: Device to move batch to

    Returns:
        Processed batch for agent group
    """
    # Move to device
    group_batch = tensordict_data.to(device)

    # Apply any agent-specific preprocessing here
    # This is a placeholder - adapt based on your data structure

    return group_batch.reshape(-1)


def train(components: Dict[str, Any], config: DictConfig) -> Dict[str, Any]:
    """Execute training loop with provided components.

    Args:
        components: Dictionary containing all training components
        config: Configuration object with training parameters

    Returns:
        Trained policy modules
    """
    # Extract components
    env = components["env"]
    policy_modules = components["policy_modules"]
    loss_modules = components["loss_modules"]
    optimizers = components["optimizers"]
    replay_buffers = components["replay_buffers"]
    collector = components["collector"]
    happo_algorithm = components["happo_algorithm"]
    device = components["device"]

    # Training parameters
    total_frames = config.training.total_frames
    num_epochs = config.training.num_epochs
    max_grad_norm = config.training.get("max_grad_norm", 1.0)

    pbar = tqdm(total=total_frames, desc="Training")
    frames = 0

    for i, tensordict_data in enumerate(collector):
        batch_frames = tensordict_data.numel()

        # HAPPO-specific: Reset factor for new batch
        if happo_algorithm:
            happo_algorithm.reset_factor(tensordict_data)
            agent_order = happo_algorithm.get_agent_order()
        else:
            agent_order = list(policy_modules.keys())

        # Training epochs
        for epoch_idx in range(num_epochs):
            # Iterate over agent groups
            for agent_group in agent_order:
                # Process batch for this agent group
                group_batch = process_batch(tensordict_data, agent_group, device)
                group_buffer = replay_buffers[agent_group]
                group_loss_module = loss_modules[agent_group]
                group_optimizer = optimizers[agent_group]

                # Compute advantages
                with torch.no_grad():
                    group_loss_module.value_estimator(
                        group_batch,
                        params=group_loss_module.critic_network_params,
                        target_params=group_loss_module.target_critic_network_params,
                    )

                    # Add HAPPO factor if using HAPPO
                    if happo_algorithm:
                        factor = happo_algorithm.get_factor_for_agent(agent_group)
                        group_batch.set((agent_group, "factor"), factor)

                # Add to replay buffer
                group_buffer.extend(group_batch.to(device))

                # Training on mini-batches
                for batch in group_buffer:
                    batch = batch.to(device)

                    # Compute loss
                    loss_vals = group_loss_module(batch)
                    total_loss = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals.get("loss_entropy", 0.0)
                    )

                    # Backward pass
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        group_loss_module.parameters(), max_grad_norm
                    )

                    # Optimizer step
                    group_optimizer.step()
                    group_optimizer.zero_grad()

                # Update HAPPO factor after training this agent
                if happo_algorithm:
                    happo_algorithm.update_factor(group_batch, agent_group)

        frames += batch_frames
        pbar.update(batch_frames)

        # Check if training is complete
        if frames >= total_frames:
            break

    pbar.close()
    return policy_modules


def train_with_factories(config: DictConfig) -> Dict[str, Any]:
    """Main training function using factory pattern.

    Args:
        config: Configuration object with all training parameters

    Returns:
        Trained policy modules
    """
    # Create component factory
    factory = ComponentFactory(config)

    # Create all components
    components = factory.create_components()

    # Train using the components
    policy_modules = train(components, config)

    return policy_modules


if __name__ == "__main__":
    # Example configuration
    from omegaconf import OmegaConf

    config = OmegaConf.create({
        "device": "cpu",
        "env_type": "smac",
        "policy_type": "mlp",
        "loss_type": "happo",
        "optimizer_type": "adam",
        "collector_type": "sync",
        "buffer_type": "tensor",
        "environment": {
            "use_dummy": True,
            "num_parallel_envs": 1,
        },
        "policy": {
            "hidden_sizes": [64, 64],
            "activation": "Tanh",
            "device": "cpu",
        },
        "loss": {
            "clip_epsilon": 0.2,
            "entropy_coeff": 0.01,
            "critic_coeff": 1.0,
            "gamma": 0.99,
            "lmbda": 0.95,
            "normalize_advantage": True,
        },
        "optimizer": {
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "eps": 1e-8,
        },
        "collector": {
            "frames_per_batch": 1000,
            "total_frames": 100000,
            "device": "cpu",
        },
        "replay_buffer": {
            "batch_size": 256,
            "buffer_size": 10000,
        },
        "training": {
            "total_frames": 100000,
            "num_epochs": 4,
            "max_grad_norm": 1.0,
        },
    })

    # Train
    trained_policies = train_with_factories(config)
    print("Training completed!")
