"""
Factory-based training script using TorchRL and modular components.

This script demonstrates a clean, modular approach to training multi-agent
reinforcement learning systems using the factory design pattern.
"""

from functools import partial
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from heteromark.algorithm import algorithm_factory
from heteromark.loss import update_actor, update_critic
from heteromark.modules import (
    CollectorFactory,
    EnvironmentFactory,
    LoggerFactory,
    LossFactory,
    OptimizerFactory,
    PolicyFactory,
    ReplayBufferFactory,
)
from heteromark.training.utils import log_info


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
        self.env_factory = EnvironmentFactory(env_type=config.env.env_type)
        self.policy_factory = PolicyFactory(
            policy_type=config.components.policy.policy_type
        )
        self.loss_factory = LossFactory(loss_type=config.algorithm.loss.loss_type)
        self.optimizer_factory = OptimizerFactory(
            optimizer_type=config.components.optimizer.optimizer_type
        )
        self.collector_factory = CollectorFactory(
            collector_type=config.components.collector.collector_type
        )
        self.buffer_factory = ReplayBufferFactory(
            buffer_type=config.components.replay_buffer.buffer_type
        )

        self.algorithm_factory = partial(
            algorithm_factory,
            name=config.algorithm.loss.loss_type,
        )  # Placeholder for future algorithm factory

        self.logger_factory = LoggerFactory(config.components.logger)

    def create_components(self) -> dict[str, Any]:
        """Create all training components using factories.

        Returns:
            Dictionary containing all initialized components
        """
        # TODO: Alle der Config anpassen
        components = {}

        # Create environment
        components["env"] = self.env_factory.create(self.config.env)

        # Create policy and value networks
        components["policy_modules"], components["value_modules"] = (
            self.policy_factory.create(self.config.components.policy, components["env"])
        )

        # Create loss modules and advantage estimators
        components["loss_modules"], components["advantage_modules"] = (
            self.loss_factory.create(
                self.config.algorithm.loss,
                components["policy_modules"],
                components["value_modules"],
            )
        )

        # Create optimizers
        components["optimizers"] = self.optimizer_factory.create(
            self.config.components.optimizer, components["loss_modules"]
        )

        # Create replay buffers
        # buffer_config = {
        #     **self.config.components.replay_buffer,
        #     "agent_groups": list(components["policy_modules"].keys()),
        # }
        components["replay_buffers"] = self.buffer_factory.create(
            self.config.components.replay_buffer, components["env"]
        )

        # Create data collector
        components["collector"] = self.collector_factory.create(
            self.config.components.collector,
            components["env"],
            components["policy_modules"],
        )
        kwargs = {
            "agent_groups": components["env"].group_map,
            "policy_modules": components["policy_modules"],
            "sample_log_prob_key": "log_prob",
            "device": self.device,
        }
        components["algorithm"] = self.algorithm_factory(**kwargs)
        # Initialize HAPPO algorithm if using HAPPO loss
        # if self.config.components.loss.loss_type == "happo":
        #     components["happo_algorithm"] = HappoAlgorithm(
        #         agent_groups=components["env"].group_map,
        #         policy_modules=components["policy_modules"],
        #         sample_log_prob_key="log_prob",
        #         device=self.device,
        #     )
        # else:
        #     components["happo_algorithm"] = None

        components["device"] = self.device
        components["logger"] = self.logger_factory.create(log_dir=self.config.log_dir)

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


from heteromark.training.utils import (
    filter_tensordict_by_agent,
)


def train(components: dict[str, Any], config: DictConfig) -> dict[str, Any]:
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
    algorithm = components["algorithm"]
    device = components["device"]
    advantage_module = components["advantage_modules"]
    logger = components["logger"]

    # Training parameters
    total_frames = config.training.total_frames
    num_epochs = config.training.num_epochs
    max_grad_norm = config.training.get("max_grad_norm", 1.0)

    pbar = tqdm(total=total_frames, desc="Training")
    frames = 0
    agent_groups = env.group_map
    random_agent_group = list(agent_groups.keys())[0]
    for _, tensordict_data in enumerate(collector):
        batch_frames = tensordict_data.numel()

        ### ADapt the reward #TODO THe Value function -> Critic doesnt make sense at the moment
        tensordict_data["reward"] = tensordict_data["reward"][:, 0]
        tensordict_data[("next", "reward")] = tensordict_data[("next", "reward")][:, 0]

        # tensordict_data[("critic", "observation")] = tensordict_data[
        #     (random_agent_group, "observation", "observation")
        # ][:, 0]
        # tensordict_data[("next", "critic", "observation")] = tensordict_data[
        #     ("next", random_agent_group, "observation", "observation")
        # ][:, 0]

        # HAPPO-specific: Reset factor for new batch

        agent_order = algorithm.get_agent_order()
        # print("Agent order:", agent_order)
        # Training epochs

        for epoch_idx in range(num_epochs):
            # Compute advantages
            with torch.no_grad():
                advantage_module(tensordict_data)

            algorithm.prepare_rollout(tensordict_data)
            # algorithm.set_adv_as_factor(tensordict_data)

            for agent_group in agent_order:
                algorithm.prepare_training(tensordict_data, agent_group)
                group_buffer = replay_buffers[agent_group]
                group_loss_module = loss_modules[agent_group]
                group_optimizer = optimizers[agent_group]

                for agent in env.group_map[agent_group]:
                    filtered_td = filter_tensordict_by_agent(
                        tensordict_data, agent_name=agent, agent_group=agent_group
                    )
                    algorithm.pre_update(filtered_td, agent_group)

                    group_buffer.empty()

                    # Add to replay buffer
                    group_buffer.extend(filtered_td.to(device))

                    # Training on mini-batches
                    for batch in group_buffer:
                        # Update actor network
                        update_actor(
                            batch=batch,
                            optimizer=group_optimizer,
                            loss_module=group_loss_module,
                            device=device,
                            max_grad_norm=max_grad_norm,
                            step=frames,
                            logger=None,
                        )

                # Update HAPPO factor after training this agent group
                # For each Agent
                algorithm.post_update(tensordict_data, agent_group)

            update_critic(
                replay_buffers,
                optimizers,
                loss_modules,
                tensordict_data,
                frames,
                device,
                None,
            )

        logger["training"].log_steps(step=frames, data=tensordict_data)
        frames += batch_frames

        pbar.update(batch_frames)

        # Check if training is complete
        if frames >= total_frames:
            break

    pbar.close()
    return policy_modules


@hydra.main(version_base=None, config_path="../../../conf", config_name="dummy_config")
def train_with_factories(config: DictConfig) -> dict[str, Any]:
    """Main training function using factory pattern.

    Args:
        config: Configuration object with all training parameters

    Returns:
        Trained policy modules
    """

    log_info(config)
    # Create component factory
    factory = ComponentFactory(config)

    # Create all components
    components = factory.create_components()

    # Train using the components
    policy_modules = train(components, config)

    return policy_modules


if __name__ == "__main__":
    trained_policies = train_with_factories()
    print("Training completed!")
