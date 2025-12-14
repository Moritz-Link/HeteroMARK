"""
Factory-based training script using TorchRL and modular components.

This script demonstrates a clean, modular approach to training multi-agent
reinforcement learning systems using the factory design pattern.
"""

from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from heteromark.algorithm.happo_algorithm import HappoAlgorithm
from heteromark.loss import update_critic
from heteromark.modules import (
    CollectorFactory,
    EnvironmentFactory,
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
        self.loss_factory = LossFactory(loss_type=config.components.loss.loss_type)
        self.optimizer_factory = OptimizerFactory(
            optimizer_type=config.components.optimizer.optimizer_type
        )
        self.collector_factory = CollectorFactory(
            collector_type=config.components.collector.collector_type
        )
        self.buffer_factory = ReplayBufferFactory(
            buffer_type=config.components.replay_buffer.buffer_type
        )

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
                self.config.components.loss,
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

        # Initialize HAPPO algorithm if using HAPPO loss
        if self.config.components.loss.loss_type == "happo":
            components["happo_algorithm"] = HappoAlgorithm(
                agent_groups=components["env"].group_map,
                policy_modules=components["policy_modules"],
                sample_log_prob_key="log_prob",
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


from heteromark.training.utils import (
    filter_tensordict_by_agent,
    set_factor_to_all,
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
    happo_algorithm = components["happo_algorithm"]
    device = components["device"]
    advantage_module = components["advantage_modules"]

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

        tensordict_data[("critic", "observation")] = tensordict_data[
            (random_agent_group, "observation", "observation")
        ][:, 0]
        tensordict_data[("next", "critic", "observation")] = tensordict_data[
            ("next", random_agent_group, "observation", "observation")
        ][:, 0]

        # HAPPO-specific: Reset factor for new batch
        if happo_algorithm:
            happo_algorithm.reset_factor(tensordict_data)
            # TODO: set all factor values at the beginning
            set_factor_to_all(
                tensordict_data, happo_algorithm.get_factor(tensordict_data)
            )
            # set_factor_of_all_agents(
            #     tensordict_data, env, happo_algorithm.get_factor(tensordict_data)
            # )
            agent_order = happo_algorithm.get_agent_order()
        else:
            agent_order = env.agents
        print("Agent order:", agent_order)
        # Training epochs

        for epoch_idx in range(num_epochs):
            # Compute advantages
            with torch.no_grad():
                advantage_module(tensordict_data)
            if happo_algorithm:
                happo_algorithm.set_adv_as_factor(tensordict_data)

            #     # Add HAPPO factor if using HAPPO
            # if happo_algorithm:
            #     factor = happo_algorithm.get_factor_for_agent(agent)
            #     filtered_td.set((agent, "factor"), factor)
            # Iterate over agent groups

            for agent_group in agent_order:
                # agent_group = get_agent_group(agent)
                # Process batch for this agent group
                # if happo_algorithm:
                #     current_factor = happo_algorithm.get_factor_for_agent()
                #     tensordict_data[agent_group]["factor"][
                #         :, get_agent_index(agent) : get_agent_index(agent) + 1
                #     ] = current_factor

                group_buffer = replay_buffers[agent_group]
                group_loss_module = loss_modules[agent_group]
                group_optimizer = optimizers[agent_group]

                for agent in env.group_map[agent_group]:
                    filtered_td = filter_tensordict_by_agent(
                        tensordict_data, agent_name=agent, agent_group=agent_group
                    )
                    filtered_td[(agent_group, "advantage")] = (
                        filtered_td["advantage"].clone().unsqueeze(-1)
                    )
                    filtered_td[(agent_group, "factor")] = (
                        filtered_td["factor"].clone().unsqueeze(-1)
                    )
                    # group_batch = process_batch(filtered_td, agent_group, device)

                    group_buffer.empty()

                    # Add to replay buffer
                    group_buffer.extend(filtered_td.to(device))

                    # Training on mini-batches
                    for batch in group_buffer:
                        batch = batch.to(device)

                        # Compute loss
                        loss_vals = group_loss_module(batch)
                        total_loss = (
                            loss_vals["loss_objective"]
                            # + loss_vals["loss_critic"]
                            + loss_vals["loss_entropy"]
                        )

                        # Backward pass
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            group_loss_module.parameters(), max_grad_norm
                        )

                        # Optimizer step
                        group_optimizer.step()
                        group_optimizer.zero_grad()
                        update_critic(
                            optimizers, loss_modules, batch, frames, None
                        )  # TODO: Logger

                # Update HAPPO factor after training this agent group
                # For each Agent
                if happo_algorithm:
                    happo_algorithm.update_factor(tensordict_data, agent_group)

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
