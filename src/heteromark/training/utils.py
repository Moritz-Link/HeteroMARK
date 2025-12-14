import torch
from tensordict import TensorDict


def get_agent_group(agent_name: str) -> str:
    """Extract the agent group from an agent name.

    Args:
        agent_name: Agent name in format 'group_index' (e.g., 'marauder_0', 'marine_1')

    Returns:
        Agent group name (e.g., 'marauder', 'marine')

    Example:
        >>> get_agent_group('marauder_0')
        'marauder'
        >>> get_agent_group('marine_3')
        'marine'
    """
    # Split by last underscore to handle group names with underscores
    parts = agent_name.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid agent name format: {agent_name}. Expected 'group_index'"
        )
    return parts[0]


def get_agent_index(agent_name: str) -> int:
    """Extract the agent index within its group from an agent name.

    Args:
        agent_name: Agent name in format 'group_index' (e.g., 'marauder_0', 'marine_1')

    Returns:
        Agent index within the group

    Example:
        >>> get_agent_index('marauder_0')
        0
        >>> get_agent_index('marine_3')
        3
    """
    # Split by last underscore to handle group names with underscores
    parts = agent_name.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid agent name format: {agent_name}. Expected 'group_index'"
        )

    try:
        return int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid agent index in name: {agent_name}")


def filter_tensordict_by_agent(
    tensordict: TensorDict, agent_name: str, agent_group: str
) -> TensorDict:
    """Filter tensordict to contain only data for a specific agent.

    This function:
    1. Removes all agent groups except the specified one
    2. Filters the remaining agent group to only include the specific agent
    3. Removes all samples where the truncated flag is True for that agent

    Args:
        tensordict: Input tensordict with structure [batch_size, num_agents, ...]
        agent_name: Agent name (e.g., 'marauder_0')
        agent_group: Agent group name (e.g., 'marauder')

    Returns:
        Filtered tensordict containing only non-truncated data for the specified agent

    Example:
        >>> filtered_td = filter_tensordict_by_agent(batch_data, 'marauder_0', 'marauder')
    """
    # Get agent index within the group
    agent_idx = get_agent_index(agent_name)

    # Create a new tensordict with only the specified agent group
    # Start by copying top-level keys that aren't agent groups
    filtered_dict = {}

    # Get all agent groups in the tensordict (keys that are TensorDicts)
    agent_groups = [
        key
        for key in tensordict.keys()
        if isinstance(tensordict[key], TensorDict)
        and key != "next"
        and key != "collector"
        and key != "critic"
    ]

    # Copy non-agent-group top-level keys
    for key in tensordict.keys():
        if key not in agent_groups:
            filtered_dict[key] = tensordict[key]

    # Filter the specified agent group
    if agent_group not in tensordict.keys():
        raise ValueError(
            f"Agent group '{agent_group}' not found in tensordict. Available groups: {agent_groups}"
        )

    agent_group_data = tensordict[agent_group]

    # Check if agent_idx is valid
    if agent_idx >= agent_group_data.batch_size[1]:
        raise ValueError(
            f"Agent index {agent_idx} out of range for group '{agent_group}' with {agent_group_data.batch_size[1]} agents"
        )

    # Index to select only the specific agent (keep agent dimension as 1)
    # Use slice indexing to keep the dimension: [batch_size, n_agents, ...] -> [batch_size, 1, ...]
    agent_specific_data = agent_group_data[:, agent_idx : agent_idx + 1]

    # Get the truncated flag for this agent [batch_size, 1, 1]
    truncated = agent_specific_data["truncated"].squeeze(-1).squeeze(-1)  # [batch_size]

    # Create mask for non-truncated samples
    non_truncated_mask = ~truncated

    # Filter all data by non-truncated mask (keeps [filtered_batch_size, 1, ...] shape)
    filtered_agent_data = agent_specific_data[non_truncated_mask]

    # Add the filtered agent group data back
    filtered_dict[agent_group] = filtered_agent_data

    # Handle 'next' tensordict if it exists
    if "next" in tensordict.keys():
        next_dict = {}

        # Filter the agent group in 'next'
        if agent_group in tensordict["next"].keys():
            # Keep agent dimension as 1: [batch_size, 1, ...]
            next_agent_data = tensordict["next"][agent_group][
                :, agent_idx : agent_idx + 1
            ]
            next_dict[agent_group] = next_agent_data[non_truncated_mask]

        # Copy non-agent-group keys from 'next'
        for key in tensordict["next"].keys():
            if key not in agent_groups:
                next_dict[key] = tensordict["next"][key][non_truncated_mask]

        filtered_dict["next"] = TensorDict(
            next_dict, batch_size=torch.Size([non_truncated_mask.sum().item()])
        )

    # Handle 'collector' tensordict if it exists
    if "collector" in tensordict.keys():
        collector_dict = {}
        for key in tensordict["collector"].keys():
            collector_dict[key] = tensordict["collector"][key][non_truncated_mask]
        filtered_dict["collector"] = TensorDict(
            collector_dict, batch_size=torch.Size([non_truncated_mask.sum().item()])
        )

    # Filter other top-level tensors by the mask
    for key in [
        "done",
        "step_count",
        "terminated",
        "truncated",
        "advantage",
        "reward",
        "state_value",
        "value_target",
        "critic",
        "factor",
    ]:
        if key in filtered_dict:
            filtered_dict[key] = filtered_dict[key][non_truncated_mask]

    # Create the final filtered tensordict
    filtered_tensordict = TensorDict(
        filtered_dict, batch_size=torch.Size([non_truncated_mask.sum().item()])
    )

    return filtered_tensordict


def set_factor_of_all_agents(tensordict: TensorDict, env, factor: torch.Tensor):
    """Set the factor value for all agents in the tensordict."""
    for agent_group, agents in env.group_map.items():
        num_agents_in_group = len(agents)
        tensordict[agent_group]["factor"] = factor.repeat(1, num_agents_in_group, 1)


def set_factor_to_all(tensordict: TensorDict, factor: torch.Tensor):
    tensordict["factor"] = factor


def log_info(config):
    """Log configuration information and training start message.

    Args:
        config: OmegaConf DictConfig containing all training parameters
    """
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    print("\nConfiguration:")
    print("-" * 80)

    # General settings
    print("\nGeneral Settings:")
    print(f"  Device: {config.components.collector.device}")
    print(f"  Environment Type: {config.env.env_type}")
    print(f"  Policy Type: {config.components.policy.policy_type}")
    print(f"  Loss Type: {config.components.loss.loss_type}")
    print(f"  Optimizer Type: {config.components.optimizer.optimizer_type}")
    print(f"  Collector Type: {config.components.collector.collector_type}")
    print(f"  Buffer Type: {config.components.replay_buffer.buffer_type}")

    # Environment settings
    if hasattr(config, "env"):
        print("\nEnvironment Settings:")
        for key, value in config.env.items():
            print(f"  {key}: {value}")

    # Policy settings
    if hasattr(config.components, "policy"):
        print("\nPolicy Settings:")
        for key, value in config.components.policy.items():
            print(f"  {key}: {value}")

    # Loss settings
    if hasattr(config.components, "loss"):
        print("\nLoss Settings:")
        for key, value in config.components.loss.items():
            print(f"  {key}: {value}")

    # Optimizer settings
    if hasattr(config.components, "optimizer"):
        print("\nOptimizer Settings:")
        for key, value in config.components.optimizer.items():
            print(f"  {key}: {value}")

    # Collector settings
    if hasattr(config.components, "collector"):
        print("\nCollector Settings:")
        for key, value in config.components.collector.items():
            print(f"  {key}: {value}")

    # Replay buffer settings
    if hasattr(config.components, "replay_buffer"):
        print("\nReplay Buffer Settings:")
        for key, value in config.components.replay_buffer.items():
            print(f"  {key}: {value}")

    # Training settings
    if hasattr(config, "training"):
        print("\nTraining Settings:")
        for key, value in config.training.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("INITIALIZING TRAINING COMPONENTS...")
    print("=" * 80 + "\n")
