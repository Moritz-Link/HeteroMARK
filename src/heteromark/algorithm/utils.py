import torch


def generate_mask_from_order(agent_order, ego_exclusive: bool):
    """
    Generate execution mask from agent order.

    Used during autoregressive training.

    Args:
        agent_order (list or torch.Tensor): Agent order of shape [*, n_agents].
        ego_exclusive (bool): Whether to exclude self from the mask.

    Returns:
        np.ndarray: Execution mask of shape [*, n_agents, n_agents].
    """
    # Convert list to tensor if necessary
    if isinstance(agent_order, list):
        agent_order = torch.tensor(agent_order, dtype=torch.long)

    shape = agent_order.shape
    n_agents = shape[-1]

    # Flatten batch dimensions
    agent_order = agent_order.view(-1, n_agents)
    bs = agent_order.shape[0]

    # Initialize masks
    cur_execution_mask = torch.zeros(bs, n_agents, device=agent_order.device)
    all_execution_mask = torch.zeros(bs, n_agents, n_agents, device=agent_order.device)

    batch_indices = torch.arange(bs, device=agent_order.device)

    for step in range(n_agents):
        agent_idx = agent_order[:, step].long()
        cur_execution_mask[batch_indices, agent_idx] = 1
        all_execution_mask[batch_indices, :, agent_idx] = 1 - cur_execution_mask
        all_execution_mask[batch_indices, agent_idx, agent_idx] = 1

    if not ego_exclusive:
        eye_mask = 1 - torch.eye(n_agents, device=agent_order.device)
        all_execution_mask = (
            all_execution_mask.view(*shape[:-1], n_agents, n_agents) * eye_mask
        )

    # Convert to numpy before returning
    return all_execution_mask.cpu().numpy()
