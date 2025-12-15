import torch


def update_critic(
    replay_buffers,
    optimizer,
    loss_module,
    tensordict_data,
    step,
    device,
    logger=None,
):
    critic_buffer = replay_buffers["critic"]
    critic_optimizer = optimizer["critic"]
    critic_loss_module = loss_module["critic"]

    critic_buffer.empty()
    critic_buffer.extend(tensordict_data.to(device))
    for batch in critic_buffer:
        batch = batch.to(device)
        critic_optimizer.zero_grad()
        # Compute loss
        critic_loss = critic_loss_module(batch)
        critic_loss.backward()
        critic_optimizer.step()
        if logger is not None:
            logger.log_critic_metrics(
                critic_loss=critic_loss.item(),
                step=step,
            )


def update_actor(
    batch, optimizer, loss_module, device, max_grad_norm, step, logger=None
):
    """Update actor network with gradient clipping.

    Args:
        batch: Mini-batch of training data
        optimizer: Optimizer for the actor network
        loss_module: Loss module for computing actor loss
        device: Device to move batch to
        max_grad_norm: Maximum gradient norm for clipping
        step: Current training step
        logger: Optional logger for metrics
    """

    # Move batch to device
    batch = batch.to(device)

    # Zero gradients
    optimizer.zero_grad()

    # Compute loss
    loss_vals = loss_module(batch)
    total_loss = (
        loss_vals["loss_objective"]
        # + loss_vals["loss_critic"]
        + loss_vals["loss_entropy"]
    )

    # Backward pass
    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)

    # Optimizer step
    optimizer.step()

    # Log metrics
    if logger is not None:
        logger.log_actor_metrics(
            loss_objective=loss_vals["loss_objective"].item(),
            loss_entropy=loss_vals["loss_entropy"].item(),
            total_loss=total_loss.item(),
            step=step,
        )
