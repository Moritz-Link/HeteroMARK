def train_mappo_inside(
    cfg: DictConfig,
    logger: TrainingLogger,
    checkpoint_logger: ModelCheckpointLogger,
    dispatcher,
    evaluator: BaseEvaluator | None = None,
):
    # Load Instance
    instance = dispatcher.setup_instance()

    # Create environment and models
    env = dispatcher.setup_environment()
    policy_module, value_module = dispatcher.setup_models()

    # Create other Components
    replay_buffer = dispatcher.setup_replay_buffer()
    loss_module, advantage_module = dispatcher.setup_loss_modules()
    optimizer = dispatcher.setup_optimizers()
    collector = dispatcher.setup_collector()
    scheduler = dispatcher.setup_schedulers()

    pbar = tqdm(total=cfg.training.total_frames)
    num_updates = 0
    frames = 0
    rollout_frames = 0

    for i, tensordict_data in enumerate(collector):
        if not isinstance(collector, AECCollector):
            tensordict_data = transform_td(tensordict_data, env)

        epoch_frames = frames
        number_frames = tensordict_data.numel()
        rollout_frames += number_frames
        epoch_frames_step_size = number_frames // cfg.training.num_epochs

        # ===============================================
        # Trennung bei MARL
        # ===============================================

        for epoch_idx in range(cfg.training.num_epochs):
            epoch_reset_frames = frames
            for agent_group, agents_in_group in env.group_map.items():
                group_batch = tensordict_data.to(device)
                group_batch = process_batch(group_batch, agent_group)
                group_batch = group_batch.reshape(-1)
                group_buffer = replay_buffer[agent_group]

                group_loss_module = loss_module[agent_group]
                group_optimizer = optimizer[agent_group]

                with torch.no_grad():
                    group_loss_module.value_estimator(
                        group_batch,
                        params=group_loss_module.critic_network_params,
                        target_params=group_loss_module.target_critic_network_params,
                    )

                group_buffer.extend(group_batch.to(device))
                agent_reset_frames = epoch_reset_frames

                for idx, batch in enumerate(group_buffer):
                    if idx == 0:
                        frames = agent_reset_frames
                    batch = batch.to(device)
                    loss_vals = group_loss_module(batch)
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Backward pass and optimization
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(
                        group_loss_module.parameters(), cfg.training.max_grad_norm
                    )

                    group_optimizer.step()
                    group_optimizer.zero_grad()

                    frames += batch.numel()

   
                    num_updates += 1
            epoch_frames += epoch_frames_step_size

        pbar.update(tensordict_data.numel())

    pbar.close()


    return policy_module
