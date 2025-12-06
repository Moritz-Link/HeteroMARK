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
    print(f"\nGeneral Settings:")
    print(f"  Device: {config.get('device', 'cpu')}")
    print(f"  Environment Type: {config.get('env_type', 'smac')}")
    print(f"  Policy Type: {config.get('policy_type', 'mlp')}")
    print(f"  Loss Type: {config.get('loss_type', 'ppo')}")
    print(f"  Optimizer Type: {config.get('optimizer_type', 'adam')}")
    print(f"  Collector Type: {config.get('collector_type', 'sync')}")
    print(f"  Buffer Type: {config.get('buffer_type', 'tensor')}")

    # Environment settings
    if hasattr(config, 'environment'):
        print(f"\nEnvironment Settings:")
        for key, value in config.environment.items():
            print(f"  {key}: {value}")

    # Policy settings
    if hasattr(config, 'policy'):
        print(f"\nPolicy Settings:")
        for key, value in config.policy.items():
            print(f"  {key}: {value}")

    # Loss settings
    if hasattr(config, 'loss'):
        print(f"\nLoss Settings:")
        for key, value in config.loss.items():
            print(f"  {key}: {value}")

    # Optimizer settings
    if hasattr(config, 'optimizer'):
        print(f"\nOptimizer Settings:")
        for key, value in config.optimizer.items():
            print(f"  {key}: {value}")

    # Collector settings
    if hasattr(config, 'collector'):
        print(f"\nCollector Settings:")
        for key, value in config.collector.items():
            print(f"  {key}: {value}")

    # Replay buffer settings
    if hasattr(config, 'replay_buffer'):
        print(f"\nReplay Buffer Settings:")
        for key, value in config.replay_buffer.items():
            print(f"  {key}: {value}")

    # Training settings
    if hasattr(config, 'training'):
        print(f"\nTraining Settings:")
        for key, value in config.training.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("INITIALIZING TRAINING COMPONENTS...")
    print("=" * 80 + "\n")
