"""Utility script to validate and print HeteroMARK configurations.

This script loads and displays configurations without running training,
useful for debugging and understanding the config system.

Usage:
    # Print default config
    python conf/validate_config.py

    # Print specific config
    python conf/validate_config.py --config-name=happo_config

    # Print with overrides
    python conf/validate_config.py loss_type=happo training.num_epochs=8
"""

from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf


def validate_config(config_name: str = "dummy_config", overrides: list[str] = None):
    """Load and validate a configuration.

    Args:
        config_name: Name of the config file (without .yaml extension)
        overrides: List of config overrides (e.g., ["loss_type=happo"])

    Returns:
        Loaded OmegaConf configuration
    """
    if overrides is None:
        overrides = []

    # Initialize Hydra with current directory (relative path)
    with initialize(version_base=None, config_path="."):
        config = compose(config_name=config_name, overrides=overrides)

    return config


def print_config(config, title: str = "Configuration"):
    """Pretty-print a configuration.

    Args:
        config: OmegaConf configuration to print
        title: Title for the output
    """
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    print(OmegaConf.to_yaml(config))
    print("=" * 70 + "\n")


def validate_required_fields(config) -> bool:
    """Validate that required configuration fields are present.

    Args:
        config: Configuration to validate

    Returns:
        True if all required fields are present, False otherwise
    """
    required_top_level = [
        "device",
        "env_type",
        "policy_type",
        "loss_type",
        "optimizer_type",
        "collector_type",
        "buffer_type",
    ]

    required_components = [
        "environment",
        "policy",
        "loss",
        "optimizer",
        "collector",
        "replay_buffer",
        "training",
    ]

    errors = []

    # Check top-level fields
    for field in required_top_level:
        if field not in config:
            errors.append(f"Missing top-level field: {field}")

    # Check component configs
    for component in required_components:
        if component not in config:
            errors.append(f"Missing component config: {component}")

    if errors:
        print("\n[ERROR] Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("\n[OK] Configuration validation passed!")
    return True


if __name__ == "__main__":
    import sys

    # Simple command-line parsing
    args = sys.argv[1:]
    config_name = "dummy_config"
    overrides = []

    # Check for --config-name flag
    if args and args[0].startswith("--config-name="):
        config_name = args[0].split("=", 1)[1]
        args = args[1:]

    # Remaining args are overrides
    overrides = args

    # Load and validate config
    try:
        config = validate_config(config_name, overrides)
        print_config(config, f"Loaded Config: {config_name}")
        validate_required_fields(config)

        # Print some useful information
        print(f"\nConfig Summary:")
        print(f"  Algorithm: {config.loss_type.upper()}")
        print(f"  Device: {config.device}")
        print(f"  Total Frames: {config.training.total_frames:,}")
        print(f"  Epochs per Batch: {config.training.num_epochs}")
        print(f"  Frames per Batch: {config.collector.frames_per_batch}")
        print(f"  Mini-batch Size: {config.replay_buffer.batch_size}")
        print(
            f"  Policy Architecture: {config.policy.hidden_sizes} ({config.policy.activation})"
        )
        print(f"  Learning Rate: {config.optimizer.learning_rate}\n")

    except Exception as e:
        print(f"\n[ERROR] Error loading configuration: {e}")
        sys.exit(1)
