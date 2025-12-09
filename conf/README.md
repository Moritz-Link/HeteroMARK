# HeteroMARK Configuration System

This directory contains the Hydra-based configuration system for HeteroMARK training. The configuration is modular and allows for easy customization and experimentation.

## Configuration Structure

### Base Component Configs

Individual component configurations are stored in separate files:

- `environment.yaml` - Environment setup (dummy/real, parallel envs)
- `policy.yaml` - Policy network architecture (hidden sizes, activation)
- `loss.yaml` - Loss function parameters (PPO/HAPPO settings)
- `optimizer.yaml` - Optimizer settings (learning rate, weight decay)
- `collector.yaml` - Data collection parameters (batch size, frames)
- `replay_buffer.yaml` - Replay buffer settings (buffer size, mini-batch size)
- `training.yaml` - Training loop parameters (epochs, gradient clipping)

### Main Configs

- `dummy_config.yaml` - Base configuration combining all components
- `happo_config.yaml` - HAPPO algorithm configuration
- `high_performance_config.yaml` - High-performance training setup
- `fast_debug_config.yaml` - Fast debugging configuration

## Usage

### 1. Use Default Configuration

```python
python -m heteromark.training.factory_train
```

This uses `dummy_config.yaml` by default.

### 2. Use Alternative Configuration

```python
# Use HAPPO algorithm
python -m heteromark.training.factory_train --config-name=happo_config

# Use high-performance settings
python -m heteromark.training.factory_train --config-name=high_performance_config

# Use fast debug settings
python -m heteromark.training.factory_train --config-name=fast_debug_config
```

### 3. Override Individual Parameters

Override any parameter from the command line:

```python
# Change loss type to HAPPO
python -m heteromark.training.factory_train loss_type=happo

# Increase training epochs
python -m heteromark.training.factory_train training.num_epochs=10

# Change learning rate and device
python -m heteromark.training.factory_train optimizer.learning_rate=1e-3 device=cuda

# Multiple overrides
python -m heteromark.training.factory_train \
    loss_type=happo \
    training.num_epochs=8 \
    policy.hidden_sizes=[128,128,64] \
    collector.frames_per_batch=256
```

### 4. Combine Config and Overrides

```python
# Start with HAPPO config and override specific values
python -m heteromark.training.factory_train \
    --config-name=happo_config \
    training.num_epochs=12 \
    optimizer.learning_rate=5e-4
```

## Creating Custom Configurations

### Method 1: Create a New Config File

Create a new YAML file in the `conf/` directory:

```yaml
# conf/my_experiment.yaml
defaults:
  - dummy_config

# Override specific settings
loss_type: happo
device: cuda

training:
  total_frames: 500000
  num_epochs: 8

policy:
  hidden_sizes: [256, 256, 128]
  activation: ReLU
```

Use it with:
```bash
python -m heteromark.training.factory_train --config-name=my_experiment
```

### Method 2: Create Component Variants

Create alternative versions of component configs:

```yaml
# conf/policy_large.yaml
hidden_sizes: [256, 256, 128, 64]
activation: ReLU
device: cuda
```

Then reference it in your main config:
```yaml
# conf/my_config.yaml
defaults:
  - environment
  - policy_large  # Use the large policy variant
  - loss
  - optimizer
  - collector
  - replay_buffer
  - training
```

## Configuration Parameters Reference

### Top-Level Parameters

- `device`: Device for training (`cpu`, `cuda`)
- `env_type`: Environment type (`smac`, `custom`)
- `policy_type`: Policy architecture type (`mlp`)
- `loss_type`: Loss function (`ppo`, `happo`)
- `optimizer_type`: Optimizer (`adam`, `sgd`, `rmsprop`)
- `collector_type`: Collector type (`sync`, `multi_sync`)
- `buffer_type`: Replay buffer type (`tensor`, `memmap`)

### Component Parameters

See individual component YAML files for detailed parameters and their descriptions.

## Hydra Features

The configuration system uses [Hydra](https://hydra.cc/), which provides:

- **Composition**: Combine multiple config files
- **Override**: Change any parameter from command line
- **Multi-run**: Run multiple experiments with different configs
- **Config Groups**: Organize configs by categories
- **Type Safety**: Automatic type conversion and validation

## Examples

### Quick Debug Run
```bash
python -m heteromark.training.factory_train --config-name=fast_debug_config
```

### HAPPO Training on GPU
```bash
python -m heteromark.training.factory_train \
    --config-name=happo_config \
    device=cuda \
    environment.num_parallel_envs=8
```

### Extended Training Run
```bash
python -m heteromark.training.factory_train \
    training.total_frames=1000000 \
    training.num_epochs=10 \
    collector.frames_per_batch=512 \
    policy.hidden_sizes=[128,128,64]
```

## Tips

1. **Start Small**: Use `fast_debug_config` for initial testing
2. **Incremental Changes**: Override specific parameters rather than creating entirely new configs
3. **Document Experiments**: Create named configs for important experiments
4. **Use Defaults**: Leverage the `defaults` list to compose configs from components
5. **Command Line First**: Test parameter changes via command line before creating new config files
