# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HeteroMARK is a PyTorch-based framework for training heterogeneous multi-agent reinforcement learning (MARL) systems. It focuses on coordinating agents that differ in observation spaces, action spaces, capabilities, and network architectures. The framework implements HAPPO (Heterogeneous-Agent Proximal Policy Optimization) and related algorithms.

## Development Commands

### Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package in editable mode
pip install -e .
```

### Running Training
```powershell
# Run SMAC demo (main entry point)
python .\main.py

# Run factory-based training (example in file)
python -m heteromark.training.factory_train
```

### Testing
```powershell
# Run basic environment test
python .\test.py

# Run pytest (if tests exist)
pytest

# Run pytest with coverage
pytest --cov=heteromark
```

### Code Quality
```powershell
# Run ruff linter (configured in pyproject.toml)
ruff check .

# Auto-fix linting issues
ruff check --fix .
```

## Architecture

### Core Design Pattern: Factory Pattern

The codebase is built on a hierarchical factory pattern with multiple layers:

1. **Individual Factories** (in `src/heteromark/modules/`):
   - `EnvironmentFactory` - Creates SMAC or PettingZoo environments
   - `PolicyFactory` - Creates actor-critic network modules per agent group
   - `LossFactory` - Creates HAPPO/PPO loss modules per agent group
   - `OptimizerFactory` - Creates optimizers per agent group
   - `CollectorFactory` - Creates data collectors
   - `ReplayBufferFactory` - Creates replay buffers per agent group

2. **ComponentFactory** (in `src/heteromark/training/factory_train.py`):
   - Orchestrates all individual factories
   - Takes a single `OmegaConf` config
   - Returns dictionary of all training components

### Agent Group Organization

All components organize agents into **groups** (e.g., "marine", "medivac"). Components return dictionaries keyed by agent group name:

```python
policy_modules = {
    "marine": TensorDictModule(...),
    "medivac": TensorDictModule(...),
}
```

Each group can have:
- Separate observation/action spaces
- Different policy networks
- Individual loss modules and optimizers
- Distinct training dynamics

### Nested Key Convention

Multi-agent data uses nested tuple keys in TensorDict structures:

```python
(agent_group, "observation", "observation")  # observations
(agent_group, "action")                      # actions
(agent_group, "state_value")                 # state values
(agent_group, "factor")                      # HAPPO factor (algorithm-specific)
```

This enables handling heterogeneous agents without special-case code.

### HAPPO Algorithm

The HAPPO algorithm (`src/heteromark/algorithm/happo_algorithm.py`) implements factor-based gradient scaling for heterogeneous agents:

**Key Concepts**:
- **Factor**: A tensor that scales policy gradients to account for updates made by previously-trained agents in the same batch
- **Update Cycle**: After each agent group is trained, the factor is updated based on importance weights (ratio of new to old log-probabilities)
- **Agent Order**: Agents are trained in randomized or fixed order per epoch

**Training Flow**:
```python
for batch_data in collector:
    happo_algorithm.reset_factor(batch_data)  # Initialize factor to ones
    agent_order = happo_algorithm.get_agent_order()

    for agent_group in agent_order:
        # Get current factor for this agent
        factor = happo_algorithm.get_factor_for_agent(agent_group)
        batch_data.set((agent_group, "factor"), factor)

        # Train agent with factor-scaled loss
        loss = loss_module(batch_data)  # Loss automatically uses factor
        loss.backward()
        optimizer.step()

        # Update factor for next agent
        happo_algorithm.update_factor(batch_data, agent_group)
```

### Training Loop Structure

The main training loop (`src/heteromark/training/factory_train.py`):

1. **Component Creation**: Create all components via `ComponentFactory`
2. **Data Collection**: Collect batches from environment via collector
3. **Multi-Epoch Training**: For each batch, run multiple training epochs
4. **Per-Agent Updates**: For each agent group in order:
   - Compute advantages using GAE
   - Add HAPPO factor (if using HAPPO)
   - Add data to agent's replay buffer
   - Train on mini-batches with gradient clipping
   - Update HAPPO factor (if using HAPPO)

### Configuration System

Uses **Hydra/OmegaConf** for structured configuration. Key config sections:

```python
config = {
    # Algorithm/component selection
    "env_type": "smac",           # "smac", "custom"
    "policy_type": "mlp",         # "mlp" (extensible)
    "loss_type": "happo",         # "happo", "ppo"
    "optimizer_type": "adam",     # "adam", "sgd", "rmsprop"
    "collector_type": "sync",     # "sync", "multi_sync"
    "buffer_type": "tensor",      # "tensor", "memmap"

    # Component-specific configs
    "environment": {...},
    "policy": {...},
    "loss": {...},
    "optimizer": {...},
    "collector": {...},
    "replay_buffer": {...},
    "training": {...},
}
```

### Environment Adapters

Three environment types in `src/heteromark/environment/`:

1. **SMAC** (`smac.py`):
   - Wraps StarCraft II multi-agent environment
   - Supports heterogeneous unit types (marines, marauders, medivacs)
   - Can create dummy environments for testing

2. **PettingZoo Parallel** (`parallel_env.py`):
   - Wraps PettingZoo parallel environments
   - Groups agents by type
   - Handles discrete action spaces

3. **PettingZoo AEC** (`pz_aec_env.py`):
   - Alternate execution framework support

All environments are wrapped with TorchRL's `PettingZooWrapper` and `TransformedEnv` for integration.

## Key Dependencies

- **TorchRL** (0.10.0): Core RL infrastructure (collectors, loss functions)
- **TensorDict** (0.10.0): Structured tensor data management
- **PettingZoo** (1.25.0): Multi-agent environment framework
- **SMAC v2**: StarCraft II multi-agent environment (installed from git)
- **Hydra** (1.3.2): Configuration management
- **PyTorch**: Core tensor/model framework

## Extensibility Points

To add new components:

1. **New Algorithm**:
   - Create class in `src/heteromark/algorithm/`
   - Add to `algorithm_factory.py`

2. **New Policy Architecture**:
   - Extend `PolicyFactory._create_*_policy()` methods in `modules/policy_factory.py`

3. **New Loss Function**:
   - Implement in `src/heteromark/loss/`
   - Add to `LossFactory` in `modules/loss_factory.py`

4. **New Environment**:
   - Create adapter in `src/heteromark/environment/`
   - Add to `EnvironmentFactory` in `modules/environment_factory.py`

All follow the factory pattern for seamless integration.

## File Structure

```
src/heteromark/
├── algorithm/              # Training algorithms
│   ├── happo_algorithm.py  # HAPPO factor management
│   └── algorithm_factory.py
├── environment/            # Environment adapters
│   ├── smac.py            # SMAC wrapper
│   ├── parallel_env.py    # PettingZoo parallel
│   └── pz_aec_env.py      # PettingZoo AEC
├── loss/                  # Loss functions
│   └── happo_loss.py      # HAPPO loss with factor support
├── modules/               # Factory implementations
│   ├── environment_factory.py
│   ├── policy_factory.py
│   ├── loss_factory.py
│   ├── optimizer_factory.py
│   ├── collector_factory.py
│   └── replay_buffer_factory.py
└── training/              # Training loops
    ├── factory_train.py   # Main factory-based training
    ├── main_train.py      # (WIP - incomplete)
    └── utils.py           # Utilities (e.g., log_info)
```

## Important Implementation Details

### Loss Module Configuration

Loss modules use nested keys for multi-agent data. When creating loss modules, keys are set per agent group:

```python
loss_module.set_keys(
    advantage=(agent_group, "advantage"),
    value_target=(agent_group, "value_target"),
    value=(agent_group, "state_value"),
    sample_log_prob=(agent_group, "action_log_prob"),
    action=(agent_group, "action"),
    reward=(agent_group, "reward"),
    done=(agent_group, "done"),
    terminated=(agent_group, "terminated"),
    factor=(agent_group, "factor"),  # HAPPO-specific
)
```

### Policy Module Structure

Policy modules consist of:
- **Actor**: `TensorDictModule` wrapping the policy network
- **Value Module**: `ValueOperator` wrapping the critic network

Both are created per agent group with potentially different architectures.

### Data Flow

```
Collector → TensorDict → Replay Buffer (per agent) → Mini-batches → Loss Module → Optimizer
                ↓
         HAPPO Algorithm (factor updates between agent updates)
```

## Code Style

Ruff is configured with the following linters:
- E: pycodestyle errors
- F: pyflakes
- I: isort (import sorting)
- B: flake8-bugbear
- UP: pyupgrade
- G: flake8-logging-format
