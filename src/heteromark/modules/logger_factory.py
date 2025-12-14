from abc import ABC, abstractmethod
from enum import Enum

import hydra
import torch
from omegaconf import DictConfig
from tensordict import TensorDictBase
from torch.utils.tensorboard import SummaryWriter

from heteromark.storage.extract_storage import get_storage


class BaseLoggerFactory(ABC):
    """Abstract base class for logger factories."""

    @abstractmethod
    def create(self) -> dict:
        """Create and return logger.

        Args:
            config: Configuration dictionary for logger creation

        Returns:
            one logger instance
        """
        pass


class LoggerFactory(BaseLoggerFactory):
    """Factory for creating loggers."""

    def __init__(self, config: dict):
        """Initialize logger factory.
        Args:
            buffer_type: Type of buffer storage (default: "tensor")
        """
        self.logger_types = config.logger_types
        self.config = config

    def create(self, log_dir) -> dict:
        """Create replay buffers based on configuration.

        Args:
            config: Configuration dictionary containing buffer specs

        Returns:
            Dictionary of replay buffers by agent group
        """
        logger = {}
        for logger_type in self.logger_types:
            print(f"Creating logger of type: {logger_type}")
            if logger_type == LoggerType.TRAINING:
                logger[LoggerType.TRAINING] = TrainLogger(log_dir=log_dir)

            else:
                raise ValueError(f"Unknown logger_type: {logger_type}")
        return logger


class LoggerBase(ABC):
    """Abstract base class for loggers."""

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.step_count = 0
        self.tag = ""

    def add_lr(self, value: float):
        self.writer.add_scalar(self.tag + "lr", value, self.step_count)

    def add_reward(self, values: list[float]):
        self.writer.add_scalar(self.tag + "reward", values, self.step_count)

    def add_return(self, values: list[float]):
        self.writer.add_scalar(self.tag + "return", values, self.step_count)

    def increment_step(self):
        self.step_count += 1

    @abstractmethod
    def log_steps(self, step: int, data: TensorDictBase):
        pass


class TrainLogger(LoggerBase):
    def __init__(self, log_dir: str):
        super().__init__(log_dir)
        self.logger_type = LoggerType.TRAINING
        self.tag = "train/"

    def log_done_steps(self, step: int, data: TensorDictBase):
        terminated_count = int(
            torch.sum(data[("next", "terminated")].to(torch.int32)).item()
        )
        truncated_count = int(
            torch.sum(data[("next", "truncated")].to(torch.int32)).item()
        )
        done_count = int(torch.sum(data[("next", "done")].to(torch.int32)).item())

        # Log terminated episode count
        if terminated_count > 0:
            self.writer.add_scalar(self.tag + "terminated", terminated_count, step)

        # Log truncated episode count
        if truncated_count > 0:
            self.writer.add_scalar(self.tag + "truncated", truncated_count, step)
        # Log truncated episode count
        if done_count > 0:
            self.writer.add_scalar(self.tag + "done", done_count, step)

    def log_steps(self, step: int, data: TensorDictBase):
        """Log training statistics from tensordict data.

        Args:
            step: Current training step
            data: TensorDict containing training data
        """
        self.log_done_steps(step, data)

        # Iterate through all keys to find terminated, truncated, and reward entries

        # Log reward statistics if we have terminated episodes with rewards
        terminated_mask = data[("next", "terminated")].to(torch.bool)
        terminated_rewards = data[("next", "reward")][terminated_mask].detach()
        if terminated_rewards.numel() > 0:
            self.writer.add_scalar(
                self.tag + "reward/min", terminated_rewards.min().item(), step
            )
            self.writer.add_scalar(
                self.tag + "reward/max", terminated_rewards.max().item(), step
            )
            self.writer.add_scalar(
                self.tag + "reward/mean",
                terminated_rewards.mean().item(),
                step,
            )

        # Increment step counter
        self.step_count += step


class LoggerType(str, Enum):
    """Enum for logger types."""

    TRAINING = "training"
    EVALUATION = "evaluation"


@hydra.main(version_base=None, config_path="../../../conf", config_name="dummy_config")
def test(config: DictConfig):
    logger_f = LoggerFactory(config=config.components.logger)
    logger = logger_f.create(log_dir=config.log_dir)
    # env = env_factory._apply_transforms(env)
    print(logger.keys())
    print(" === Logger created :", logger, "===")
    return logger


if __name__ == "__main__":
    logger = test()
    from heteromark.storage.extract_storage import get_storage

    storage = get_storage("after_collection_w_advantage")
    print(storage)
