from abc import ABC, abstractmethod

from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ReplayBuffer


class BaseReplayBufferFactory(ABC):
    """Abstract base class for replay buffer factories."""

    @abstractmethod
    def create(self, config: dict) -> dict:
        """Create and return replay buffers.

        Args:
            config: Configuration dictionary for buffer creation

        Returns:
            Dictionary of replay buffers by agent group
        """
        pass


class ReplayBufferFactory(BaseReplayBufferFactory):
    """Factory for creating replay buffers."""

    def __init__(self, buffer_type: str = "tensor"):
        """Initialize replay buffer factory.

        Args:
            buffer_type: Type of buffer storage (default: "tensor")
        """
        self.buffer_type = buffer_type

    def create(self, config: dict) -> dict:
        """Create replay buffers based on configuration.

        Args:
            config: Configuration dictionary containing buffer specs

        Returns:
            Dictionary of replay buffers by agent group
        """
        if self.buffer_type == "tensor":
            return self._create_tensor_buffers(config)
        elif self.buffer_type == "memmap":
            return self._create_memmap_buffers(config)
        else:
            raise ValueError(f"Unknown buffer type: {self.buffer_type}")

    def _create_tensor_buffers(self, config: dict) -> dict:
        """Create tensor-based replay buffers.

        Args:
            config: Configuration with buffer parameters

        Returns:
            Dictionary of replay buffers by agent group
        """
        buffers = {}
        batch_size = config.get("batch_size", 256)
        buffer_size = config.get("buffer_size", 10000)
        agent_groups = config.get("agent_groups", ["default"])

        for agent_group in agent_groups:
            storage = LazyTensorStorage(max_size=buffer_size)
            buffer = ReplayBuffer(
                storage=storage,
                batch_size=batch_size,
            )
            buffers[agent_group] = buffer

        return buffers

    def _create_memmap_buffers(self, config: dict) -> dict:
        """Create memory-mapped replay buffers for large datasets.

        Args:
            config: Configuration with buffer parameters

        Returns:
            Dictionary of replay buffers by agent group
        """
        buffers = {}
        batch_size = config.get("batch_size", 256)
        buffer_size = config.get("buffer_size", 10000)
        agent_groups = config.get("agent_groups", ["default"])
        scratch_dir = config.get("scratch_dir", None)

        for agent_group in agent_groups:
            storage = LazyMemmapStorage(
                max_size=buffer_size,
                scratch_dir=scratch_dir,
            )
            buffer = ReplayBuffer(
                storage=storage,
                batch_size=batch_size,
            )
            buffers[agent_group] = buffer

        return buffers


def get_dummy_replaybuffer_from_factory(loss_modules):
    rbuffer_factory = ReplayBufferFactory("tensor")
    rbuffer = rbuffer_factory.create()
    return rbuffer


if __name__ == "__main__":
    rbuffer_factory = ReplayBufferFactory("tensor")
    rbuffer = rbuffer_factory.create()
