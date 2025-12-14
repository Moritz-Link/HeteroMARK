from heteromark.algorithm.happo_algorithm import HappoAlgorithm
from heteromark.algorithm.ppo_algorithm import PpoAlgorithm

ALGORITHMS = {
    "happo": HappoAlgorithm,
    "ppo": PpoAlgorithm,
}


def algorithm_factory(name, **kwargs):
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}")

    else:
        return ALGORITHMS[name](**kwargs)
