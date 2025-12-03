from heteromark.algorithm.happo_algorithm import HappoAlgorithm

ALGORITHMS = {
    "happo": HappoAlgorithm,
}


def algorithm_factory(name, **kwargs):
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {name}")

    else:
        return ALGORITHMS[name](**kwargs)
