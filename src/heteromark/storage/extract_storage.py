from tensordict import TensorDict

STORAGE_NAMES = {
    "after_collection_w_advantage": "after_collection_w_advantage.td",
}


def get_storage(name):
    if name not in STORAGE_NAMES:
        raise ValueError(f"Storage name {name} not recognized.")
    return TensorDict.load(STORAGE_NAMES[name])


# td.save("buffer.td")
