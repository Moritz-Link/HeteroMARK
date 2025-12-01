from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import replace

from smacv2.env import StarCraft2Env
import numpy as np
from absl import logging
import time

from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper







def create_env(specs):

    
    env = StarCraftCapabilityEnvWrapper(
        capability_config=specs["distributed_config"],
        map_name=specs["map_name"],
        debug=True,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
    )

    return env



def create_dummy_env():
    distribution_config = {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    }
    specs = {
        "distributed_config": distribution_config,
        "map_name": "10gen_terran",
        
    }

    return create_env(specs)



if __name__ == "__main__":
    env = create_dummy_parallel_pz_env()
    obs = env.reset()
    print("Initial observation:", obs)


