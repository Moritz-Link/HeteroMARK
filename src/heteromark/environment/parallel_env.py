import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from torchrl.data.tensor_specs import Categorical
from torchrl.envs.libs.gym import (
    convert_box_spec,
    convert_dict_spec,
    register_gym_spec_conversion,
)

from heteromark.environment.smac import create_dummy_env


# Register conversion for gym.spaces.Discrete
@register_gym_spec_conversion(spaces.Discrete)
def _discrete_converter(space, **kwargs):
    return Categorical(n=space.n, shape=())


@register_gym_spec_conversion(spaces.Dict)
def _dict_converter(space, **kwargs):
    return convert_dict_spec(space, **kwargs)


@register_gym_spec_conversion(spaces.Box)
def _box_converter(space, **kwargs):
    return convert_box_spec(space, **kwargs)


class smac_parallel_env(ParallelEnv):
    def __init__(self, env, max_cycles):
        self.max_cycles = max_cycles
        self.env = env
        self.env.reset()
        self.reset_flag = 0
        self.agents, self.action_spaces = self._init_agents()
        self.possible_agents = self.agents[:]
        self.alive_agents = self.agents[:]
        self.dead_agents = []
        self.device = "cpu"

        observation_size = env.get_obs_size()
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(observation_size,),
                        dtype="float32",
                    ),
                    "action_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.action_spaces[name].n,),
                        dtype=np.int8,
                    ),
                }
            )
            for name in self.agents
        }
        self._reward = 0

    def _init_agents(self):
        last_type = ""
        agents = []
        action_spaces = {}
        self.agents_id = {}
        i = 0
        for agent_id, agent_info in self.env.agents.items():
            unit_action_space = spaces.Discrete(
                self.env.get_total_actions() - 1
            )  # no-op in dead units is not an action
            if agent_info.unit_type == self.env.marine_id:
                agent_type = "marine"
            elif agent_info.unit_type == self.env.marauder_id:
                agent_type = "marauder"
            elif agent_info.unit_type == self.env.medivac_id:
                agent_type = "medivac"
            elif agent_info.unit_type == self.env.hydralisk_id:
                agent_type = "hydralisk"
            elif agent_info.unit_type == self.env.zergling_id:
                agent_type = "zergling"
            elif agent_info.unit_type == self.env.baneling_id:
                agent_type = "baneling"
            elif agent_info.unit_type == self.env.stalker_id:
                agent_type = "stalker"
            elif agent_info.unit_type == self.env.colossus_id:
                agent_type = "colossus"
            elif agent_info.unit_type == self.env.zealot_id:
                agent_type = "zealot"
            else:
                raise AssertionError(f"agent type {agent_type} not supported")

            if agent_type == last_type:
                i += 1
            else:
                i = 0

            agents.append(f"{agent_type}_{i}")
            self.agents_id[agents[-1]] = agent_id
            action_spaces[agents[-1]] = unit_action_space
            last_type = agent_type

        return agents, action_spaces

    def seed(self, seed=None):
        if seed is None:
            self.env._seed = 42
        else:
            self.env._seed = seed
        self.env.full_restart()

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()

    @property
    def death_tracker_ally(self):
        return self.env.env.death_tracker_ally

    def reset(self, seed: int | None = None):
        self.env._episode_count = 1
        self.env.reset()

        self.agents = self.possible_agents[:]
        self.frames = 0
        self.all_dones = {agent: False for agent in self.possible_agents}
        all_infos = {agent: {} for agent in self.agents}
        self.alive_agents = self.agents[:]
        self.dead_agents = []
        # observation_dict, info_dict
        return self._observe_all(), all_infos

    def get_agent_smac_id(self, agent):
        return self.agents_id[agent]

    def _all_rewards(self, reward):
        all_rewards = [reward] * len(self.agents)
        return {agent: reward for agent, reward in zip(self.agents, all_rewards)}

    def _observe_all(self):
        all_obs = []
        for agent in self.agents:
            agent_id = self.get_agent_smac_id(agent)
            obs = self.env.get_obs_agent(agent_id)
            action_mask = self.env.get_avail_agent_actions(agent_id)
            action_mask = action_mask[1:]
            action_mask = np.array(action_mask).astype(np.int8)
            obs = np.asarray(obs, dtype=np.float32)
            all_obs.append({"observation": obs, "action_mask": action_mask})
        return {agent: obs for agent, obs in zip(self.agents, all_obs)}

    def _all_dones(self, step_done=False):
        dones = [True] * len(self.agents)
        if not step_done:
            for i, agent in enumerate(self.agents):
                agent_done = False
                agent_id = self.get_agent_smac_id(agent)
                agent_info = self.env.get_unit_by_id(agent_id)
                if agent_info.health == 0:
                    agent_done = True
                dones[i] = agent_done
        return {agent: bool(done) for agent, done in zip(self.agents, dones)}

    def step(self, all_actions):
        action_list = [0] * self.env.n_agents  # 0 == no_op
        for agent in self.agents:
            agent_id = self.get_agent_smac_id(agent)
            if agent in all_actions:
                if all_actions[agent] is None or agent in self.dead_agents:
                    action_list[agent_id] = 0
                else:
                    action_list[agent_id] = all_actions[agent] + 1
        self._reward, terminated, smac_info = self.env.step(action_list)
        self.frames += 1
        done = terminated or self.frames >= self.max_cycles

        all_infos = {agent: {} for agent in self.agents}
        # all_infos.update(smac_info)

        # TODO: Endet immer, sobald einer der Agenten truncated?
        all_terminations = {agent: terminated for agent in self.agents}
        all_truncation = self._all_dones(done)
        all_rewards = self._all_rewards(self._reward)
        all_observes = self._observe_all()
        # Update alive and dead agents
        self.alive_agents = [
            agent for agent in self.agents if not all_truncation[agent]
        ]
        self.dead_agents = [agent for agent in self.agents if all_truncation[agent]]

        # Update action_mask for dead_agents
        for agent in self.dead_agents:
            all_observes[agent]["action_mask"] = np.ones_like(
                all_observes[agent]["action_mask"]
            )

        # observation_dict, rewards_dict, terminations_dict, truncations_dict, info_dict
        return all_observes, all_rewards, all_terminations, all_truncation, all_infos

    def __del__(self):
        self.env.close()


def create_dummy_parallel_pz_env():
    basic_env = create_dummy_env()
    specs = {}
    specs["max_cycles"] = 500
    # env = parallel_env(500, specs["max_cycles"])
    env = smac_parallel_env(basic_env, specs["max_cycles"])
    print("=== Created Dummy ENv ===")
    return env


if __name__ == "__main__":
    env = create_dummy_parallel_pz_env()
    obs = env.reset()
    print("Initial observation:", obs)
