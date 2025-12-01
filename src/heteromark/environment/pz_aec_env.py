import torch
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector

from src.environment.smac import create_dummy_env

class SmacAecEnv(AECEnv):
    metadata = {"name": "smac_aec"}
    def __init__(
        self,
        parallel_env: dict,
        device: torch.device | None = None,
    ):
        self.device = device
        super().__init__(device=device, batch_size=[])
        # Override action spec to be based on number of operations

        self.env = parallel_env
        self.observation_type = self.env.observation_provider.observation_type

        self.NONE = 999
        self._agent_selector = AgentSelector(self.agents)
        self._agent_name_mapping = {
            agent: i for i, agent in enumerate(self.agents)
        }
        self.group_map = None #TODO Set this group map!?
    @property
    def agents(self):
        return self.env.agents
    @property
    def possible_agents(self):
        return self.env.possible_agents
    
    @property
    def obervation_space(self):
        return self.env.observation_space
    @property
    def _reward(self):
        return self.env._rewards
    
    def _init_agents(self):
        return self.env._init_agents()
    
    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode="human"):
        return self.env.render(mode=mode)
    
    def close(self):
        return self.env.close()
    
    def get_agent_smac_id(self, agent):
        return self.env.get_agent_smac_id(agent)    
    
    def _all_rewards(self, reward):
        return self.env._all_rewards(reward)
    
    def _observe_all(self):
        return self.env._observe_all()
    def _all_dones(self, step_done=False):
        return self.env._all_dones(step_done=step_done)
    
    def __del__(self):
        self.env.close()



if __name__ == "__main__":
    env = GraphMatrixEnv(
        instance=JSSPInstance.from_file("jssp_instances/ft06"),
        env_kwargs={
            "observation_provider": "lb_bipartite_gnn",
            "observation_kwargs": {
                "time_norm": 55,
                "max_edges": 102,
                "self_loop": False,
            },
        },
    )
    env.reset()
    print(env)
    import tqdm

    for _ in tqdm.tqdm(range(500), desc="Evaluating episodes"):
        env.reset()
        rollout = env.rollout(100, env.mask_rand_step, auto_reset=True)
        print(rollout)