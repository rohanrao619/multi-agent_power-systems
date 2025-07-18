import numpy as np

from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import ParallelEnv
from .energy_trading import EnergyTradingEnv

from torch import Tensor
from tensordict import TensorDict
from benchmarl.experiment import Experiment

# Level 2: Environment for Contract Proposal
## Assume frozen downstream environment, already trained on the entire contract space

class ContractProposalEnv(ParallelEnv):

    def __init__(self, config, render_mode=None):

        # Zoo requirement
        self.render_mode = render_mode

        # Forced
        config['use_contracts'] = True

        # Base environment
        self.trading_env = EnergyTradingEnv(config, render_mode=render_mode)

        # Base experminent data, get policy
        self.base_exp = Experiment.reload_from_file(config['base_exp_path'])

        # Setup agents
        self.possible_agents = self.trading_env.possible_agents

        # Episode length, week to start with
        self.eps_len = config.get('contract_eps_len', 7)

    def observation_space(self, agent):
        
        # Same for all
        # Observation Space: [soc, FiT] + [ToU, load] x ToU periods
        return spaces.Box(low=0, high=1024, shape=(2+2*len(self.trading_env.ToU),), dtype=np.float32)
    
    def action_space(self, agent):
        
        # Same for all
        # Action Space: [commit qnt and price] X ToU periods
        return spaces.Box(low=np.array([0, 0, 0, -1, -1, -1]), high=np.ones(shape=(6,)), shape=(2*len(self.trading_env.ToU),), dtype=np.float32)

    # Zoo requirement
    def render(self):
        pass

    def reset(self, seed=None, options=None):

        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.timestep = 0
        self.day = np.random.randint(0, self.trading_env.n_days-self.eps_len) # Need at least eps_len days to complete an episode

        self.agents = self.possible_agents.copy()

        self.state_vars = {
            aid: {
                "soc": self.trading_env._gaussian_init(self.trading_env.es_capacity[0] + (self.trading_env.es_capacity[1] - self.trading_env.es_capacity[0]) / 2,
                                           self.trading_env.es_capacity[1] - self.trading_env.es_capacity[0],
                                           self.trading_env.es_capacity[0], self.trading_env.es_capacity[1]),
                "grid_reliance": 0.0
            } for aid in self.agents
        }

        obs = {aid: self._get_obs(aid) for aid in self.agents}
        infos = {aid: {"grid_reliance": 0.0} for aid in self.agents}

        return obs, infos
    
    def _get_obs(self, aid):

        soc = self.state_vars[aid]["soc"]

        # Should be forecasted? How do we know beforehand?
        load = self._get_load(aid)

        obs = [soc, self.trading_env.FiT] + self.trading_env.ToU + load

        return np.array(obs, dtype=np.float32)
    
    def _get_load(self, aid):

        aid = self.trading_env.aid_mapping[aid]

        idx = self.trading_env.eps_len * self.day + self.timestep*self.trading_env.eps_len
        is_prosumer = self.trading_env.data[aid]["prosumer"]      
        
        demand = np.array(self.trading_env.data[aid]["demand"][idx:idx + self.trading_env.eps_len])
        pv = np.array(self.trading_env.data[aid]["pv"][idx:idx + self.trading_env.eps_len])

        # Convert timemap to numpy array for element-wise comparison
        timemap = np.array(self.trading_env.timemap)
        
        tou_demands = [self._aggregate_over_ToU(demand[timemap == i]) for i in range(len(self.trading_env.ToU))]
        tou_pvs = [self._aggregate_over_ToU(pv[timemap == i]) for i in range(len(self.trading_env.ToU))]
        
        load = [a-b for a, b in zip(tou_demands, tou_pvs)] if is_prosumer else tou_demands

        return load
    
    def _aggregate_over_ToU(self, data):

        return sum(data)/len(data)
    
    # Apply base policy to trade, assuming single group
    def _trade_forward(self, obs):

        stacked_obs = Tensor([obs[aid] for aid in self.base_exp.group_map['agents']])
        obs_tdict = TensorDict(agents=TensorDict(observation=stacked_obs))

        actions = self.base_exp.policy.forward(obs_tdict)['agents']['action'].cpu().numpy()
        action_dict = {aid: actions[i] for i, aid in enumerate(self.base_exp.group_map['agents'])}
        
        return action_dict
    
    def step(self, action_dict):

        # Decode actions
        contract_bids = [{aid: (action_dict[aid][i], action_dict[aid][i+3]) for aid in self.agents} for i in range(len(self.trading_env.ToU))]

        # Reset base environment
        obs, infos = self.trading_env.reset(options={"contract_bids": contract_bids,
                                                     "day": self.day + self.timestep,
                                                     "state_vars": self.state_vars})
        
        # Init rewards
        tot_rewards = {aid: 0.0 for aid in self.agents}
        
        # Step base environment for an episode
        for _ in range(self.trading_env.eps_len):

            # Base environment actor
            actions = self._trade_forward(obs)
            
            obs, rewards, terminations, truncations, infos = self.trading_env.step(actions)
            for aid in self.agents:
                tot_rewards[aid] += rewards[aid]

        # Update SOC
        for aid in self.agents:
            self.state_vars[aid]["soc"] = self.trading_env.state_vars[aid]["soc"]
            self.state_vars[aid]["grid_reliance"] = self.trading_env.state_vars[aid]["grid_reliance"]
        
        # Time Controls
        self.timestep += 1
        done = (self.timestep >= self.eps_len)

        # Finishing Touches
        obs = {aid: self._get_obs(aid) if not done else np.zeros((8,), dtype=np.float32) for aid in self.agents} # Gibberish at the end, does not matter
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: done for aid in self.agents}
        infos = {aid: {"grid_reliance": self.state_vars[aid]["grid_reliance"]} for aid in self.agents}
        
        return obs, tot_rewards, terminations, truncations, infos