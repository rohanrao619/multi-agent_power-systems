import json
import numpy as np

from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import ParallelEnv

# Basic environment, inspired from https://doi.org/10.24963/ijcai.2021/401
## Environment config items: dt, eps_len, es_P, es_capacity, es_efficiency, ToU, FiT, data_path

class EnergyTradingEnv(ParallelEnv):
    
    def __init__(self, config, render_mode=None):

        # Zoo requirement
        self.render_mode = render_mode

        # Decision every hour, episode runs over a day
        self.eps_len = config.get("eps_len", 24)
        self.dt = config.get("dt", 1)  # Time step in hours

        # PV and Demand Data
        self.data = json.load(open(config.get("data_path"), "r"))

        # Setup Agents, use the agent mapping later to access data
        self._setup_agents()

        # 1 year of data
        self.n_days = int(len(self.data[self.aid_mapping[self.possible_agents[0]]]["pv"])/24)

        # Battery (ES) Config
        self.es_P = config.get("es_P", 2)  # Power rating of the battery
        self.es_capacity = config.get("es_capacity", [2, 10]) # [min soc, max soc]
        self.es_efficiency = config.get("es_efficiency", [0.95,0.95])  # [charge efficiency, discharge efficiency]

        # Grid Config, Time of Use and FiT Prices
        self.ToU = config.get("ToU", [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, # 12 AM - 8 AM
                                      0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, # 9 AM - 4 PM
                                      0.18, 0.18, 0.18, 0.18, # 5 PM - 8 PM
                                      0.08, 0.08, 0.08]) # 9 PM - 11 PM
        
        self.FiT = config.get("FiT", 0.04)  # Feed-in Tariff
        

    def _setup_agents(self):

        prosumer_idx = 1
        consumer_idx = 1

        # For better agent names
        self.aid_mapping = {}

        for aid in self.data.keys():

            if self.data[aid]["prosumer"]:
                self.aid_mapping[f"prosumer_{prosumer_idx}"] = aid
                prosumer_idx += 1
            else:
                self.aid_mapping[f"consumer_{consumer_idx}"] = aid
                consumer_idx += 1
        
        self.possible_agents = sorted(self.aid_mapping.keys())

    
    def observation_space(self, agent):
        
        # Same for all
        # Observation Space: [load, soc, ToU, FiT]
        return spaces.Box(low=0, high=1024, shape=(4,), dtype=np.float32)
    
    def action_space(self, agent):
        
        # Same for all
        # Action Space: [price and soc control]
        return spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), shape=(2,), dtype=np.float32)
    

    # Zoo requirement
    def render(self):
        pass


    def reset(self, seed=None, options=None):

        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        
        self.timestep = 0
        self.day = np.random.randint(0, self.n_days)

        self.agents = self.possible_agents.copy()

        self.state = {
            aid: {
                "soc": self._gaussian_init(self.es_capacity[0] + (self.es_capacity[1] - self.es_capacity[0]) / 2,
                                           self.es_capacity[1] - self.es_capacity[0],
                                           self.es_capacity[0], self.es_capacity[1]),
            } for aid in self.agents
        }
        
        obs = {aid: self._get_obs(aid) for aid in self.agents}
        infos = {aid: {} for aid in self.agents}

        return obs, infos

    
    def step(self, action_dict):
        
        # P2P quotes
        quotes = {}

        # Update State
        for aid, action in action_dict.items():

            price, soc_control = action
            price = self.FiT + price * (self.ToU[self.timestep] - self.FiT)  # Scale Price

            soc = self.state[aid]["soc"]
            load = self._get_load(aid)

            if soc_control >= 0:
                
                # Charging
                charge_P = min(self.es_P * soc_control,
                               (self.es_capacity[1] - soc)/self.es_efficiency[0]*self.dt)
                
                self.state[aid]["soc"] = soc + charge_P * self.dt * self.es_efficiency[0]
                qnt = load + charge_P * self.dt

            else:

                # Discharging
                discharge_P = min(self.es_P * soc_control,
                                  (self.es_capacity[0] - soc)*self.es_efficiency[1]/self.dt)
                
                self.state[aid]["soc"] = soc + (discharge_P * self.dt)/self.es_efficiency[1]
                qnt = load + discharge_P * self.dt

            # Prepare trade action
            quotes[aid] = (qnt, price)
        
        # Run the double auction
        matches, trades, open_book = self._run_double_auction(quotes)

        # Costs/Earnings from successful trades
        rewards = {aid: -(trades[aid]["price"] * trades[aid]["quantity"]) for aid in self.agents}

        # Settle unmet quotes with ToU and FiT
        for buyer in open_book["buyers"]:
            aid, price, qnt = buyer
            rewards[aid] -= qnt * self.ToU[self.timestep]

        for seller in open_book["sellers"]:
            aid, price, qnt = seller
            rewards[aid] += qnt * self.FiT
        
        # Time Controls
        self.timestep += 1
        done = (self.timestep >= self.eps_len)

        # Finishing Touches
        obs = {aid: self._get_obs(aid) if not done else np.zeros((4,), dtype=np.float32) for aid in self.agents} # Gibberish at the end, does not matter
        terminations = {aid: False for aid in self.agents}
        truncations = {aid: done for aid in self.agents}
        infos = {aid: {} for aid in self.agents}
        
        return obs, rewards, terminations, truncations, infos
    
    
    def _get_obs(self, aid):

        soc = self.state[aid]["soc"]

        # Should be forecasted? How do we know beforehand?
        load = self._get_load(aid)

        return np.array([load,
                         soc,
                         self.ToU[self.timestep],
                         self.FiT],
                         dtype=np.float32)
    
    def _get_load(self, aid):

        aid = self.aid_mapping[aid]

        idx = self.eps_len * self.day + self.timestep
        is_prosumer = self.data[aid]["prosumer"]      
        
        demand = self.data[aid]["demand"][idx]
        pv = self.data[aid]["pv"][idx]
        load = demand - pv if is_prosumer else demand

        return load
    
    
    def _gaussian_init(self, mean, bucket, v_min, v_max):

        # Gaussian (99.7%) on the bucket [mean-bucket/2, mean + bucket/2], clipped appropriately
        return np.clip(np.random.normal(mean, bucket/6), max(v_min, mean - bucket/2), min(v_max, mean + bucket/2))
    
    
    def _run_double_auction(self, quotes):
       
        buyers = []
        sellers = []

        # Parse actions
        for aid, action in quotes.items():
            
            quantity, price = action
            if quantity > 0:
                buyers.append((aid, price, quantity))  # buy quantity
            elif quantity < 0:
                sellers.append((aid, price, -quantity))  # sell quantity (as positive)
            # quantity == 0 → no trade, ignored

        # Sort by willingness: buyers (high price first), sellers (low price first)
        buyers.sort(key=lambda x: -x[1])
        sellers.sort(key=lambda x: x[1])

        matches = []
        trades = {aid: {"quantity": 0, "price": 0} for aid in self.agents}

        buyer_idx = 0
        seller_idx = 0

        while buyer_idx < len(buyers) and seller_idx < len(sellers):
            buyer_id, bid_price, bid_qty = buyers[buyer_idx]
            seller_id, ask_price, ask_qty = sellers[seller_idx]

            if bid_price >= ask_price:
                trade_qty = min(bid_qty, ask_qty)
                clearing_price = (bid_price + ask_price) / 2.0

                matches.append({
                    "buyer": buyer_id,
                    "seller": seller_id,
                    "price": clearing_price,
                    "amount": trade_qty,
                })

                trades[buyer_id]["price"] = (trades[buyer_id]["price"] * trades[buyer_id]["quantity"] + clearing_price * trade_qty) / (trades[buyer_id]["quantity"] + trade_qty)
                trades[buyer_id]["quantity"] += trade_qty

                trades[seller_id]["price"] = (trades[seller_id]["price"] * abs(trades[seller_id]["quantity"]) + clearing_price * trade_qty) / (abs(trades[seller_id]["quantity"]) + trade_qty)
                trades[seller_id]["quantity"] -= trade_qty

                buyers[buyer_idx] = (buyer_id, bid_price, bid_qty - trade_qty)
                sellers[seller_idx] = (seller_id, ask_price, ask_qty - trade_qty)

                if buyers[buyer_idx][2] == 0:
                    buyer_idx += 1
                if sellers[seller_idx][2] == 0:
                    seller_idx += 1
            else:
                break  # No more compatible matches
        
        open_book = {
            "buyers": buyers[buyer_idx:] if buyer_idx < len(buyers) else [],
            "sellers": sellers[seller_idx:] if seller_idx < len(sellers) else []}

        return matches, trades, open_book
