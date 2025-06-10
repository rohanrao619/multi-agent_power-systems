import numpy as np
from pprint import pprint

from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig

from env.energy_trading import EnergyTradingEnv

def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"

# Shared policy (can be extended to individual policies)
policies = {
    "shared_policy": (
        None,  # default PPO policy
        spaces.Box(low=0, high=1024, shape=(4,), dtype=np.float32),
        spaces.Box(low=np.array([0, -1]), high=np.array([1, 1]), shape=(2,), dtype=np.float32),
        {})}

config = {"data_path": "data/ausgrid/ausgrid.json"}

# Define PPO config with multi-agent setup
ppo_config = (
    PPOConfig()
    .environment(env=EnergyTradingEnv, env_config=config)
    .env_runners(num_env_runners=1)
    .framework("torch")
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    .training(train_batch_size=24*32))

# Build the algorithm.
algo = ppo_config.build_algo()

# After the pile of warnings
print("\n####### Starting Training #######")

# Train it for 10 iterations ...
for i in range(10):
    results = algo.train()
    print(f"\nIteration {i + 1}:\nMean Episode Reward = {results['env_runners']['episode_return_mean']:.2f}$, Total Timesteps = {results['env_runners']['num_env_steps_sampled_lifetime']}")
    # print("Agent Returns: ", end="")
    # print(", ".join(f"{agent}: {returns:.2f}$" for agent, returns in results['env_runners']['agent_episode_returns_mean'].items()))