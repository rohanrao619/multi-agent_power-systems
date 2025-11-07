import sys
sys.path.append(".")  # Adjust the path to import "local" benchmarl

import torch

from benchmarl.algorithms import MaddpgConfig, MappoConfig, IppoConfig
from benchmarl.environments import EnergyTradingTask, ContractProposalTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    # # Loads from "benchmarl/conf/task/energy_trading/simple_p2p.yaml"
    # task = EnergyTradingTask.SIMPLE_P2P.get_from_yaml()

    # Loads from "benchmarl/conf/task/contract_proposal/hourly_commits.yaml"
    task = ContractProposalTask.HOURLY_COMMITS.get_from_yaml()
    task.config["base_exp_path"] = "results/mix/stage_1/mappo/seed_7/checkpoints/checkpoint_385024.pt"

    # Modify as needed
    algorithm_config = MappoConfig.get_from_yaml()
    experiment_config = ExperimentConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # Experiment Config
    experiment_config.share_policy_params = False
    experiment_config.max_n_frames = 65536
    experiment_config.gamma = 0.99
    experiment_config.lr = 2.5e-4
    experiment_config.clip_grad_val = 5 # Unexplored, Untuned

    # Actor Config
    model_config.num_cells = [64, 32]
    model_config.activation_class = torch.nn.ReLU

    # Critic Config
    critic_model_config.num_cells = [256, 128]
    critic_model_config.activation_class = torch.nn.ReLU
    algorithm_config.share_param_critic = False
    
    # Algorithm Config, Off-Policy, MADDPG
    # algorithm_config.use_double_auction_critic = False # Only implemented for MADDPG, comment out for MAPPO
    # algorithm_config.exploration_type = 'gaussian' # Only implemented for MADDPG, comment out for MAPPO
    experiment_config.off_policy_collected_frames_per_batch = 8192
    experiment_config.off_policy_n_optimizer_steps = 128
    experiment_config.off_policy_train_batch_size = 256
    experiment_config.off_policy_memory_size = 131072
    experiment_config.exploration_eps_init = 0.8
    experiment_config.exploration_eps_end = 0.1
    experiment_config.exploration_anneal_frames = 262144

    # Algorithm Config, On-Policy, MAPPO
    experiment_config.on_policy_collected_frames_per_batch = 4096
    experiment_config.on_policy_n_minibatch_iters = 2
    experiment_config.on_policy_minibatch_size = 256
    algorithm_config.clip_epsilon = 0.2
    algorithm_config.entropy_coef = 0.02
    algorithm_config.critic_coef = 1.0
    algorithm_config.lmbda = 0.96
    
    # Logging Config
    experiment_config.loggers = ['csv'] # No WandB for now
    experiment_config.save_folder = 'results/mix/stage_2/mappo'
    experiment_config.checkpoint_interval = 4096
    experiment_config.checkpoint_at_end = True
    experiment_config.keep_checkpoints_num = None # Keep all checkpoints
    experiment_config.log_individual_agents = True
    
    # Evaluation Config
    experiment_config.evaluation = True
    experiment_config.evaluation_static = True
    experiment_config.evaluation_deterministic_actions = True
    experiment_config.evaluation_interval = 4096
    experiment_config.evaluation_episodes = 256

    # Task Config
    task.config["data_path"] = "data/ausgrid/group_4.json"
    task.config["dt"] = 1
    task.config["eps_len"] = 24
    task.config["es_P"] = 2
    task.config["es_capacity"] = [0, 8]
    task.config["es_efficiency"] = [0.95, 0.95]
    task.config["ToU"] = [0.15, 0.22, 0.44]
    task.config["FiT"] = 0.04
    task.config["use_double_auction"] = True
    task.config["use_pooling"] = False
    task.config["use_grid_only"] = False
    task.config["use_contracts"] = True
    task.config["use_absolute_contracts"] = False
    task.config["max_contract_qnt"] = 1.0

    # Hardware Config
    experiment_config.sampling_device = "cpu"
    experiment_config.buffer_device = "cpu"
    experiment_config.train_device = "cuda"

    # Crippled due to PettingZoo wrapper seeding issues
    experiment_config.parallel_collection = False
    experiment_config.on_policy_n_envs_per_worker = 1
    experiment_config.off_policy_n_envs_per_worker = 1

    experiment = Experiment(task=task,
                            algorithm_config=algorithm_config,
                            model_config=model_config,
                            critic_model_config=critic_model_config,
                            seed=15,
                            config=experiment_config)
    
    experiment.run()