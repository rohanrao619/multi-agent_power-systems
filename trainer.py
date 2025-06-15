import sys
sys.path.append(".")  # Adjust the path to import "local" benchmarl

from benchmarl.algorithms import MaddpgConfig
from benchmarl.environments import EnergyTradingTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    # Loads from "benchmarl/conf/task/energy_trading/simple_p2p.yaml"
    task = EnergyTradingTask.SIMPLE_P2P.get_from_yaml()

    # Basic configs, modify as needed
    algorithm_config = MaddpgConfig.get_from_yaml()
    experiment_config = ExperimentConfig.get_from_yaml()
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    # No wandb for now
    experiment_config.loggers = ['csv']
    experiment_config.save_folder = 'results'

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=42,
        config=experiment_config)
    
    experiment.run()