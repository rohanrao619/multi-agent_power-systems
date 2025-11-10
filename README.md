## Exploring Cooperation and Competition in P2P Energy Markets through Contract-Augmented Multi-Agent RL

This is the codebase for my Master's Thesis at TUM. Read the full version [here](Saini_Rohan_Rao_Masterarbeit.pdf).

- `benchmarl`: MARL Library, cloned from [BenchMARL](https://github.com/facebookresearch/BenchMARL/commit/567acd9) to add custom environments and algorithms.
- `data`: Ausgrid Dataset for P2P Energy Market Simulations. Taken from [here](https://github.com/pierre-haessig/ausgrid-solar-data).
- `env`: Trading and Contracting Environments using [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/).
- `thesis_results`: Notebooks to reproduce the Results and Discussions from the Thesis.
- [evaluation.ipynb](evaluation.ipynb): Crude Evaluation for Sanity Check.
- [trainer.py](trainer.py): Control Center for running Experiments.
- [requirements.txt](requirements.txt): Required Packages + Python 3.13.3.