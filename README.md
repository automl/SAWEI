# Self-Adjusting Weighted Expected Improvement for Bayesian Optimization
This is the repository for the paper accepted at AutoML Conf'23.

🎆 We also have a lean and convenient SMAC3 integration.
For this clone the SMAC3 repo, checkout the branch and install it (assuming you already activated your favorite virtual env):
```bash
git clone https://github.com/automl/SMAC3.git && cd SMAC3
git checkout feature/sawei
pip install -e .
```
You can find a usage example [here](https://github.com/automl/SMAC3/blob/feature/sawei/examples/6_advanced_features/2_SAWEI.py).

For the experimental code from this repository, see below.


## Installation
> :warning: Only tested on Linux.


First, create a fresh conda environment and activate it.
```bash
conda create -n SAWEI python=3.10.0 -y
conda activate SAWEI
```

Then, download the repository from [here](https://anon-github.automl.cc/r/SAWEI-9CD6/), unpack it and change into it.
```bash
cd SAWEI
```

Clone SMAC3 in SAWEI-9CD6, we need a special branch. Install SMAC.
```bash
git clone https://github.com/automl/SMAC3.git
cd SMAC3
git checkout sawei
pip install -e . --upgrade
```

Go back to the SAWEI-9CD6 folder.
```bash
cd ..
# Install requirements
pip install pre-commit
pip install -r requirements.txt --upgrade
# Install DAC-BO
make install-dev

# Install AWEI
cd AWEI
pip install -e . --upgrade
```

## Repository and Code Structure
With our method SAWEI we dynamically configure BO. For BO we use SMAC as an implementation.
In order to dynamically interact with SMAC, we use the formulation of a MDP (Markov Decision Process).
We translate our method in the following way to the MDP:
- Action space: Setting $\alpha$ of Weighted Expected Improvement (WEI) to control the exploration-exploitation trade-off.
- State space: For now, the state consists of remaining budget. Ignored.
- Reward: Log regret of the incumbent. Ignored.

Therefore we have to main components in our code:
- dacbo: Defining the dynamic interface to BO/SMAC
- AWEI: Our method.

```
└───AWEI
    |   └───runscripts
    |   └───awei: code for heuristics, baselines and our method SAWEI (`adaptive_weighted_ei.py`)
    |   |       └───configs: settings for all methods and experiments
    |   |       └───`optimize.py` runscript for evaluating methods
    |   |       └───`evaluate.ipynb`: Plotting script
    |   |       └───...
└───dacbo: defines the dynamic interface (gym) to SMAC in `new_env.py`
```



## Abstract
Bayesian Optimization (BO) is a class of surrogate-based, sample-efficient algorithms for optimizing black-box problems with small evaluation budgets.
The BO pipeline itself is highly configurable with many different design choices regarding the initial design, surrogate model, and acquisition function (AF). Unfortunately, our understanding of how to select suitable components for a problem at hand is very limited. 
In this work, we focus on the definition of the AF, whose main purpose is to balance the trade-off between exploring regions with high uncertainty and those with high promise for good solutions. 
We propose Self-Adjusting Weighted Expected Improvement (SAWEI), where we let the exploration-exploitation trade-off self-adjust in a data-driven manner, based on a convergence criterion for BO. 
On the BBOB functions of the COCO benchmarking platform, our method exhibits a favorable any-time performance compared to handcrafted baselines and serves as a robust default choice for any problem structure.
The suitability of our method also transfers to HPOBench.
With SAWEI, we are a step closer to on-the-fly, data-driven, and robust BO designs that automatically adjust their sampling behavior to the problem at hand. 

## Example
Our scripts use [hydra](https://hydra.cc/) making configuring experiments covenient and requires a special commandline syntax. 
Here we show how to optimize the 1st instance of function 15 (Rastrigin) of the BBOB benchmark for 8 dimensions with our method SAWEI.
```bash
cd AWEI
python awei/optimize.py '+policy=awei' 'seed=89' +instance_set/BBOB=default 'instance_set.fid=15' 'instance_set.instance=1' 'n_eval_episodes=1' +dim=2d
```
If you want to configure the dimensions, checkout options in `AWEI/awei/configs/dim`.
You can find the logs in `AWEI/awei/awei_runs/DATE/TIME`.
The logs have following content:
- `initial_design.json`: The configurations for the initial design/design of experiments (DoE).
- `rollout_data.json`: The rollout data of the optimization ("step", "state", "action", "reward", "instance", "cost", "configuration", "episode", "policy_name", "seed")
- `wei_history.json`: The summands of the Weighted Expected Improvement (WEI) ("n_evaluated", "alpha", "pi_term", "ei_term", "pi_pure_term", "pi_mod_term"). "pi_term" corresponds to the exploitation term, "ei_term" to the exploration term of WEI.
- `ubr_history.json`: The upper bound regret (UBR) with its summands ("n_evaluated", "ubr", "min_ucb", "min_lcb")

## Experiments
You can find all runcommands for the BBOB benchmarks in `AWEI/run_BBOB.sh`. All runcommands for the ablation of SAWEI on BBOB are in `AWEI/run_BBOB_ablation.sh`.
The runcommands for HPOBench are in `AWEI/run_HPOBench.sh`.

:warning: You might need to specify your own slurm or local setup via `+slurm=yoursetupconfig` as a command line override. We set a local launcher per default so no slurm cluster is required. 

Please find the data in this [google drive](https://drive.google.com/drive/folders/12jmpJ1VRS3rzRcCd6rrrcjusP19RAmmV?usp=sharing).

### Plotting
All plots for the paper are generated in `AWEI/awei/evaluate.ipynb`.


## Cite Us
```bibtex
@inproceedings{benjamins-automl23a
    author    = {Carolin Benjamins and
                Elena Raponi and
                Anja Jankovic and
                Carola Doerr and
                Marius Lindauer},
    title     = {Self-Adjusting Weighted Expected Improvement for Bayesian Optimization},
    year      = {2023}
}
```
