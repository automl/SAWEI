# Towards Self-Adjusting Weighted Expected Improvement for Bayesian Optimization
This is the repository for the submission.

With Self-Adjusting Weighted Expected Improvement (SAWEI) we aim to self-adjust the exploration-exploitation trade-off in Bayesian Optimization (BO).
For this, we adjust the weight α of Weighted Expected Improvement [Sobester et al., 2005].
There are two crucial puzzle pieces:
1. When to adjust α?
2. How to adjust α?

## When To Adjust α
We adjust α whenever the Upper Bound Regret (UBR) [Makarova et al., 2022] converges.
Makarova et al. use the UBR as a stopping criterion whereas we use it as a signal about the current state of BO's progress.
UBR convergence means: We adjust α whenver the gradient of the UBR is 0. In the implementation we actually check the moving IQM of the UBR to smooth the signal and because we are interested in the general trend.

## How to Adjust α
We adjust α in the *opposite* direction of the current search attitude.
That means if the attitude currently is exploring, we in/decrease α such that WEI is more exploitative and vice versa.
We determine the attitude as follows:
If the EI term of WEI is bigger than the PI term, the attitude is exploring.
If the PI term is bigger, it is exploiting.
Now based on the attitude we add or subtract a constant to/from α.

## Experiments
We compared our method SAWEI with baselines from the literature and handcrafted ones on the 24 BBOB functions of the COCO benchmark [Hansen et al., 2020].

You can find all runcommands for the BBOB benchmarks in `AWEI/README.md`.

:warning: You might need to specify your own slurm or local setup via `+slurm=yoursetupconfig` as a command line override.

### Plotting
All plots for the paper are generated in `AWEI/awei/evaluate.ipynb`.

## Installation
> :warning: Only tested on Linux.

First, create a fresh conda environment and activate it.
```bash
conda env create -n sawei python=3.10
conda activate sawei
```

Then, clone this repo and checkout the correct branch.
```bash
git clone https://github.com/automl/SAWEI.git
cd SAWEI
```

Clone SMAC3 in SAWEI, we need a special branch. Install SMAC.
```bash
git clone https://github.com/automl/SMAC3.git
cd SMAC3
git checkout sawei
pip install -e . --upgrade
```

Go back to the SAWEI folder.
```bash
cd ..
# Install requirements
pip install pre-commit
pip install -r requirements.txt --upgrade
# Install SAWEI
make install-dev

# Install AWEI
cd AWEI
pip install -e . --upgrade
```

## Repository Structure
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
In optimization, we often encounter expensive black-box problems with unknown problem structures.
Bayesian Optimization (BO) is a popular, surrogate-assisted and thus sample-efficient approach for this setting.
The BO pipeline itself is highly configurable with many different design choices regarding the initial design, surrogate model and acquisition function (AF). Unfortunately, our understanding of how to select suitable components for a problem at hand is very limited. 
In this work, we focus on the choice of the AF, whose main purpose it is to balance the trade-off between exploring regions with high uncertainty and those with high promise for good solutions. 
We propose Self-Adjusting Weighted Expected Improvement (SAWEI), where we let the exploration-exploitation trade-off self-adjust in a data-driven manner based on a convergence criterion for BO. 
On the BBOB functions of the COCO benchmark, our method performs favorably compared to handcrafted baselines and serves as a robust default choice for any problem structure.
With SAWEI, we are a step closer to on-the-fly, data-driven and robust BO designs that automatically adjust their sampling behavior to the problem at hand. 

## Cite Us
TBD Poster @ GECCO'23

