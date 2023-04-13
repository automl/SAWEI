# Towards Self-Adjusting Weighted Expected Improvement
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

### Installation
```bash
make install  # installs dacbo
cd AWEI
pip install -e .  # installs SAWEI
```

### Run
```bash
# SAWEI
python awei/optimize.py '+policy=awei' 'policy.atol_rel=0.05,0.1,0.5,1' 'seed=range(1,21)' 'instance_set.fid=range(1,25)' '+dim=2d' '+slurm=local' -m

# Baselines
python awei/optimize.py '+policy/baseline/weischedules=glob(*)' 'seed=range(1,21)' 'instance_set.fid=range(1,25)' '+dim=2d' '+slurm=local' -m
```