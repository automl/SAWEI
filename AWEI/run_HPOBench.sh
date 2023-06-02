MODEL=rf,xgb,svm,lr,nn
TASK_ID='53,3917,9952,10101,146818'
INST=default_det
SLURM=local

# AWEI
python awei/optimize.py '+policy=awei' 'policy.auto_alpha=false,true' 'seed=range(1,11)' +instance_set/HPOBench=$INST 'n_eval_episodes=1' instance_set.model=$MODEL '+slurm=$SLURM' instance_set.task_id=$TASK_ID -m
# Baseline WEI schedules
python awei/optimize.py '+policy/baseline/weischedules=ei,pi,explore' 'seed=range(1,11)' +instance_set/HPOBench=$INST 'n_eval_episodes=1' instance_set.model=$MODEL '+slurm=$SLURM' instance_set.task_id=$TASK_ID -m
python awei/optimize.py '+policy/baseline/weischedules=ei_pi_s25_v,ei_pi_s50_v,ei_pi_s75_v' 'seed=range(1,11)' +instance_set/HPOBench=$INST 'n_eval_episodes=1' instance_set.model=$MODEL '+slurm=$SLURM' instance_set.task_id=$TASK_ID -m
python awei/optimize.py '+policy/baseline/weischedules=gspulse,linear_ei_to_pi,linear_pi_to_ei' 'seed=range(1,11)' +instance_set/HPOBench=$INST 'n_eval_episodes=1' instance_set.model=$MODEL '+slurm=$SLURM' instance_set.task_id=$TASK_ID -m
# LCB
python awei/optimize.py '+policy/baseline=lcb' 'seed=range(1,11)' +instance_set/HPOBench=$INST 'n_eval_episodes=1' instance_set.model=$MODEL +slurm=$SLURM instance_set.task_id=$TASK_ID -m
# Pofal
python awei/optimize.py '+policy/baseline=portfolio_allocation' '+policy/baseline/pofal=n9' 'seed=range(1,11)' +instance_set/HPOBench=$INST 'n_eval_episodes=1' instance_set.model=$MODEL '+slurm=$SLURM' instance_set.task_id=$TASK_ID -m