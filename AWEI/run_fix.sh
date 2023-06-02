MODEL=rf,xgb,svm,lr,nn
TASK_ID='53,3917,9952,10101,146818'
INST=default_det

python awei/optimize.py '+policy/baseline/weischedules=ei_pi_s25_v,ei_pi_s50_v,ei_pi_s75_v' 'seed=range(1,11)' +instance_set/HPOBench=$INST 'n_eval_episodes=1' instance_set.model=$MODEL '+slurm=noctua' instance_set.task_id=$TASK_ID -m


for inst in {1..3}
do
    echo $inst
    python awei/optimize.py '+policy/baseline/weischedules=ei_pi_s25_v,ei_pi_s50_v,ei_pi_s75_v' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=noctua 'instance_set.fid=range(1,25)' +dim=8d instance_set.instance=$inst -m
done

# instance=1,2,3
# python awei/optimize.py '+policy/baseline/weischedules=ei_pi_s25_v' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=noctua 'instance_set.fid=range(1,25)' +dim=8d instance_set.instance=$instance -m
# python awei/optimize.py '+policy/baseline/weischedules=ei_pi_s50_v' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=noctua 'instance_set.fid=range(1,25)' +dim=8d instance_set.instance=$instance -m
# python awei/optimize.py '+policy/baseline/weischedules=ei_pi_s75_v' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=noctua 'instance_set.fid=range(1,25)' +dim=8d instance_set.instance=$instance -m