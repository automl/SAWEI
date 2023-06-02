instance=1,2,3
SLURM=local

python awei/optimize.py '+policy=awei' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=$SLURM 'instance_set.fid=range(1,25)' +dim=8d instance_set.instance=$instance -m
python awei/optimize.py '+policy/baseline/weischedules=ei,pi,explore,ei_pi_s25_v,ei_pi_s50_v,ei_pi_s75_v,gspulse,linear_ei_to_pi,linear_pi_to_ei' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=$SLURM 'instance_set.fid=range(1,12)' +dim=8d instance_set.instance=$instance -m
python awei/optimize.py '+policy/baseline/weischedules=ei,pi,explore,ei_pi_s25_v,ei_pi_s50_v,ei_pi_s75_v,gspulse,linear_ei_to_pi,linear_pi_to_ei' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=$SLURM 'instance_set.fid=range(12,21)' +dim=8d instance_set.instance=$instance -m
python awei/optimize.py '+policy/baseline/weischedules=ei,pi,explore,ei_pi_s25_v,ei_pi_s50_v,ei_pi_s75_v,gspulse,linear_ei_to_pi,linear_pi_to_ei' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=$SLURM 'instance_set.fid=range(21,25)' +dim=8d instance_set.instance=$instance -m
python awei/optimize.py '+policy/baseline=portfolio_allocation' '+policy/baseline/pofal=n9' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=$SLURM 'instance_set.fid=range(1,25)' +dim=8d instance_set.instance=$instance -m
python awei/optimize.py '+policy/baseline=lcb' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=$SLURM 'instance_set.fid=range(1,25)' +dim=8d instance_set.instance=$instance -m
python awei/optimize.py '+policy=awei' 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=$SLURM 'instance_set.fid=range(1,25)' +dim=8d policy.auto_alpha=true instance_set.instance=$instance -m
