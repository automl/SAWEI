INSTANCE='1,2,3'
DELTA=0.05,0.1,0.25,auto
TRACK_ATTITUDE=last,until_inc_change,until_last_switch
ATOL_REL='0.05,0.1,0.5,1'
FID='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24'


for fid in {8..24}
    do
    for inst in {1..1}
        do
            echo $fid $inst
            python awei/optimize.py '+policy=awei' policy.atol_rel=$ATOL_REL policy.delta=$DELTA policy.track_attitude=$TRACK_ATTITUDE 'seed=range(1,11)' +instance_set/BBOB=default n_eval_episodes=1 +slurm=noctua instance_set.fid=$fid +dim=8d instance_set.instance=$inst -m
        done
    done



