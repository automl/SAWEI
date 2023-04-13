from pathlib import Path

from utils.json_utils import lazy_json_dump

"""
Saving structure:

benchmarks/benchmark_name/action_space_id/state_id/synthfunname
"""

base_dir = Path("benchmarks")
benchmark_name = "SynthFunctionBenchmark"

# ACTION SPACE
action_values = ["u_EI", "u_SimpleLCB"]

reward_range = [-1e10, 1e10]
budget = 50

# STATE SPACE
observation_space_description = [
    "remaining_steps",
    # "last_action",
    "history_raw",
]

instance_set_paths = [
    "instance_sets/synthetic_functions/manylocaloptima2d.csv",
    "instance_sets/synthetic_functions/manylocaloptima2dtest.csv",
]

# DON'T TOUCH
state_description_map = {
    "remaining_steps": "Remaining Budget",
    "last_action": "Last Action(s)",
    "history_raw": "History of Function Values",
}
state_description = [state_description_map[obs_type] for obs_type in observation_space_description]

config = {
    "action_space_class": "Discrete",
    "action_space_args": [len(action_values)],
    "action_values": action_values,
    "observation_space_class": "Box",
    "observation_space_type": "<class 'numpy.float32'>",
    "observation_space_description": observation_space_description,
    "reward_range": reward_range,
    "budget": budget,
    "cutoff": budget,
    "seed": 0,
    "benchmark_info": {
        "identifier": "synthetic_functions",
        "name": "Synthetic Function Optimization",
        "reward": "Negative best function value",
        "state_description": state_description,
    },
    "no_logs_to_file": True,
    "wrappers": [],
}

action_space_id = "__".join(action_values)
state_space_id = "__".join(observation_space_description)

for instance_set_path in instance_set_paths:
    # adjust function specific data
    config["instance_set_path"] = instance_set_path

    # if synth_fun_name == SyntheticFunctions.QUADRATIC:
    #     config["budget"] = config["cutoff"] = 25
    #
    # print(synth_fun_name.name)
    # stem = f"config_{synth_fun_name.name}.json"
    instance_set_path = Path(instance_set_path)
    stem = f"config_{instance_set_path.stem}.json"
    path = base_dir / benchmark_name / action_space_id / state_space_id
    path.mkdir(parents=True, exist_ok=True)
    fname = path / stem
    lazy_json_dump(data=config, filename=fname)
