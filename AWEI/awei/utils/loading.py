from pathlib import Path
from multiprocessing import Pool
import pandas as pd
from rich import print as printr
from itertools import chain
from omegaconf import OmegaConf, ListConfig
import numpy as np
from pathlib import Path
from multiprocessing import Pool

from dacbo.instances import create_instance_set


rollout_data_fn = "rollout_data.json"
hydra_cfg_fn = ".hydra/config.yaml"
ubr_fn = "ubr_history.json"
wei_fn = "wei_history.json"


def read_run(p) -> pd.DataFrame:
    p = Path(p)
    df = pd.read_json(p, orient="split")

    cfg_fn = Path(p).parent / hydra_cfg_fn
    cfg = OmegaConf.load(cfg_fn)
    
    instance_set = create_instance_set(cfg)
    # instances = df["instance"]
    # entries = [instance_set[i] for i in instances]
    df["benchmark"] = cfg.benchmark
    if cfg.benchmark == "BBOB":
        if len(instance_set) == 1:
            import ioh
            from ioh import ProblemType
            if type(cfg.instance_set) == ListConfig:
                assert len(cfg.instance_set) == 1
                cfg_instance = cfg.instance_set[0]
            else:
                cfg_instance = cfg.instance_set
            df["fid"] = cfg_instance.fid
            df["bbob_instance"] = cfg_instance.instance
            df["dimension"] = cfg_instance.dimension
            problem = ioh.get_problem(
                fid=cfg_instance.fid,
                instance=cfg_instance.instance,
                dimension=cfg_instance.dimension,
                problem_type=ProblemType.BBOB,
            )
            df["x_opt"] = [problem.optimum.x] * len(df)
            df["y_opt"] = problem.optimum.y
        else:
            raise NotImplementedError

    if (p.parent / ubr_fn).is_file():
        try:
            ubrs = pd.read_json(p.parent / ubr_fn, orient="split")
            ubr = ubrs["ubr"]
        except ValueError:
            ubrs = pd.read_json(p.parent / ubr_fn)
            ubr = -ubrs["data"]  # TODO check this
        df["ubr"] = ubr
        df["ubr_gradient"] = np.gradient(df["ubr"])
        df["ubr-opt"] = df["ubr"] - df["y_opt"]
    
    if (p.parent / wei_fn).is_file():
        try:
            weis = pd.read_json(p.parent / wei_fn, orient="split")
            df["alpha"] = weis["alpha"]
            df["pi_term"] = weis["pi_term"]
            df["ei_term"] = weis["ei_term"]
        except KeyError:
            pass


    return df


def read_results(paths, fn=read_run):
    with Pool(processes=None) as pool:
        rollout_df = pool.map(fn, paths)
    if rollout_df:
        rollout_df = pd.concat(rollout_df).reset_index()
    return rollout_df
