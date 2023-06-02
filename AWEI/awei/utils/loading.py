from pathlib import Path
from multiprocessing import Pool
import pandas as pd
from rich import print as printr
from itertools import chain
from omegaconf import OmegaConf, ListConfig
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from ioh import ProblemType

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
    
        # instances = df["instance"]
    # entries = [instance_set[i] for i in instances]
    df["benchmark"] = cfg.benchmark
    if cfg.benchmark == "BBOB":
        instance_set = create_instance_set(cfg)
        if len(instance_set) == 1:
            import ioh
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
    elif cfg.benchmark.startswith("HPOBench"):
        if type(cfg.instance_set) == ListConfig:
            assert len(cfg.instance_set) == 1
            cfg_instance = cfg.instance_set[0]
        else:
            cfg_instance = cfg.instance_set
        df["task_id"] = cfg_instance.task_id
        df["model"] = cfg_instance.model

        # instance_set = create_instance_set(cfg=cfg)
        # df["y_opt"] = instance_set[0].y_optimum

    df.loc[(df["reward"].isna()) & (df["cost"].notna()), "reward"] = np.inf 
    
    n_initial_design = cfg.budget_doe

    if (p.parent / ubr_fn).is_file():
        try:
            ubrs = pd.read_json(p.parent / ubr_fn, orient="split")
            ubr = ubrs["ubr"]
        except ValueError:
            ubrs = pd.read_json(p.parent / ubr_fn)
            ubr = -ubrs["data"]  # TODO check this

        if "min_ucb" in ubrs:
            del ubrs["min_ucb"]
        if "min_lcb" in ubrs:
            del ubrs["min_lcb"]

        X = ubrs["n_evaluated"] - n_initial_design
        ubrs["step"] = X
        ubrs["ubr"] = ubr

        df = pd.merge(df, ubrs, how="left", on="step")
        # df["ubr"] = ubr

        df["ubr_gradient"] = np.gradient(df["ubr"])
        if "y_opt" in df:
            df["ubr-opt"] = df["ubr"] - df["y_opt"]
    
    if (p.parent / wei_fn).is_file():
        try:
            weis = pd.read_json(p.parent / wei_fn, orient="split")
            X = weis["n_evaluated"] - n_initial_design
            weis["step"] = X

            keep = ["alpha", "step"]
            ignore = [c for c in weis.columns if c not in keep]
            for i in ignore:
                del weis[i]

            df = pd.merge(df, weis, how="left", on="step")
            # df["alpha"] = weis["alpha"]
            # df["pi_term"] = weis["pi_term"]
            # df["ei_term"] = weis["ei_term"]
        except KeyError:
            pass


    return df


def read_results(paths, fn=read_run):
    with Pool(processes=None) as pool:
        rollout_df = pool.map(fn, paths)
    if rollout_df:
        rollout_df = pd.concat(rollout_df).reset_index()
    return rollout_df
