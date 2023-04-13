from __future__ import annotations

import typing

import warnings

import numpy as np
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant
from scipy.stats.qmc import Sobol
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gaussian_process.kernels import (
    ConstantKernel,
    HammingKernel,
    Matern,
    WhiteKernel,
)
from smac.epm.gaussian_process.utils.prior import HorseshoePrior, LognormalPrior
from smac.epm.utils import get_rng, get_types
from smac.facade.smac_ac_facade import SMAC4AC
from smac.initial_design.sobol_design import SobolDesign
from smac.intensification.intensification import Intensifier
from smac.optimizer.configuration_chooser.random_chooser import ChooserProb
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats

from dacbo.utils.action_handling import (
    AcquisitionFunctions,
    acquisition_function_factory,
)
from dacbo.utils.target_algorithm_handling import get_tae_cs, get_tae_runner


class CustomSobolDesign(SobolDesign):
    def _select_configurations(self) -> typing.List[Configuration]:
        """Selects a single configuration to run

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """

        params = self.cs.get_hyperparameters()

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        dim = len(params) - constants
        sobol_gen = Sobol(
            d=dim,
            scramble=True,
            seed=53,  # self.rng.randint(low=0, high=10000000)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sobol = sobol_gen.random(self.init_budget)

        return self._transform_continuous_designs(design=sobol, origin="Sobol", cs=self.cs)


def get_costs(smac: SMAC4AC):
    rhdata = smac.get_runhistory().data
    costs = np.array([v.cost for v in rhdata.values()])
    return costs


def optimize_smac(smac: SMAC4AC):
    smac.optimize()
    rh = smac.get_runhistory()
    rhdata = rh.data
    costs = np.array([v.cost for v in rhdata.values()])
    # traj = smac.get_trajectory()
    # traj = np.array([t.train_perf for t in traj])
    traj = np.array(list(np.amin(costs[: i + 1]) for i in range(len(costs))))
    return traj


def get_gp_kernel(rng, cs):
    _, rng = get_rng(rng=rng)

    types, bounds = get_types(cs, instance_features=None)

    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(np.exp(-10), np.exp(2)),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
    )

    cont_dims = np.where(np.array(types) == 0)[0]
    cat_dims = np.where(np.array(types) != 0)[0]

    if len(cont_dims) > 0:
        exp_kernel = Matern(
            np.ones([len(cont_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cont_dims))],
            nu=2.5,
            operate_on=cont_dims,
        )

    if len(cat_dims) > 0:
        ham_kernel = HammingKernel(
            np.ones([len(cat_dims)]),
            [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cat_dims))],
            operate_on=cat_dims,
        )

    assert (len(cont_dims) + len(cat_dims)) == len(cs.get_hyperparameters())

    noise_kernel = WhiteKernel(
        noise_level=1e-8,
        noise_level_bounds=(np.exp(-25), np.exp(2)),
        prior=HorseshoePrior(scale=0.1, rng=rng),
    )

    if len(cont_dims) > 0 and len(cat_dims) > 0:
        # both
        kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
    elif len(cont_dims) > 0 and len(cat_dims) == 0:
        # only cont
        kernel = cov_amp * exp_kernel + noise_kernel
    elif len(cont_dims) == 0 and len(cat_dims) > 0:
        # only cont
        kernel = cov_amp * ham_kernel + noise_kernel
    else:
        raise ValueError()

    return kernel


def get_acq_fun(acquisition_function_id: AcquisitionFunctions):
    acquisition_function = acquisition_function_factory[acquisition_function_id]
    acquisition_function_kwargs = {}
    if acquisition_function_id == AcquisitionFunctions.LCB:
        acquisition_function_kwargs = {
            "par": 0.9999,  # 10,  # default = 1 (exploration parameter)
        }
    elif acquisition_function_id == AcquisitionFunctions.EI:
        acquisition_function_kwargs = {
            "par": 0.0,  # 0.1,  # default=0, little exploration
        }
    elif acquisition_function_id == AcquisitionFunctions.SimpleLCB:
        acquisition_function_kwargs = {"par": 10}  # let's explore!
    elif acquisition_function_id == AcquisitionFunctions.PI:
        acquisition_function_kwargs = {}

    # model_type = "gp"
    # integrate_acquisition_function = True if model_type == "gp_mcmc" else False
    # acquisition_function_optimizer_kwargs = {'n_sls_iterations': 10}
    # acquisition_function_instance, acquisition_function_optimizer_instance = get_acquisition_function_instance(
    #     model_instance=model_instance,
    #     scenario=smac.scenario,
    #     rng=rng,
    #     acquisition_function_kwargs=acquisition_function_kwargs,
    #     acquisition_function=acquisition_function,
    #     integrate_acquisition_function=integrate_acquisition_function,
    #     acquisition_function_optimizer_kwargs=acquisition_function_optimizer_kwargs,
    #     acquisition_function_optimizer=None,
    # )

    return acquisition_function, acquisition_function_kwargs


def set_stats_scenario(stats: Stats, scenario: Scenario):
    # can't do: self.stats.__scenario = self.scenario
    # because internally it is saved into _BOEnv__scenario, not _Stats__scenario
    setattr(stats, f"_{type(stats).__name__}__scenario", scenario)


def build_smac_kwargs(
    budget,
    target_algorithm,
    seed,
    acquisition_function,
    acquisition_function_kwargs,
    smac_outdir=None,
    runhistory=None,
    restore_incumbent=None,
    stats=None,
    initial_budget: int | None = None,
):
    rng = seed
    _, rng = get_rng(rng=rng)

    tae_runner = get_tae_runner(target_algorithm)
    cs = get_tae_cs(target_algorithm)
    if initial_budget is None:
        initial_budget = 3 * len(cs.get_hyperparameters_dict())

    budget = (
        budget + initial_budget
    )  # budget is cstep (1 based)  # TODO creates an initial design of initial_budget + 1 points
    if runhistory is None:
        len_rh = 0
    else:
        len_rh = len(runhistory)
    # cs.seed(rng.randint(MAXINT))

    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount_limit": budget,
            # max. number of function evaluations; for this example set to a low number
            "cs": cs,  # configuration space
            "deterministic": "true",
            "output_dir": smac_outdir,
        }
    )
    scenario.intensification_percentage = 1e-10  # TODO what does this do?

    if stats is not None:
        set_stats_scenario(stats=stats, scenario=scenario)

    kernel = get_gp_kernel(rng=rng, cs=cs)
    model_cls = GaussianProcess
    model_kwargs = {
        "kernel": kernel,
        "normalize_y": True,
        # "seed": rng.randint(MAXINT),  # seed if type(seed) is int else None,  # seed can also be a random generator
    }

    smac_kwargs = dict(
        rng=rng,
        scenario=scenario,
        tae_runner=tae_runner,
        tae_runner_kwargs=None,
        runhistory=runhistory,
        intensifier=Intensifier,
        intensifier_kwargs={"min_chall": 1},  # only 1 configuration per SMBO iteration
        acquisition_function=acquisition_function,
        acquisition_function_kwargs=acquisition_function_kwargs,
        integrate_acquisition_function=False,  # set to True if using GP_mcmc
        acquisition_function_optimizer=None,
        acquisition_function_optimizer_kwargs=None,  # TODO n_sls_iterations?
        model=model_cls,
        model_kwargs=model_kwargs,
        runhistory2epm=RunHistory2EPM4Cost,
        runhistory2epm_kwargs=None,
        initial_design=SobolDesign,
        initial_design_kwargs={  # static
            # "n_configs_x_params": 2,
            # "max_config_fracs": 0.05,
            "init_budget": initial_budget,
        },
        stats=stats,
        restore_incumbent=restore_incumbent,
        random_configuration_chooser=ChooserProb,
        random_configuration_chooser_kwargs={"prob": 0},
    )
    return smac_kwargs
