#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:30:48 2021

@author: numina
"""
import inspect

import numpy as np
from smac.epm.base_epm import BaseEPM
from smac.epm.gaussian_process import BaseModel
from smac.epm.gaussian_process.gp import GaussianProcess
from smac.epm.gaussian_process.kernels import (ConstantKernel, HammingKernel,
                                               Matern, WhiteKernel)
from smac.epm.gaussian_process.mcmc import MCMCGaussianProcess
from smac.epm.gaussian_process.utils.prior import (HorseshoePrior,
                                                   LognormalPrior)
from smac.epm.random_forest.rf_with_instances import RandomForestWithInstances
from smac.epm.utils import get_rng, get_types
from smac.facade.smac_ac_facade import SMAC4AC
from smac.initial_design.sobol_design import SobolDesign
# model
# acquisition function
from smac.optimizer.acquisition import (EI, TS, AbstractAcquisitionFunction,
                                        IntegratedAcquisitionFunction, LogEI)
from smac.optimizer.acquisition.maximizer import (AcquisitionFunctionMaximizer,
                                                  LocalAndSortedRandomSearch,
                                                  RandomSearch)
from smac.runhistory.runhistory import DataOrigin, RunHistory
from smac.runhistory.runhistory2epm import (RunHistory2EPM4Cost,
                                            RunHistory2EPM4LogScaledCost)
from smac.scenario.scenario import Scenario
from smac.utils.constants import MAXINT
from smac.utils.io.input_reader import InputReader


def build_configuration_space(cs_file: str):
    cs = InputReader().read_pcs_file(cs_file)
    return cs


# TODO automatically write configspace files based on available SOPs
def get_runhistory_data(runhistory: RunHistory, save_external: bool = False) -> dict:
    """
    Shamelessly copied from RunHistory object.

    Parameters
    ----------
    runhistory: RunHistory
    save_external : bool
        Whether to save external data in the runhistory file.

    Returns
    -------
    runhistory_data: dict
        "data": list(tuple(list, list))
            One list entry looks as follows:
            ([int(config_id), str(instance_id) or None, int(seed), float(budget) or 0 if not given], runinfo)
            Runinfo: [?, cost, StatusType, time_start, time_end, info(?, is a dict)]
        "config_origins": dict
            Keys: config ids, values: origin.
        "configs": dict
            Keys: config ids, values: configuration.

    """
    data = [([int(k.config_id),
              str(k.instance_id) if k.instance_id is not None else None,
              int(k.seed),
              float(k.budget) if k[3] is not None else 0], list(v))
            for k, v in runhistory.data.items()
            if save_external or runhistory.external[k] == DataOrigin.INTERNAL]
    config_ids_to_serialize = set([entry[0][0] for entry in data])
    configs = {id_: conf.get_dictionary()
               for id_, conf in runhistory.ids_config.items()
               if id_ in config_ids_to_serialize}
    config_origins = {id_: conf.origin
                      for id_, conf in runhistory.ids_config.items()
                      if (id_ in config_ids_to_serialize and conf.origin is not None)}

    runhistory_data = {
        "data": data,
        "config_origins": config_origins,
        "configs": configs}

    return runhistory_data


def get_acquisition_function_instance(
        model_instance: BaseEPM,
        scenario: Scenario,
        rng: np.random.RandomState,
        acquisition_function_kwargs: dict = None,
        acquisition_function: AbstractAcquisitionFunction = None,
        integrate_acquisition_function: bool = False,
        acquisition_function_optimizer_kwargs: dict = None,
        acquisition_function_optimizer: AcquisitionFunctionMaximizer = None,
):
    # initial acquisition function
    acq_def_kwargs = {'model': model_instance}
    if acquisition_function_kwargs is not None:
        acq_def_kwargs.update(acquisition_function_kwargs)
    if acquisition_function is None:
        if scenario.transform_y in ["LOG", "LOGS"]:  # type: ignore[attr-defined] # noqa F821
            acquisition_function_instance = (
                LogEI(**acq_def_kwargs)  # type: ignore[arg-type] # noqa F821
            )  # type: AbstractAcquisitionFunction
        else:
            acquisition_function_instance = EI(**acq_def_kwargs)  # type: ignore[arg-type] # noqa F821
    elif inspect.isclass(acquisition_function):
        acquisition_function_instance = acquisition_function(**acq_def_kwargs)
    else:
        raise TypeError(
            "Argument acquisition_function must be None or an object implementing the "
            "AbstractAcquisitionFunction, not %s."
            % type(acquisition_function)
        )
    if integrate_acquisition_function:
        acquisition_function_instance = IntegratedAcquisitionFunction(
            acquisition_function=acquisition_function_instance,
            **acq_def_kwargs
        )

    # initialize optimizer on acquisition function
    acq_func_opt_kwargs = {
        'acquisition_function': acquisition_function_instance,
        'config_space': scenario.cs,  # type: ignore[attr-defined] # noqa F821
        'rng': rng,
    }
    if acquisition_function_optimizer_kwargs is not None:
        acq_func_opt_kwargs.update(acquisition_function_optimizer_kwargs)
    if acquisition_function_optimizer is None:
        if isinstance(acquisition_function_instance, TS):
            acquisition_function_optimizer_class = RandomSearch
            # filter unwanted kwargs
            allowed_keys = ["acquisition_function", "rng", "config_space"]
            delete_keys = [k for k in acq_func_opt_kwargs if k not in allowed_keys]
            for k in delete_keys:
                del acq_func_opt_kwargs[k]
        else:
            # adjust kwargs for LocalAndSortedRandomSearch
            for key, value in {
                'max_steps': scenario.sls_max_steps,  # type: ignore[attr-defined] # noqa F821
                'n_steps_plateau_walk': scenario.sls_n_steps_plateau_walk,  # type: ignore[attr-defined] # noqa F821
            }.items():
                if key not in acq_func_opt_kwargs:
                    acq_func_opt_kwargs[key] = value
            acquisition_function_optimizer_class = LocalAndSortedRandomSearch
        acquisition_function_optimizer_instance = (
            acquisition_function_optimizer_class(**acq_func_opt_kwargs)  # type: ignore[arg-type] # noqa F821
        )  # type: AcquisitionFunctionMaximizer
    elif inspect.isclass(acquisition_function_optimizer):
        acquisition_function_optimizer_instance = acquisition_function_optimizer(
            **acq_func_opt_kwargs)  # type: ignore[arg-type] # noqa F821
    else:
        raise TypeError(
            "Argument acquisition_function_optimizer must be None or an object implementing the "
            "AcquisitionFunctionMaximizer, but is '%s'" %
            type(acquisition_function_optimizer)
        )

    return acquisition_function_instance, acquisition_function_optimizer_instance


def get_gp_kernel(scenario: Scenario, rng: np.random.RandomState):
    types, bounds = get_types(scenario.cs, scenario.feature_array)  # type: ignore[attr-defined] # noqa F821

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

    assert (len(cont_dims) + len(cat_dims)) == len(scenario.cs.get_hyperparameters())

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


def get_model_instance(
        scenario: Scenario,
        rng: np.random.RandomState,
        gp_seed: int = None,
        model_type: str = "gp",
        model: BaseEPM = None,
        model_kwargs: dict = None,
):
    valid_model_types = ["gp", "gp_mcmc", "random_forest"]
    model_kwargs = model_kwargs or dict()

    kernel = get_gp_kernel(scenario=scenario, rng=rng)

    if model_type == "gp":
        model_class = GaussianProcess  # type: typing.Type[BaseModel]
        model = model_class
        model_kwargs['kernel'] = kernel
        model_kwargs['normalize_y'] = True
        if gp_seed is not None:
            model_kwargs['seed'] = gp_seed
        else:
            model_kwargs['seed'] = rng.randint(0, 2 ** 20)
    elif model_type == "gp_mcmc":
        model_class = MCMCGaussianProcess
        model = model_class
        # kwargs['integrate_acquisition_function'] = True

        model_kwargs['kernel'] = kernel

        n_mcmc_walkers = 3 * len(kernel.theta)
        if n_mcmc_walkers % 2 == 1:
            n_mcmc_walkers += 1
        model_kwargs['n_mcmc_walkers'] = n_mcmc_walkers
        model_kwargs['chain_length'] = 250
        model_kwargs['burnin_steps'] = 250
        model_kwargs['normalize_y'] = True
        if gp_seed is not None:
            model_kwargs['seed'] = gp_seed
        else:
            model_kwargs['seed'] = rng.randint(0, 2 ** 20)
    elif model_type == "random_forest":
        for key, value in {
            'log_y': scenario.transform_y in ["LOG", "LOGS"],  # type: ignore[attr-defined] # noqa F821
            'num_trees': scenario.rf_num_trees,  # type: ignore[attr-defined] # noqa F821
            'do_bootstrapping': scenario.rf_do_bootstrapping,  # type: ignore[attr-defined] # noqa F821
            'ratio_features': scenario.rf_ratio_features,  # type: ignore[attr-defined] # noqa F821
            'min_samples_split': scenario.rf_min_samples_split,  # type: ignore[attr-defined] # noqa F821
            'min_samples_leaf': scenario.rf_min_samples_leaf,  # type: ignore[attr-defined] # noqa F821
            'max_depth': scenario.rf_max_depth,  # type: ignore[attr-defined] # noqa F821
        }.items():
            if key not in model_kwargs:
                model_kwargs[key] = value
        model_class = RandomForestWithInstances
        model = model_class
    else:
        raise ValueError(f'model_type should be one of {valid_model_types}')

    # initial EPM
    types, bounds = get_types(scenario.cs, scenario.feature_array)  # type: ignore[attr-defined] # noqa F821
    model_def_kwargs = {
        'types': types,
        'bounds': bounds,
        'instance_features': scenario.feature_array,
        'seed': rng.randint(MAXINT),
        'pca_components': scenario.PCA_DIM,
    }
    if model_kwargs is not None:
        model_def_kwargs.update(model_kwargs)
    if model is None:
        raise ValueError("Model shouldn't be None at this stage. :-D")

        # for key, value in {
        #     'log_y': scenario.transform_y in ["LOG", "LOGS"],  # type: ignore[attr-defined] # noqa F821
        #     'num_trees': scenario.rf_num_trees,  # type: ignore[attr-defined] # noqa F821
        #     'do_bootstrapping': scenario.rf_do_bootstrapping,  # type: ignore[attr-defined] # noqa F821
        #     'ratio_features': scenario.rf_ratio_features,  # type: ignore[attr-defined] # noqa F821
        #     'min_samples_split': scenario.rf_min_samples_split,  # type: ignore[attr-defined] # noqa F821
        #     'min_samples_leaf': scenario.rf_min_samples_leaf,  # type: ignore[attr-defined] # noqa F821
        #     'max_depth': scenario.rf_max_depth,  # type: ignore[attr-defined] # noqa F821
        # }.items():
        #     if key not in model_def_kwargs:
        #         model_def_kwargs[key] = value
        # model_def_kwargs['configspace'] = scenario.cs  # type: ignore[attr-defined] # noqa F821
        # model_instance = (
        #     RandomForestWithInstances(**model_def_kwargs)  # type: ignore[arg-type] # noqa F821
        # )  # type: AbstractEPM
    elif inspect.isclass(model):
        model_def_kwargs['configspace'] = scenario.cs  # type: ignore[attr-defined] # noqa F821
        model_instance = model(**model_def_kwargs)  # type: ignore[arg-type] # noqa F821
    else:
        raise TypeError(
            "Model not recognized: %s" % (type(model)))

    return model_instance
