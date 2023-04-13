#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:40:42 2021

@author: numina
"""
from enum import Enum, auto

from smac.optimizer.acquisition import EI, EIPS, LCB, PI, TS, LogEI
from smacaux.custom_smac import SimpleLCB

ID_ACQFUNC = "u_"
ID_SURRMODEL = "m_"


def action_translator(action_id):
    action_dict = {
        # acquisition functions
        "u_EI": AcquisitionFunctions.EI,
        "u_PI": AcquisitionFunctions.PI,
        "u_LCB": AcquisitionFunctions.LCB,
        "u_SimpleLCB": AcquisitionFunctions.SimpleLCB,
        "u_EIPS": AcquisitionFunctions.EIPS,
        "u_LOGEI": AcquisitionFunctions.LOGEI,
        "u_TS": AcquisitionFunctions.TS,
        # models
        "m_GP": BOModels.GAUSSIANPROCESS,
        "m_GPMCMC": BOModels.GAUSSIANPROCESSMCMC,
        "m_RF": BOModels.RANDOMFOREST,
    }

    action = None
    if action_id in action_dict:
        action = action_dict[action_id]
    else:
        raise ValueError(f"action_id '{action_id}' unknown. Available action ids are " f"{list(action_dict.keys())}.")

    return action


class AcquisitionFunctions(Enum):
    EI = auto()
    PI = auto()
    LCB = auto()
    EIPS = auto()
    LOGEI = auto()
    SimpleLCB = auto()
    TS = auto()


AF = AcquisitionFunctions

acquisition_function_factory = {
    AF.EI: EI,
    AF.PI: PI,
    AF.EIPS: EIPS,
    AF.LOGEI: LogEI,
    AF.LCB: LCB,
    AF.SimpleLCB: SimpleLCB,
    AF.TS: TS,
}


class BOModels(Enum):
    GAUSSIANPROCESS = auto()
    GAUSSIANPROCESSMCMC = auto()
    RANDOMFOREST = auto()


model_factory = {
    BOModels.GAUSSIANPROCESS: "gp",
    BOModels.GAUSSIANPROCESSMCMC: "gp_mcmc",
    BOModels.RANDOMFOREST: "random_forest",
    # BOModels.GAUSSIANPROCESS: GaussianProcess,
    # BOModels.GAUSSIANPROCESSMCMC: GaussianProcessMCMC,
}
