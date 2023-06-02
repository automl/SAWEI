#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:45:22 2021

@author: numina
"""
import datetime
import logging
import os
import sys

DATADIR = "data"
RUNSDIR = "runs"
CONFIGSPACESDIR = "configspaces"


def create_runid():
    # should not be used to initialize numpy random generator
    return int(datetime.datetime.today().strftime("%Y%m%d%H%M%S"))


def create_outputdir(function_name: str, n_dim=None):
    if not n_dim:
        postfix = ""
    else:
        postfix = f"_{n_dim:03}"

    function_name = function_name.lower()
    output_dir = os.path.join(DATADIR, RUNSDIR, f"{function_name}{postfix}")
    return output_dir


def get_cs_fname(function_name: str, n_dim=None, identifier=None):
    if not identifier:
        identifier = 0
        identifier = f"_{identifier:03}"

    out_dir = create_outputdir(function_name, n_dim)
    base = os.path.basename(out_dir)

    cs_fname = os.path.join(DATADIR, CONFIGSPACESDIR, f"{base}{identifier}.json")

    return cs_fname


def get_logging_formatter():
    formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s")
    return formatter


def get_standard_logger(
    identifier: str,
    level: int = logging.INFO,
    stream_format: str = "%(asctime)s|%(levelname)s|%(name)s|%(message)s",
):
    logger = logging.getLogger(identifier)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(stream_format)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level=level)
    stream_handler.setFormatter(formatter)

    handlers = logger.handlers
    for handler in handlers:
        logger.removeHandler(handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    return logger
