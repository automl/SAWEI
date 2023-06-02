from typing import Any, Dict

import csv
import os
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from instance_sets.set_creation import create_instance_set
from target_algorithms.registration import TargetAlgorithms


def read_instance_set(kwargs: Dict):
    path = kwargs.get("instance_set_path", None)
    if path is not None:
        instance_set = read_instance_set_from_csv(path)
        instance_set_path = path
    else:
        family_id = kwargs.get("family_id", None)
        ta_set_id = kwargs.get("ta_set_id", None)
        n_dim = kwargs.get("n_dim", 2)
        instance_set, instance_set_path = create_instance_set(
            family_id=family_id,
            ta_set_id=ta_set_id,
            n_dim=n_dim,
            fp_function_families=None,
            handle_uncompatible_dims="warning",
            to_csv=True,
        )
        if instance_set_path.is_absolute():
            if "instance_sets" in str(instance_set_path):
                parts = np.array(instance_set_path.parts)
                id_split = np.where([p == "instance_sets" for p in parts])[0][0]
                new_parts = parts[id_split:]
                new_path = os.path.join(*new_parts)
                instance_set_path = Path(new_path)

    return instance_set, instance_set_path


def select_instance(instance_set: Dict, instance_id: Any):
    return instance_set[instance_id]


def read_instance_set_from_csv(path):
    """
    Read instance set from csv file into self.config["instance_set"].

    Csv filename specified in `config["instance_set_path"]`.
    Style of csv file is as follows (don't leave whitespaces):
        ID;TYPE;DIM;CSFNAME

        ID: Integer index of instance.
        TYPE: Name of synthetic function. Must match entries defined in
            SyntheticFunctions Enum.
        DIM: Number of dimensions.
        CSFNAME: Path to configuration space file (relative to DAC-BO
                                                   directory).

    Returns
    -------
    None.

    """
    try:
        instance_set_df = pd.read_csv(
            path,
            index_col="instance_id",
            sep=None,
            engine="python",
            converters={"parameters": literal_eval},
        )
        instance_set_df["configuration_space_file"].replace(np.nan, "", inplace=True)
        instance_set = instance_set_df.T.to_dict()
    except ValueError:
        instance_set = {}
        with open(path, "r") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for row in reader:
                idx = int(row["ID"])
                function_type = row["TYPE"]
                n_dim = int(row["DIM"])
                cs_file = row["CSFNAME"]

                instance = {
                    "target_algorithm": str(TargetAlgorithms[function_type]),
                    "configuration_space_file": cs_file,
                    "n_dim": n_dim,
                }
                instance_set[idx] = instance
    return instance_set
