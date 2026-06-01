"""
bseflow.config
--------------
Loads the bseflow config from bseflow.yaml in the working directory
"""

import os
import warnings
from pathlib import Path
import yaml

_DEFAULTS = {
    "output": {
        "rates_dir": "bseflow_output",
        "seeds_subdir": "fc_stage_seeds",
        "sankey_dir": "sankey_htmls",
    },
    "plotting": {
        "usetex": True,
        "fontsize": 20,
    },
}

_cached_config = None

def get_config(reload=False):
    """
    Load the bseflow config from bseflow.yaml in the working directory if it exists. Cache result.
    """
    global _cached_config
    if _cached_config is not None and not reload:
        return _cached_config

    config_path = Path.cwd() / "bseflow.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                _cached_config = data
                return _cached_config
            else:
                warnings.warn(f"bseflow.yaml exists but is not a valid dictionary. Using defaults.")
        except Exception as e:
            warnings.warn("bseflow: could not parse bseflow.yaml: {} — using defaults.".format(e))
    
    _cached_config = _DEFAULTS
    return _cached_config

# --- helpers ----

def get_rates_dir():
    return get_config()["output"]["rates_dir"]

def get_seeds_subdir():
    cfg = get_config()["output"]
    return os.path.join(cfg["rates_dir"], cfg["seeds_subdir"])

def get_sankey_dir():
    return get_config()["output"]["sankey_dir"]

def get_usetex():
    return get_config()["plotting"]["usetex"]

def get_fontsize():
    return get_config()["plotting"]["fontsize"]

def get_group(internal_name):
    """Translate an internal group name to the COMPAS HDF5 group name"""
    return get_config()['compas_fields']['groups'][internal_name]

def get_field(internal_name):
    """Translate an internal field name to the COMPAS HDF5 field name"""
    return get_config()['compas_fields']['fields'][internal_name]

def get_sn_type_code(sn_type_name):
    """Translate an internal SN type name to its integer code, (e.g. PISN->4)"""
    return get_config()['compas_fields']['sn_type_codes'][sn_type_name]