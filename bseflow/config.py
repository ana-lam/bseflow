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
    "compas_fields": {
        "groups": {
            "systems": "BSE_System_Parameters",
            "formationChannels": "BSE_System_Parameters",
            "commonEnvelopes": "BSE_Common_Envelopes",
            "RLOF": "BSE_RLOF",
            "supernovae": "BSE_Supernovae",
            "doubleCompactObjects": "BSE_Double_Compact_Objects",
        },
        "fields": {
            # BSE_System_Parameters
            "stellar_merger": "Stellar_Merger",
            "disbound": "Unbound",
            "weight": None,
            "Metallicity1": "Metallicity@ZAMS(1)",
            "mass1": "Mass@ZAMS(1)",
            "mass2": "Mass@ZAMS(2)",
            "stellar_type_K1": "Stellar_Type(1)",
            "stellar_type_K2": "Stellar_Type(2)",
            # BSE_Common_Envelopes
            "stellarType1": "Stellar_Type(1)<CE",
            "stellarType2": "Stellar_Type(2)<CE",
            "stellarMerger": "Merger",
            "finalStellarType1": "Stellar_Type(1)",
            "finalStellarType2": "Stellar_Type(2)",
            "optimisticCommonEnvelopeFlag": "Optimistic_CE",
            "randomSeed": "SEED",
            # BSE_RLOF
            "radius1": "Radius(1)<MT",
            "radius2": "Radius(2)<MT",
            "flagCEE": "CEE>MT",
            "type1": "Stellar_Type(1)>MT",
            "type2": "Stellar_Type(2)>MT",
            "type1Prev": "Stellar_Type(1)<MT",
            "type2Prev": "Stellar_Type(2)<MT",
            "flagRLOF1": "RLOF(1)<MT",
            "flagRLOF2": "RLOF(2)<MT",
            # BSE_Supernovae
            "Survived": "Unbound",
            "previousStellarTypeSN": "Stellar_Type_Prev(SN)",
            "previousStellarTypeCompanion": "Stellar_Type(CP)",
            "flagPISN": "SN_Type(SN)",
            "flagPPISN": "SN_Type(SN)",
            # BSE_Double_Compact_Objects
            "mergesInHubbleTimeFlag": "Merges_Hubble_Time",
            "M1": "Mass(1)",
            "M2":  "Mass(2)",
        },
        "sn_type_codes": {
            "CCSN": 1,
            "ECSN": 2,
            "PISN": 4,
            "PPISN": 8,
            "USSN": 16,
            "AIC": 32,
            "SNIA": 64,
            "HeSD": 128,
        },
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
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load bseflow.yaml. "
                "Install it with:  pip install pyyaml"
            )
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                _cached_config = data
                return _cached_config
            else:
                warnings.warn("bseflow.yaml exists but is not a valid dictionary. Falling back to defaults.")
        except Exception as e:
            warnings.warn("bseflow: could not parse bseflow.yaml: {} — falling back to defaults.".format(e))

    # no bseflow.yaml in cwd — fall back to the bundled template
    import importlib.resources as pkg_resources
    try:
        import yaml
        with pkg_resources.open_text("bseflow", "default_config.yaml") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            _cached_config = data
            return _cached_config
    except Exception as e:
        warnings.warn("bseflow: could not load bundled default_config.yaml: {} — using hardcoded defaults.".format(e))

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