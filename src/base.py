"""
Module containing base utility functions used for the benchmarks.

"""

from __future__ import annotations

import importlib.util
from typing import Any

import pandas as pd

# NOTE: dataset loaders are imported lazily inside get_dataloader() so that each
# per-package virtual environment only needs the dependencies of the package(s)
# it actually uses (e.g. an environment running tsfresh on the four paper
# datasets does not need librosa/soundfile, which are only required by the
# optional IDMT/MIMII audio loaders).

def match_configuration_to_feature_extraction_methods(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Match the configuration to the feature extraction methods.

    Parameters
    ----------
    configuration : dict[str, Any]
        Configuration dictionary containing the settings for the feature extraction methods.

    Returns
    -------
    dict[str, Any]
        Updated configuration dictionary with the settings for the feature extraction methods.

    """
    if importlib.util.find_spec("tsfresh") is not None:
        configuration["tsfresh"] = match_tsfresh_settings(configuration["tsfresh"])
    if importlib.util.find_spec("tsfel") is not None:
        configuration["tsfel"] = match_tsfel_settings(configuration["tsfel"])
    if importlib.util.find_spec("seglearn") is not None:
        configuration["seglearn"] = match_seglearn_settings(configuration["seglearn"])
    if importlib.util.find_spec("kats") is not None:
        configuration["kats"] = match_kats_settings(configuration["kats"])
    if importlib.util.find_spec("tsfeatures") is not None:
        configuration["tsfeatures"] = match_tsfeatures_settings(configuration["tsfeatures"])

    configuration["pycatch22"] = match_pycatch22_settings(configuration["pycatch22"])

    return configuration

def match_tsfresh_settings(tsfresh_settings: dict[str, Any]) -> dict[str, Any]:
    """Match the tsfresh settings to the feature extraction methods."""
    try:
        from tsfresh.feature_extraction import (
            ComprehensiveFCParameters,
            EfficientFCParameters,
            MinimalFCParameters,
        )
    except ImportError as e:
        msg = "tsfresh is not installed."
        raise ModuleNotFoundError(msg) from e

    if tsfresh_settings["default_fc_parameters"] == "minimal":
        default_fc_parameters = MinimalFCParameters()
    elif tsfresh_settings["default_fc_parameters"] == "efficient":
        default_fc_parameters = EfficientFCParameters()
    elif tsfresh_settings["default_fc_parameters"] == "comprehensive":
        default_fc_parameters = ComprehensiveFCParameters()
    else:
        msg = "Invalid default_fc_parameters value in configuration file."
        raise ValueError(msg)
    tsfresh_settings["default_fc_parameters"] = default_fc_parameters

    if tsfresh_settings["n_jobs"] == "none":
        tsfresh_settings["n_jobs"] = None

    return tsfresh_settings

def match_tsfel_settings(tsfel_settings: dict[str, Any]) -> dict[str, Any]:
    """Match the tsfel settings to the feature extraction methods."""
    try:
        import tsfel
    except ImportError as e:
        msg = "TSFEL is not installed."
        raise ModuleNotFoundError(msg) from e

    if tsfel_settings["domain"] == "none":
        cfg = tsfel.get_features_by_domain()
    elif tsfel_settings["domain"] == "statistical":
        cfg = tsfel.get_features_by_domain("statistical")
    elif tsfel_settings["domain"] == "temporal":
        cfg = tsfel.get_features_by_domain("temporal")
    elif tsfel_settings["domain"] == "spectral":
        cfg = tsfel.get_features_by_domain("spectral")
    else:
        msg = "Invalid domain value in configuration file."
        raise ValueError(msg)
    tsfel_settings["cfg"] = cfg

    return tsfel_settings

def match_pycatch22_settings(pycatch22_settings: dict[str, Any]) -> dict[str, Any]:
    """Match the pycatch22 settings to the feature extraction methods."""
    if pycatch22_settings["catch24"] == "False":
        catch24 = False
    elif pycatch22_settings["catch24"] == "True":
        catch24 = True
    else:
        msg = "Invalid catch24 value in configuration file."
        raise ValueError(msg)
    pycatch22_settings["catch24"] = catch24

    return pycatch22_settings

def match_seglearn_settings(seglearn_settings: dict[str, Any]) -> dict[str, Any]:
    """Match the seglearn settings to the feature extraction methods."""
    try:
        from seglearn import all_features, base_features
    except ImportError as e:
        msg = "seglearn is not installed."
        raise ModuleNotFoundError(msg) from e

    if seglearn_settings["features_to_extract"] == "default":
        features_to_extract = base_features()
    elif seglearn_settings["features_to_extract"] == "all":
        features_to_extract = all_features()
    else:
        msg = "Invalid features value in configuration file."
        raise ValueError(msg)
    seglearn_settings["features_to_extract"] = features_to_extract

    return seglearn_settings

def match_tsfeatures_settings(tsfeatures_settings: dict[str, Any]) -> dict[str, Any]:
    """Match the tsfeatures settings to the feature extraction methods."""
    try:
        from tsfeatures import tsfeatures
    except ImportError as e:
        msg = "tsfeatures is not installed."
        raise ModuleNotFoundError(msg) from e

    if tsfeatures_settings["threads"] == "none":
        tsfeatures_settings["threads"] = None

    return tsfeatures_settings

def match_kats_settings(kats_settings: dict[str, Any]) -> dict[str, Any]:
    """Match the kats settings to the feature extraction methods."""
    if isinstance(kats_settings["selected_features"], str):
        kats_settings["selected_features"] = (
            None
            if kats_settings["selected_features"] == "none"
            else [kats_settings["selected_features"]]
        )

    return kats_settings

def get_dataloader(dataset: str) -> object:
    """Get the correct dataloader for the selected dataset.

    Loaders are imported lazily so that ``import src.base`` only requires
    ``pandas``; heavy/optional dependencies (e.g. ``h5py`` for Bosch, ``librosa``
    for the audio datasets) are only required by the environments that use them.

    Raises
    ------
    ModuleNotFoundError
        If the loader's dependencies are not installed in the active environment.
    ValueError
        If the dataset value in the configuration file is invalid.
    """
    if dataset == "CNC_Mill_Tool_Wear":
        from src.loading.cnc_mill_tool_wear import Dataloader as CNCDataloader

        dataloader = CNCDataloader()
    elif dataset == "Condition_Monitoring_of_hydraulic_systems":
        from src.loading.condition_monitoring_of_hydraulic_systems import (
            Dataloader as HydraulicDataloader,
        )

        dataloader = HydraulicDataloader()
    elif dataset == "Turning_Dataset_for_Chatter_Diagnosis":
        from src.loading.turning_chatter import Dataloader as TurningDataloader

        dataloader = TurningDataloader()
    elif dataset == "IDMT-ISA_Compressed_Air":
        from src.loading.idmt_isa_compressed_air_dataset import (
            Dataloader as IDMTDataloader,
        )

        dataloader = IDMTDataloader()
    elif dataset == "MIMII":
        from src.loading.mimii import Dataloader as MIMIIDataloader

        dataloader = MIMIIDataloader()
    elif dataset == "Bosch_CNC":
        from src.loading.bosch_cnc_machining import Dataloader as BoschDataloader

        dataloader = BoschDataloader()
    else:
        msg = "Invalid dataset value in configuration file."
        raise ValueError(msg)

    return dataloader

def get_feature_extractor(
    package: str,
    features: dict[str, pd.DataFrame],
    sampling_frequency: int | dict[str, int],
) -> object:
    """Get the correct feature extractor for the selected package.

    Parameters
    ----------
    package : str
        The name of the package to use for feature extraction.
    features : dict[str, pd.DataFrame]
        The time-series data to extract features from.
    sampling_frequency : int | dict[str, int]
        The sampling frequency of the time-series data.

    Returns
    -------
    object
        The feature extractor object for the selected package.

    Raises
    ------
    ModuleNotFoundError
        If the package is not installed.
    ValueError
        If the package value in the configuration file is invalid.
    """
    if package == "tsfresh":
        try:
            from src.extraction.tsfresh import (
                FeatureExtractor as TSFreshFeatureExtractor,
            )

            feature_extractor = TSFreshFeatureExtractor(features)
        except ImportError as e:
            msg = "tsfresh is not installed."
            raise ModuleNotFoundError(msg) from e

    elif package == "tsfel":
        try:
            from src.extraction.tsfel import FeatureExtractor as TSFELFeatureExtractor

            feature_extractor = TSFELFeatureExtractor(features, sampling_frequency)
        except ImportError as e:
            msg = "TSFEL is not installed."
            raise ModuleNotFoundError(msg) from e

    elif package == "seglearn":
        try:
            from src.extraction.seglearn import (
                FeatureExtractor as SeglearnFeatureExtractor,
            )

            feature_extractor = SeglearnFeatureExtractor(features)
        except ImportError as e:
            msg = "seglearn is not installed."
            raise ModuleNotFoundError(msg) from e

    elif package == "kats":
        try:
            from src.extraction.kats import FeatureExtractor as KatsFeatureExtractor

            feature_extractor = KatsFeatureExtractor(features, sampling_frequency)
        except ImportError as e:
            msg = "KATS is not installed."
            raise ModuleNotFoundError(msg) from e

    elif package == "pycatch22":
        try:
            from src.extraction.catch22 import (
                FeatureExtractor as PyCatch22FeatureExtractor,
            )

            feature_extractor = PyCatch22FeatureExtractor(features, sampling_frequency)
        except ImportError as e:
            msg = "pycatch22 is not installed."
            raise ModuleNotFoundError(msg) from e

    elif package == "tsfeatures":
        try:
            from src.extraction.tsfeatures import (
                FeatureExtractor as TSFeaturesFeatureExtractor,
            )

            feature_extractor = TSFeaturesFeatureExtractor(features, sampling_frequency)
        except ImportError as e:
            msg = "tsfeatures is not installed."
            raise ModuleNotFoundError(msg) from e

    else:
        msg = "Invalid package value in configuration file."
        raise ValueError(msg)

    return feature_extractor

def get_sampling_frequency(dataset: str) -> int | dict[str, int]:
    """Get the correct sampling frequency for the selected dataset."""
    if dataset == "CNC_Mill_Tool_Wear":
        sampling_frequency = 10
    elif dataset == "Condition_Monitoring_of_hydraulic_systems":
        sampling_frequency = {
            "PS1": 100,
            "PS2": 100,
            "PS3": 100,
            "PS4": 100,
            "PS5": 100,
            "PS6": 100,
            "EPS1": 100,
            "FS1": 10,
            "FS2": 10,
            "TS1": 1,
            "TS2": 1,
            "TS3": 1,
            "TS4": 1,
            "VS1": 1,
            "CE": 1,
            "CP": 1,
            "SE": 1,
        }
    elif dataset == "Turning_Dataset_for_Chatter_Diagnosis":
        sampling_frequency = 10000
    elif dataset == "IDMT-ISA_Compressed_Air":
        sampling_frequency = 48000
    elif dataset == "MIMII":
        sampling_frequency = 16000
    elif dataset == "Bosch_CNC":
        sampling_frequency = 2000
    else:
        msg = "Invalid dataset value in configuration file."
        raise ValueError(msg)

    return sampling_frequency
