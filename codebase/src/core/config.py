import yaml
import re
from pathlib import Path

_config = None

def validate_config(config: dict) -> None:
    """
    Validates the structure and values of the configuration dictionary.
    Raises ValueError if the configuration is invalid.
    """
    if 'ANOMALY_DETECTION' in config:
        ad_config = config['ANOMALY_DETECTION']
        if not isinstance(ad_config, dict):
            raise ValueError("ANOMALY_DETECTION section must be a dictionary.")

        # Validate METHOD
        if 'METHOD' in ad_config:
            method = ad_config['METHOD']
            allowed_methods = ['pixel-level', 'whole-image', 'patch-based', 'sliding-window']
            if method not in allowed_methods:
                raise ValueError(f"Invalid ANOMALY_DETECTION.METHOD: '{method}'. Allowed values are {allowed_methods}.")

        # Validate METRICS
        if 'METRICS' in ad_config:
            metrics = ad_config['METRICS']
            if not isinstance(metrics, list):
                raise ValueError("ANOMALY_DETECTION.METRICS must be a list.")
            
            allowed_metrics = ['L1', 'L2', 'SSIM', 'LPIPS']
            for metric in metrics:
                if metric not in allowed_metrics:
                    raise ValueError(f"Invalid metric in ANOMALY_DETECTION.METRICS: '{metric}'. Allowed values are {allowed_metrics}.")

        # Validate BACKGROUND_REMOVAL
        if 'BACKGROUND_REMOVAL' in ad_config:
            br_config = ad_config['BACKGROUND_REMOVAL']
            if not isinstance(br_config, dict):
                raise ValueError("ANOMALY_DETECTION.BACKGROUND_REMOVAL must be a dictionary.")

            if 'METHOD' in br_config:
                br_method = br_config['METHOD']
                allowed_br_methods = ['none', 'pca', 'avg_trace']
                if br_method not in allowed_br_methods:
                    raise ValueError(f"Invalid ANOMALY_DETECTION.BACKGROUND_REMOVAL.METHOD: '{br_method}'. Allowed values are {allowed_br_methods}.")

                if br_method == 'pca':
                    if 'PCA_COMPONENTS' not in br_config:
                        raise ValueError("ANOMALY_DETECTION.BACKGROUND_REMOVAL.PCA_COMPONENTS must be specified for the 'pca' method.")
                    pca_components = br_config['PCA_COMPONENTS']
                    if not isinstance(pca_components, int) or pca_components <= 0:
                        raise ValueError("ANOMALY_DETECTION.BACKGROUND_REMOVAL.PCA_COMPONENTS must be a positive integer.")
        
        # Validate Z_SCORE_NORMALIZATION
        if 'Z_SCORE_NORMALIZATION' in ad_config:
            zsn_config = ad_config['Z_SCORE_NORMALIZATION']
            if not isinstance(zsn_config, dict):
                raise ValueError("ANOMALY_DETECTION.Z_SCORE_NORMALIZATION must be a dictionary.")
            if 'ENABLED' not in zsn_config:
                raise ValueError("ANOMALY_DETECTION.Z_SCORE_NORMALIZATION must have an 'ENABLED' key.")
            if not isinstance(zsn_config['ENABLED'], bool):
                raise ValueError("ANOMALY_DETECTION.Z_SCORE_NORMALIZATION.ENABLED must be a boolean.")


def set_config(config: dict) -> None:
    global _config
    validate_config(config)
    _config = config

def get_config() -> dict:
    if _config is None:
        raise RuntimeError("Configuration has not been set yet.")
    return _config

def load_config(path: str) -> dict:
    """Loads and resolves a YAML configuration file."""
    with open(path, "r") as f:
        config = yaml.full_load(f)
    return resolve_config_references(config)

def resolve_config_references(config: dict, root: dict | None = None) -> dict:
    """Recursively resolve ${...} references in the config."""
    if root is None:
        root = config

    pattern = re.compile(r"\$\{([^\}]+)\}")

    def resolve_value(value):
        if isinstance(value, str):
            while True:
                match = pattern.search(value)
                if not match:
                    break
                path = match.group(1).split(".")
                ref = root
                for key in path:
                    ref = ref.get(key)
                    if ref is None:
                        raise KeyError(f"Cannot resolve reference: {match.group(1)}")
                value = value.replace(match.group(0), str(ref))
            return value
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(v) for v in value]
        else:
            return value

    return {k: resolve_value(v) for k, v in config.items()}
