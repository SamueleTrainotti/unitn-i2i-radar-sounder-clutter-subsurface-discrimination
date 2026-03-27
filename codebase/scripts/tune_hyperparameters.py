import optuna
from pathlib import Path
import json
import datetime
import torch
from torch.utils.data import DataLoader

from core import get_logger, get_config
import scripting
from dataset import MapDataset

def _suggest_hyperparameters(trial: optuna.Trial, tune_config: dict) -> dict:
    """
    Suggests hyperparameters for a trial based on the structured tuning configuration.

    This function parses the `tune_config` dictionary to dynamically generate
    Optuna suggestions for common, model-specific, and selector parameters.

    Args:
        trial: The Optuna trial object.
        tune_config: The dictionary loaded from tune_config.yaml.

    Returns:
        A dictionary containing all suggested hyperparameters for the trial.
    """
    params = {}

    def suggest(p_conf):
        name = p_conf["name"]
        p_type = p_conf["type"]
        if p_type == "float":
            return trial.suggest_float(name, p_conf["low"], p_conf["high"], log=p_conf.get("log", False))
        elif p_type == "categorical":
            return trial.suggest_categorical(name, p_conf["choices"])
        elif p_type == "int":
            return trial.suggest_int(name, p_conf["low"], p_conf["high"], log=p_conf.get("log", False))
        raise ValueError(f"Unsupported parameter type: {p_type}")

    # 1. Suggest common parameters
    for p_conf in tune_config.get("common_params", []):
        params[p_conf["name"]] = suggest(p_conf)

    # 2. Suggest model from the selector
    model_name = suggest(tune_config["model_selector"])
    params[tune_config["model_selector"]["name"]] = model_name

    # 3. Suggest model-specific parameters
    if model_name in tune_config.get("model_specific_params", {}):
        for p_conf in tune_config["model_specific_params"][model_name]:
            params[p_conf["name"]] = suggest(p_conf)

    return params

def objective(trial: optuna.Trial, tune_config: dict, n_epochs: int = None) -> float:
    """
    Objective function for Optuna optimization.

    It creates and trains a model with hyperparameters suggested by the trial,
    evaluates it, and returns the primary metric (SSIM) to be optimized.

    Args:
        trial: An Optuna trial object.
        tune_config: The dictionary loaded from the tuning configuration file.
        n_epochs: Number of epochs to train the model. If None, uses the value from the config.

    Returns:
        The SSIM score of the evaluated model.
    """
    logger = get_logger()
    config = get_config()
    device = config["DEVICE"]

    # --- Log Trial Start ---
    logger.info("=" * 60)
    logger.info(f"STARTING TRIAL {trial.number}")
    logger.info("=" * 60)

    # Suggest hyperparameters from the config file
    hyperparams = _suggest_hyperparameters(trial, tune_config)
    model_name = hyperparams["model_name"]

    with logger.indent():
        logger.info("Suggested Hyperparameters:")
        with logger.indent():
            for key, value in hyperparams.items():
                logger.info(f"{key}: {value}")

        # Initialize the model based on the selected name
        if model_name == "pix2pix":
            from models import Pix2Pix
            model = Pix2Pix()
        elif model_name == "cyclegan":
            from models import CycleGAN
            model = CycleGAN()
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # --- Dynamically update model config from hyperparameters ---
        logger.info("Applying suggested hyperparameters to model config...")
        model.config["GENERATOR_LEARNING_RATE"] = hyperparams["g_lr"]
        model.config["DISCRIMINATOR_LEARNING_RATE"] = hyperparams["d_lr"]
        model.config["BETA1"] = hyperparams["beta1"]
        if model_name == "pix2pix":
            model.config["L1_LAMBDA"] = hyperparams["pix2pix_l1_lambda"]
        elif model_name == "cyclegan":
            model.config["LAMBDA_CYCLE"] = hyperparams["cyclegan_lambda_cycle"]
            model.config["LAMBDA_IDENTITY"] = hyperparams["cyclegan_lambda_identity"]

        # Reconfigure optimizers with the new hyperparameter values
        model.configure_optimizers()

        if n_epochs is not None:
            model.config["NUM_EPOCHS"] = n_epochs

        # --- Training ---
        logger.info(f"Starting training for {model_name}...")
        model.train()
        logger.info("Finished training.")

        # --- Evaluation ---
        # The best metric is now obtained from the early stopping mechanism
        best_metric = model.best_metric if model.early_stopping_enabled else model.history[-1].get(tune_config.get("metric_to_optimize", "val_ssim"))
        if best_metric is None:
            logger.warning("Could not retrieve the best metric for the trial. Returning 0.0")
            best_metric = 0.0

        # --- Log and return ---
        logger.info("Evaluation finished. Final best metric:")
        logger.info(f"{tune_config.get('metric_to_optimize', 'val_ssim')}: {best_metric:.4f}")

    logger.info("-" * 60)
    logger.info(f"TRIAL {trial.number} COMPLETED - Objective Value: {best_metric:.4f}")
    logger.info("-" * 60)

    # Return the value to be optimized
    return best_metric


def tune_hyperparameters(tune_config: dict):
    """
    Perform hyperparameter tuning using Optuna.

    Args:
        tune_config: The dictionary loaded from the tuning configuration file.
    """
    logger = get_logger()
    
    run_settings = tune_config.get("run_settings", {})
    n_trials = run_settings.get("n_trials", 20)
    n_epochs = run_settings.get("n_epochs", None)
    logger.info(
        f"Starting hyperparameter tuning with Optuna for {n_trials} trials..."
    )

    # Silence Optuna's default INFO-level logger to avoid redundant messages
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Create a study object and optimize the objective function
    study = optuna.create_study(
        direction=run_settings.get("direction", "maximize"),  # We want to maximize SSIM
        sampler=optuna.samplers.TPESampler(),
    )  # Using TPE sampler
    study.optimize(lambda trial: objective(trial, tune_config, n_epochs), n_trials=n_trials)

    logger.info("Hyperparameter tuning finished.")

    # Get and log the best trial
    best_trial = study.best_trial
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Value (SSIM): {best_trial.value:.4f}")
    logger.info("Params:")
    with logger.indent():
        for key, value in best_trial.params.items():
            logger.info(f"{key}: {value}")

    # Save the best hyperparameters to a file
    output_dir = Path(get_config()["OUTPUT_DATA"]["BASE_DIR"]) / "hyperparameter_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"best_hyperparameters_{timestamp}.json", "w") as f:
        json.dump(best_trial.params, f, indent=2)

    logger.info(
        f"Best hyperparameters saved to {output_dir / f'best_hyperparameters_{timestamp}.json'}"
    )
    
def main(tune_config: dict):
    """Main function to run the hyperparameter tuning."""
    tune_hyperparameters(tune_config)

if __name__ == "__main__":
    scripting.logged_main(
        "Hyperparameter Tuning",
        main,
    )
