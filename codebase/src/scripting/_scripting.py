import argparse
import datetime
import logging
import pathlib
import shutil
import sys
import typing

import yaml

from core import set_logger, set_config, load_config
from core.logger import TaskLoggerAdapter, TqdmHandler


def is_valid_file(path: str) -> str:
    """Identity function (i.e., the input is passed through and is not modified)
    that checks whether the given path is a valid file or not, raising an
    argparse.ArgumentTypeError if not valid.

    Parameters
    ----------
    path : str
        String representing a path to a file.

    Returns
    -------
    str
        The same string as in input

    Raises
    ------
    argparse.ArgumentTypeError
        An exception is raised if the given string does not represent a valid
        path to an existing file.
    """
    file = pathlib.Path(path)
    if not file.is_file:
        raise argparse.ArgumentTypeError(f'{path} does not exist')
    return path

def get_parser(description: str) -> argparse.ArgumentParser:
    """Function that generates the argument parser for the processor. Here, all
    the arguments and help message are defined.

    Parameters
    ----------
    description : str
        Text description to include in the help message of the script.

    Returns
    -------
    argparse.ArgumentParser
        The created argument parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--conf",
        "-c",
        dest="config_file_path",
        required=True,
        metavar="FILE",
        type=lambda x: is_valid_file(x),  # type: ignore
        help="The YAML configuration file.",
    )
    parser.add_argument(
        "--tune-conf",
        "-t",
        dest="tune_config_file_path",
        required=False,
        metavar="FILE",
        type=lambda x: is_valid_file(x),  # type: ignore
        help="The YAML configuration file for hyperparameter tuning. "
        "Only used by tuning scripts.",
    )
    parser.add_argument(
        "--run-ids",
        "-r",
        dest="run_ids",
        required=False,
        nargs="+",
        help="The IDs of the runs to be used.",
    )
    return parser

def clear_folder(folder_path: str, preserve_subfolders: bool = False):
    """
    Ensures the folder exists and clears all contents inside it.
    
    Args:
        folder_path (str): Path to the folder.
    """
    path = pathlib.Path(folder_path)

    # Check if exists
    if path.exists():
        # Clear all contents
        for item in path.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    if not preserve_subfolders:
                        # If preserve_subfolders is False, delete the directory and its contents
                        shutil.rmtree(item)
                    else:
                        # If preserve_subfolders is True, just delete the contents of the directory
                        clear_folder(item.as_posix(), preserve_subfolders=True)
            except Exception as e:
                logging.error(f'Failed to delete {item}. Reason: {e}')
    else:
        raise FileNotFoundError(f'Folder {folder_path} does not exist.')

    logging.debug(f'Cleared contents of folder {path}')
    
def ensure_folder_exists(folder_path: str | dict) -> None:
    """
    Ensures that the folder exists, creating it if it does not.
    If a dictionary is provided, recursively ensures all values (folders) exist.

    Args:
        folder_path (str or dict): Path to the folder, or a dictionary of paths.
    """
    if isinstance(folder_path, dict):
        logging.debug(f'Ensuring folders exist: {folder_path}')
        for value in folder_path.values():
            ensure_folder_exists(value)
    elif isinstance(folder_path, str):
        path = pathlib.Path(folder_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logging.debug(f'Created folder: {folder_path}')
        else:
            logging.debug(f'Folder already exists: {folder_path}')
    else:
        logging.error(f'Invalid type for folder_path: {type(folder_path)}')
        raise TypeError("folder_path must be a string or a dictionary of strings.")


from core import set_logger, set_config, load_config, setup_run_environment

def logged_main(description: str, main_fn: typing.Callable, setup_run: bool = True, skip_run_setup: bool = False, add_args_fn: typing.Callable[[argparse.ArgumentParser], None] = None, run_topic: str = None, **kwargs) -> None:
    print(f"DEBUG: logged_main started. description={description}")
    start_time = datetime.datetime.now()

    # ---- Parsing
    # ---- Parsing
    parser = get_parser(description)
    if add_args_fn:
        add_args_fn(parser)
        
    args = parser.parse_args()
    
    # Extract unknown args to pass to main_fn if needed (not standard argparse behavior but useful)
    # Actually, let's just inspect args and pass extra known ones to kwargs
    for arg_name, arg_value in vars(args).items():
        if arg_name not in ["config_file_path", "tune_config_file_path", "run_ids"]:
            kwargs[arg_name] = arg_value

    # ---- Loading configuration file
    config_file = args.config_file_path
    config = load_config(config_file)
    set_config(config)

    # ---- Setup Run Environment (or skip it)
    if not skip_run_setup:
        run_name = None
        if args.run_ids and len(args.run_ids) == 1:
            # If we have a single target run, we can name this execution accordingly
            if run_topic == "anomaly_detection":
                run_name = f"anomaly_eval_{args.run_ids[0]}"
            elif run_topic == "benchmark":
                run_name = f"benchmark_{args.run_ids[0]}"
            elif run_topic == "test":
                run_name = f"test_{args.run_ids[0]}"

        if not setup_run:
            # This is a special case for scripts like anomaly detection that need to
            # merge a training config before the run environment is built.
            if 'MODEL_PATH' in config:
                model_path = config['MODEL_PATH']
                training_config_path = pathlib.Path(model_path).parent / "config.yaml"
                
                if training_config_path.exists():
                    with open(training_config_path, 'r') as f:
                        training_config = yaml.safe_load(f)
                    
                    # Merge configs: detection-specific values override training values
                    merged_config = training_config.copy()
                    # custom deep merge for nested dictionaries
                    for key, value in config.items():
                        if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                            merged_config[key].update(value)
                        else:
                            merged_config[key] = value

                    config = merged_config
                    set_config(config) # Update the global config
                else:
                    raise FileNotFoundError(f"Training config not found at {training_config_path}")
        
        # Now that the config is finalized (either standard or merged), setup the run environment
        config = setup_run_environment(run_topic=run_topic, run_name=run_name)
        
        if "LEARNING_RATE" in config:
            config["LEARNING_RATE"] = float(config["LEARNING_RATE"])
        if "GENERATOR_LEARNING_RATE" in config:
            config["GENERATOR_LEARNING_RATE"] = float(config["GENERATOR_LEARNING_RATE"])
        if "DISCRIMINATOR_LEARNING_RATE" in config:
            config["DISCRIMINATOR_LEARNING_RATE"] = float(config["DISCRIMINATOR_LEARNING_RATE"])

        # ---- Ensure input data folders exist
        ensure_folder_exists(config["INPUT_DATA"])
    
    # ---- Load tuning configuration file if provided (works for both modes)
    if args.tune_config_file_path:
        with open(args.tune_config_file_path) as yaml_file:
            tune_config = yaml.full_load(yaml_file)
        # Pass it to the main function
        kwargs["tune_config"] = tune_config

    # ---- Pass run_ids if provided (works for both modes)
    if args.run_ids:
        kwargs["run_ids"] = args.run_ids
    
    # ---- Logging
    simple_logger = logging.getLogger()
    simple_logger.setLevel(logging.NOTSET)
    log_level = config.get("log_level", "info") # Default to info if not present

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    # Always add a console handler
    if sys.stdout.isatty():
        console_handler = TqdmHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        frmttr_console = logging.Formatter(
            "%(asctime)s [%(name)s.%(filename)s-%(levelname)s]: %(message)s"
        )
        console_handler.setFormatter(frmttr_console)
        simple_logger.addHandler(console_handler)

    task_logger = TaskLoggerAdapter(simple_logger)

    # Add a file handler only if we are doing a standard run setup
    if not skip_run_setup:
        log_filename = (
            pathlib.Path(config["OUTPUT_DATA"]["LOGS"]) / f'{main_fn.__name__}_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.log'
        )
        logfile_handler = logging.FileHandler(filename=log_filename)
        logfile_handler.setLevel(logging.DEBUG)
        frmttr_logfile = logging.Formatter(
            "%(asctime)s [%(name)s.%(filename)s-%(levelname)s]: %(message)s"
        )
        logfile_handler.setFormatter(frmttr_logfile)
        simple_logger.addHandler(logfile_handler)
        
        if log_level.lower() != "info":
            task_logger.disable_pbar = True

        max_key_length = max(len(str(key)) for key in config.keys())
        aligned_output = ""
        for key, value in config.items():
            aligned_output += f'{key:<{max_key_length}} : {value}\n'

        task_logger.info("Configuration Complete.")
        task_logger.info("YAML Config File:\n%s", aligned_output)
    
    task_logger.info(f'Start execution at: {start_time}')
    
    set_logger(task_logger)
    
    try:
        main_fn(**kwargs)
    finally:
        task_logger.info(f'Close all. Execution time: {datetime.datetime.now()-start_time}')
        simple_logger.handlers.clear()
        logging.shutdown()
