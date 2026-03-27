import os
import shutil
import yaml
from datetime import datetime
from core.config import get_config, set_config

def setup_run_environment(run_topic: str = None, run_name: str = None):
    """
    Sets up the directory structure for a new training run.
    - Creates a unique directory for the run.
    - Saves the configuration file to that directory.
    - Updates the global configuration with the new, absolute paths.
    
    Args:
        run_topic (str, optional): Subfolder for organizing runs (e.g. "training", "benchmark").
        run_name (str, optional): Custom name for the run directory.
    """
    config = get_config()
    
    base_dir = config["OUTPUT_DATA"]["BASE_DIR"]
    run_to_resume = config.get("RUN_TO_RESUME")

    if run_to_resume:
        # If resuming a run, use the provided run_id
        run_id = run_to_resume
        run_dir = os.path.join(base_dir, run_id)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory to resume not found: {run_dir}")
    else:
        # Otherwise, create a new run directory
        if run_name:
            run_id = run_name
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_name = config.get("MODEL_NAME", "model")
            run_id = f"{timestamp}_{model_name}"
        
        # If a topic is provided, create the run directory inside that topic folder
        if run_topic:
            run_dir = os.path.join(base_dir, run_topic, run_id)
        else:
            run_dir = os.path.join(base_dir, run_id)
            
        os.makedirs(run_dir, exist_ok=True)
    
    # 3. Update config with the new run directory
    config["RUN_DIR"] = run_dir
    
    # Define default output directories relative to the run directory
    default_dirs = {
        "LOGS": "logs",
        "TEMP": "temp",
        "EVALUATION": "evaluation",
        "CHECKPOINTS": "checkpoints",
        "TEMP_DATASET": "temp/dataset"
    }

    output_data_config = config.get("OUTPUT_DATA", {})

    # Set default paths if they are not provided in the config
    for key, value in default_dirs.items():
        if key not in output_data_config:
            output_data_config[key] = value

    config["OUTPUT_DATA"] = output_data_config

    # 4. Customize structure based on topic
    dirs_to_create = {"LOGS"} # Always create logs
    
    if run_topic in ["benchmark", "test", "anomaly_detection"]:
        # Flatten evaluation: put results directly in run_dir
        # We override the value from config to be "." (current dir)
        output_data_config["EVALUATION"] = "."
        
        # Only create temporary folders for data loading
        dirs_to_create.update({"TEMP", "TEMP_DATASET"})
        # Do NOT add CHECKPOINTS or explicit EVALUATION folder (since it is root)
    else:
        # Default / Training: Create everything
        dirs_to_create.update({"TEMP", "TEMP_DATASET", "EVALUATION", "CHECKPOINTS"})

    # 5. Create subdirectories and resolve paths
    for key, value in output_data_config.items():
        if key == "BASE_DIR":
            continue
            
        # Create the full, absolute path
        abs_path = os.path.join(run_dir, value)
        
        # Only create the directory if it is in our allow-list for this topic
        # OR if the key seems to be a custom one not in our standard list, 
        # we might default to creating it? For safety, let's stick to the list 
        # + "EVALUATION" if it was strictly defined in dirs_to_create.
        
        # Special case: If path is just ".", abs_path is run_dir, which exists.
        # We use strict checking against dirs_to_create.
        if key in dirs_to_create:
            os.makedirs(abs_path, exist_ok=True)
            
        # Update the config with the absolute path
        output_data_config[key] = abs_path
            
    # 5. Save a copy of the modified config file
    config_copy_path = os.path.join(run_dir, "config.yaml")
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    # 6. Update the global config
    set_config(config)
    
    # The logger needs to be re-initialized to use the new log path.
    # This will be handled in the main script.
    
    return config
