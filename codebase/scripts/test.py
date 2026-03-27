from core import get_logger, get_config, Evaluator
import torch
from torch.utils.data import DataLoader # type: ignore
from dataset import MapDataset
from models import CycleGAN, Pix2Pix
import scripting

def test(model=None):
    """
    Main testing logic. Loads a trained generator and evaluates on the validation set.

    - Loads generator and weights
    - Loads validation dataset
    - Saves visual + statistical outputs for a batch of examples
    Args:
        config (dict): Configuration dictionary containing model and dataset parameters.
        model (Model, optional): Pre-initialized model instance. If None, it will be created based on the config.
    Raises:
        ValueError: If the model is not provided and the config does not specify a valid model name.
    """
    # Load configuration
    config = get_config()
    # Get the logger
    logger = get_logger()
    
    logger.info(f"{'=' * 10} Starting testing process {'=' * 10}")
    
    if model is None:
        if config["MODEL_NAME"] == "pix2pix":
            model = Pix2Pix(training=False)
        elif config["MODEL_NAME"] == "cyclegan":
            model = CycleGAN(training=False)
        else:
            raise ValueError("Model must be provided or specified in the config.")
    
    with logger.indent():
        # Load the given model parameters with latest checkpoint
        logger.info(f"Loading model: {model.name}")
        model.load()
        
    # Load validation dataset
    val_dataset = MapDataset(
        root_dir=config["INPUT_DATA"]["VAL"],
        temp_dir=config["OUTPUT_DATA"]["TEMP"],
        temp_dataset=config["OUTPUT_DATA"]["TEMP_DATASET"],
        SAVE_DATASET=config.get("SAVE_DATASET", False),
        LOAD_DATASET=config.get("LOAD_DATASET", False),
        normalization_config=config.get("NORMALIZATION")
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Handle model-specific generator retrieval and data loading
    with logger.indent():
        evaluator = Evaluator(config["DEVICE"])
        if isinstance(model, CycleGAN):
            generator_to_test = config.get("GENERATOR_TO_TEST", "G_AB") # Default to G_AB
            logger.info(f"Testing CycleGAN generator: {generator_to_test}")
            gen = model.get_generator(generator_to_test)

            # If testing G_BA (real -> sim), we need to feed the 'real' image (y) as input.
            # The MapDataset returns (sim, real), so we swap them.
            if generator_to_test == "G_BA":
                logger.debug("Re-ordering dataset for G_BA (real -> sim) testing.")
                # The save_some_examples function expects (input, target)
                # For G_BA, input is 'real' (y) and target is 'sim' (x)
                val_loader = ((y, x) for x, y in val_loader)
        else:
            # For other models like Pix2Pix, assume a single generator
            logger.info(f"Testing default generator for model: {model.name}")
            gen = model.get_generator()
        
        metrics = evaluator.evaluate_model(gen, val_loader)
        logger.info(f"Evaluation metrics: {metrics}")

if __name__ == "__main__":
    scripting.logged_main(
        "Test a trained model and save examples",
        test,
        run_topic="test"
    )
