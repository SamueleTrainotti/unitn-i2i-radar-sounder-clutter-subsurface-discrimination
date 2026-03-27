from core.config import get_config
import scripting

def train(model=None):
    """
    Main training logic.

    - Loads training and validation datasets.
    - Initializes and optionally loads a model.
    - Calls the model's internal train() method.

    Args:
        config (dict): Configuration with model and training parameters.
        model (Model, optional): Pre-initialized model instance. If None, one will be created based on config.

    Raises:
        ValueError: If model is not specified and MODEL_NAME is unknown.
    """
    # Load configuration
    config = get_config()
    
    if model is None:
        if config["MODEL_NAME"] == "pix2pix":
            from models import Pix2Pix
            model = Pix2Pix()
        elif config["MODEL_NAME"] == "cyclegan":
            from models import CycleGAN
            model = CycleGAN()
        else:
            raise ValueError("Model must be provided or specified in the config.")
    
    # Train the model using its internal training logic
    model.train()

if __name__ == "__main__":
    import torch
    torch.backends.cudnn.benchmark = True
    scripting.logged_main(
        "Train a GAN model (Pix2Pix or CycleGAN)",
        train,
        run_topic="training"
    )
