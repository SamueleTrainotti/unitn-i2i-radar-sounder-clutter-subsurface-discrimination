from models import Model
from models.architectures import UnetGenerator, PatchGAN
from core import get_logger
from utils import count_parameters, get_gradient_norm
import torch # type: ignore

class Pix2Pix(Model):
    def __init__(self, training: bool = True):
        super().__init__(name="pix2pix", training=training)
        # Efficiently instantiate loss functions once
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.L1 = torch.nn.L1Loss()

    def build_models(self):
        self.logger.info("Building Pix2Pix generator and discriminator models")
        patch_size = self.config.get('PATCH_SIZE', 256)
        normalization_type = self.config.get("NORMALIZATION_TYPE", "range_zero_to_one")
        receptive_field = self.config.get("DISCRIMINATOR_RECEPTIVE_FIELD", "70x70")
        self.generator = UnetGenerator(
            in_channels=1, 
            patch_size=patch_size, 
            normalization_type=normalization_type
        ).to(self.device)
        self.discriminator = PatchGAN(in_channels=2, receptive_field=receptive_field).to(self.device)
        self.logger.info(f"Generator parameters: {count_parameters(self.generator):,}")
        self.logger.info(f"Discriminator parameters: {count_parameters(self.discriminator):,}")
        # CRITICAL FIX: Register models for saving/loading
        self.models['generator'] = self.generator
        self.models['discriminator'] = self.discriminator
    
    def configure_optimizers(self):
        self.logger.info("Configuring Pix2Pix optimizers")
        g_lr = self.config.get('GENERATOR_LEARNING_RATE')
        d_lr = self.config.get('DISCRIMINATOR_LEARNING_RATE', self.config['GENERATOR_LEARNING_RATE'])
        betas = (0.5, 0.999)
        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=g_lr, betas=betas)
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=d_lr, betas=betas)
        # CRITICAL FIX: Register optimizers for saving/loading
        self.optimizers['opt_G'] = self.opt_G
        self.optimizers['opt_D'] = self.opt_D

    @property
    def latest_checkpoint(self):
        """
        Property to get the path of the latest checkpoint.
        Returns:
            str: Path to the latest checkpoint file.
        """
        return f"{self.config['OUTPUT_DATA']['CHECKPOINTS']}/{self.config['CHECKPOINT_P2P_LATEST']}"
    
    def training_step(self, data, scaler: torch.cuda.amp.GradScaler):
        x, y = data
        
        x = x.unsqueeze(1).to(self.device)
        y = y.unsqueeze(1).to(self.device)

        # Discriminator training
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp_enabled):
            fake_y = self.generator(x)
            D_real = self.discriminator(torch.cat([x, y], dim=1))
            D_fake = self.discriminator(torch.cat([x, fake_y.detach()], dim=1))

            label_smoothing = self.config.get('LABEL_SMOOTHING', 0.0)
            real_label = 1.0 - label_smoothing
            D_real_loss = self.BCE(D_real, torch.full_like(D_real, real_label, device=self.device))
            D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        self.opt_D.zero_grad()
        scaler.scale(D_loss).backward()
        
        # Log gradient norms
        if self.config.get("LOG_GRADIENTS", True):
             D_grad_norm = get_gradient_norm(self.discriminator)
        else:
             D_grad_norm = 0.0
             
        scaler.step(self.opt_D)
        scaler.update()

        # Generator training
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp_enabled):
            D_fake = self.discriminator(torch.cat([x, fake_y], dim=1))
            G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
            L1_loss = self.L1(fake_y, y) * self.config['L1_LAMBDA']
            G_loss = G_fake_loss + L1_loss

        self.opt_G.zero_grad()
        scaler.scale(G_loss).backward()
        
        # Log gradient norms
        if self.config.get("LOG_GRADIENTS", True):
             G_grad_norm = get_gradient_norm(self.generator)
        else:
             G_grad_norm = 0.0

        scaler.step(self.opt_G)
        scaler.update()

        metrics = {
            'D_loss': D_loss.item(),
            'G_loss': G_loss.item(),
            'G_fake_loss': G_fake_loss.item(),
            'L1_loss': L1_loss.item(),
            'G_fake_loss_prop': (G_fake_loss / G_loss).item() if G_loss > 0 else 0,
            'L1_loss_prop': (L1_loss / G_loss).item() if G_loss > 0 else 0,
            'D_real': torch.sigmoid(D_real).mean().item(),
            'D_fake': torch.sigmoid(D_fake).mean().item(),
            'D_grad_norm': D_grad_norm,
            'G_grad_norm': G_grad_norm
        }
        self.logger.debug(f"Pix2Pix training step metrics: {metrics}")
        # Return metrics for logging
        return metrics

    def get_generator(self):
        """
        Returns the generator model.
        Returns:
            Generator: The generator model instance.
        """
        return self.generator
