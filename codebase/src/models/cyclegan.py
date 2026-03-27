from models import Model
from models.architectures import ResnetGenerator, PatchGAN
from core import get_logger
from utils import count_parameters, get_gradient_norm
import torch # type: ignore

class CycleGAN(Model):
    def __init__(self, training: bool = True):
        super().__init__(name="cyclegan", training=training)
        # Efficiently instantiate loss functions once
        self.adv_loss = torch.nn.MSELoss()  # Using LSGAN
        self.cycle_loss_fn = torch.nn.L1Loss()
        self.identity_loss_fn = torch.nn.L1Loss()

    def build_models(self):
        self.logger.info("Building CycleGAN generators and discriminators")
        normalization_type = self.config.get("NORMALIZATION_TYPE", "range_zero_to_one")
        receptive_field = self.config.get("DISCRIMINATOR_RECEPTIVE_FIELD", "70x70")
        self.G_AB = ResnetGenerator(normalization_type=normalization_type).to(self.device)
        self.G_BA = ResnetGenerator(normalization_type=normalization_type).to(self.device)
        self.D_A = PatchGAN(in_channels=1, receptive_field=receptive_field).to(self.device)
        self.D_B = PatchGAN(in_channels=1, receptive_field=receptive_field).to(self.device)
        self.logger.info(f"Generator (G_AB/G_BA) parameters: {count_parameters(self.G_AB):,}")
        self.logger.info(f"Discriminator (D_A/D_B) parameters: {count_parameters(self.D_A):,}")
        # CRITICAL FIX: Register models for saving/loading
        self.models['G_AB'] = self.G_AB
        self.models['G_BA'] = self.G_BA
        self.models['D_A'] = self.D_A
        self.models['D_B'] = self.D_B

    def configure_optimizers(self):
        self.logger.info("Configuring CycleGAN optimizers")
        g_lr = self.config['GENERATOR_LEARNING_RATE']
        d_lr = self.config['DISCRIMINATOR_LEARNING_RATE']
        betas = (0.5, 0.999)
        self.opt_G = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()), lr=g_lr, betas=betas)
        self.opt_D_A = torch.optim.Adam(self.D_A.parameters(), lr=d_lr, betas=betas)
        self.opt_D_B = torch.optim.Adam(self.D_B.parameters(), lr=d_lr, betas=betas)
        # CRITICAL FIX: Register optimizers for saving/loading
        self.optimizers['opt_G'] = self.opt_G
        self.optimizers['opt_D_A'] = self.opt_D_A
        self.optimizers['opt_D_B'] = self.opt_D_B
        
    @property
    def latest_checkpoint(self):
        """
        Property to get the path of the latest checkpoint.
        Returns:
            str: Path to the latest checkpoint file.
        """
        return f"{self.config['OUTPUT_DATA']['CHECKPOINTS']}/{self.config['CHECKPOINT_CG_LATEST']}"

    def training_step(self, data, scaler: torch.cuda.amp.GradScaler):
        # real_a is simulated radargramm
        # real_b is real radargramm
        real_A, real_B = data
        real_A = real_A.unsqueeze(1).to(self.device)
        real_B = real_B.unsqueeze(1).to(self.device)

        # === Train Discriminators ===
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp_enabled):
            fake_B = self.G_AB(real_A).detach()
            fake_A = self.G_BA(real_B).detach()

            label_smoothing = self.config.get('LABEL_SMOOTHING', 0.0)
            real_label = 1.0 - label_smoothing

            # D_A
            D_real_A = self.D_A(real_A)
            D_fake_A = self.D_A(fake_A)
            D_A_real_loss = self.adv_loss(D_real_A, torch.full_like(D_real_A, real_label, device=self.device))
            D_A_fake_loss = self.adv_loss(D_fake_A, torch.zeros_like(D_fake_A))
            D_A_loss = (D_A_real_loss + D_A_fake_loss) * 0.5

        self.opt_D_A.zero_grad()
        scaler.scale(D_A_loss).backward()
        
        if self.config.get("LOG_GRADIENTS", True):
             D_A_grad_norm = get_gradient_norm(self.D_A)
        else:
             D_A_grad_norm = 0.0
             
        scaler.step(self.opt_D_A)
        scaler.update()

        # D_B
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp_enabled):
            D_real_B = self.D_B(real_B)
            D_fake_B = self.D_B(fake_B)
            D_B_real_loss = self.adv_loss(D_real_B, torch.full_like(D_real_B, real_label, device=self.device))
            D_B_fake_loss = self.adv_loss(D_fake_B, torch.zeros_like(D_fake_B))
            D_B_loss = (D_B_real_loss + D_B_fake_loss) * 0.5

        self.opt_D_B.zero_grad()
        scaler.scale(D_B_loss).backward()
        
        if self.config.get("LOG_GRADIENTS", True):
             D_B_grad_norm = get_gradient_norm(self.D_B)
        else:
             D_B_grad_norm = 0.0

        scaler.step(self.opt_D_B)
        scaler.update()

        # === Train Generators ===
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp_enabled):
            # Adversarial losses
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)
            D_fake_B = self.D_B(fake_B)
            D_fake_A = self.D_A(fake_A)
            G_AB_loss = self.adv_loss(D_fake_B, torch.ones_like(D_fake_B))
            G_BA_loss = self.adv_loss(D_fake_A, torch.ones_like(D_fake_A))

            # Cycle losses
            rec_A = self.G_BA(fake_B)
            rec_B = self.G_AB(fake_A)
            cycle_A_loss = self.cycle_loss_fn(rec_A, real_A)
            cycle_B_loss = self.cycle_loss_fn(rec_B, real_B)
            cycle_loss = cycle_A_loss + cycle_B_loss

            # Identity losses
            idt_A = self.G_BA(real_A)
            idt_B = self.G_AB(real_B)
            idt_A_loss = self.identity_loss_fn(idt_A, real_A)
            idt_B_loss = self.identity_loss_fn(idt_B, real_B)
            identity_loss = idt_A_loss + idt_B_loss

            lambda_cycle = self.config['LAMBDA_CYCLE']
            lambda_id = self.config.get('LAMBDA_IDENTITY', 0.0)

            G_loss = (
                G_AB_loss + G_BA_loss +
                lambda_cycle * cycle_loss +
                lambda_id * identity_loss
            )

        self.opt_G.zero_grad()
        scaler.scale(G_loss).backward()
        
        if self.config.get("LOG_GRADIENTS", True):
             # Calculate combined norm for both generators as they share an optimizer
             total_norm = 0.0
             for model in [self.G_AB, self.G_BA]:
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
             G_grad_norm = total_norm ** 0.5
        else:
             G_grad_norm = 0.0
             
        scaler.step(self.opt_G)
        scaler.update()

        metrics = {
            'D_A_loss': D_A_loss.item(),
            'D_B_loss': D_B_loss.item(),
            'G_loss': G_loss.item(),
            'G_AB_loss': G_AB_loss.item(), # Logged individually
            'G_BA_loss': G_BA_loss.item(), # Logged individually
            'cycle_loss': cycle_loss.item(),
            'identity_loss': identity_loss.item(),
            'D_A_grad_norm': D_A_grad_norm,
            'D_B_grad_norm': D_B_grad_norm,
            'G_grad_norm': G_grad_norm
        }
        self.logger.debug(f"CycleGAN training step metrics: {metrics}")
        return metrics
        
    def get_generator(self, name="G_AB"):
        """
        Returns a specified generator model.
        Args:
            name (str): The name of the generator to return ("G_AB" or "G_BA").
        Returns:
            The requested generator model.
        """
        if name == "G_AB":
            return self.G_AB
        elif name == "G_BA":
            return self.G_BA
        raise ValueError(f"Unknown generator name: {name}")