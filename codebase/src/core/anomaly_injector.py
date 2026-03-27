import torch
import random
import numpy as np

class RealisticAnomalyInjector:
    def __init__(self, config):
        self.config = config
        # Attenuation range relative to surface (normalized value or dB)
        # If data is normalized [-1, 1], attenuation of 0.2-0.5 is significant.
        self.min_attenuation = 0.2 
        self.max_attenuation = 0.6 
        
    def inject_dipping_layer(self, image_tensor):
        """
        Injects a realistic dipping layer (inclined subsurface reflector).
        Args:
            image_tensor: Tensor [C, H, W] (Single image, usually C=1)
        Returns:
            injected_image: Tensor with anomaly
            mask: Binary mask of the anomaly
        """
        C, H, W = image_tensor.shape
        device = image_tensor.device
        
        # 1. Find surface (max value for each column)
        # Assuming the highest peak is the surface
        surface_vals, surface_idxs = torch.max(image_tensor[0], dim=0) # [W]
        
        # 2. Random parameters for the layer
        # Start at a random depth below surface (e.g. 20-50 pixels below)
        start_col = random.randint(0, W // 4)
        end_col = random.randint(3 * W // 4, W - 1)
        
        # Initial depth relative to surface at that point
        depth_offset_start = random.randint(30, 80) 
        # Slope: end is at a different depth
        slope = random.uniform(-0.5, 0.5) 
        
        # Layer thickness (in pixels)
        thickness = random.randint(2, 5)
        
        mask = torch.zeros((H, W), device=device)
        anomaly_layer = torch.zeros((H, W), device=device)
        
        # 3. Geometry Generation
        for x in range(start_col, end_col):
            # Calculate y based on line equation
            rel_x = x - start_col
            depth_offset = depth_offset_start + (slope * rel_x)
            
            # Absolute y is: surface position + depth offset
            surf_y = surface_idxs[x].item()
            target_y = int(surf_y + depth_offset)
            
            if 0 <= target_y < H:
                # Draw thickness
                y_min = max(0, target_y - thickness // 2)
                y_max = min(H, target_y + thickness // 2 + 1)
                mask[y_min:y_max, x] = 1.0
                
                # 4. Physical Intensity Calculation
                # Intensity = Surface Intensity - Attenuation
                attenuation = random.uniform(self.min_attenuation, self.max_attenuation)
                base_intensity = surface_vals[x].item() - attenuation
                
                # Add Texture (Speckle Noise)
                # The anomaly is not solid, varies pixel by pixel
                noise = torch.randn((y_max - y_min), device=device) * 0.1
                col_intensity = base_intensity + noise
                
                anomaly_layer[y_min:y_max, x] = col_intensity

        # 5. Merge with original image
        # Use 'max' to simulate radar reflection (strongest signal wins)
        # Or sum for constructive interference, but max is visually safer here
        injected_image = torch.max(image_tensor[0], anomaly_layer)
        
        # Back to [C, H, W]
        injected_image = injected_image.unsqueeze(0)
        
        return injected_image, mask.unsqueeze(0).unsqueeze(0)

    def forward(self, batch_images):
        """Applies injection to a batch"""
        injected_batch = []
        masks_batch = []
        
        for img in batch_images:
            # Coin flip: inject or not? (For ROC calculation need balanced dataset or all injected)
            # For now inject on all to calculate Sensitivity
            inj_img, mask = self.inject_dipping_layer(img)
            injected_batch.append(inj_img)
            masks_batch.append(mask)
            
        return torch.stack(injected_batch), torch.stack(masks_batch)
