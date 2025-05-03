import os
import torch
import logging
import yaml
import traceback
import numpy as np
import json
import math
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from transformers import T5EncoderModel, T5Tokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Subset
from PIL import Image
import gc
from sklearn.model_selection import KFold
from itertools import product
try:
    from bitsandbytes.optim import AdamW8bit
except ImportError:
    print("Warning: bitsandbytes not found. Using torch.optim.AdamW.")
    AdamW8bit = torch.optim.AdamW
try:
    from src.utils.dataset import load_dataset, CocoFinetuneDataset
except ImportError:
    print("Warning: Could not import dataset utilities.")
    class CocoFinetuneDataset(torch.utils.data.Dataset): pass
    def load_dataset(path, splits): return []

# --- Configuration ---
LOG_FILE = '/home/iris/Documents/deep_learning/src/logs/train_sdxl_with_refiner.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s %(message)s',
    filemode='w',
)
logger = get_logger(__name__)
print(f"Logging to {LOG_FILE}")

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        print(f"Error: Configuration file not found: {config_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        print(f"Error loading configuration: {e}")
        return None

# --- Custom SDXL Pipeline with T5 ---
# class StableDiffusionXLPipelineWithT5(StableDiffusionXLPipeline):
#     def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, logger_instance=None):
#         super().__init__(
#             vae=vae,
#             text_encoder=text_encoder,
#             tokenizer=tokenizer,
#             unet=unet,
#             scheduler=scheduler,
#             text_encoder_2=None,
#             tokenizer_2=None
#         )
#         self.logger = logger_instance or logging.getLogger(__name__)
#         self.tokenizer.model_max_length = 512
#         self.projection_layer = torch.nn.Linear(768, 1280).to(device=self.unet.device, dtype=self.unet.dtype)
#         torch.nn.init.normal_(self.projection_layer.weight, mean=0.0, std=0.01)
#         torch.nn.init.zeros_(self.projection_layer.bias)
#         self.refiner_unet = None  # Managed separately, not a pipeline component
        # Explicitly register only the required components
        # self.register_modules(
        #     vae=vae,
        #     text_encoder=text_encoder,
        #     tokenizer=tokenizer,
        #     unet=unet,
        #     scheduler=scheduler,
        #     text_encoder_2=None,
        #     tokenizer_2=None
        # )

class StableDiffusionXLPipelineWithT5(StableDiffusionXLPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, logger_instance=None):
        # Call super init, explicitly setting unused SDXL components to None
        super().__init__(
            vae=vae,
            text_encoder=text_encoder, # T5 Encoder passed as primary
            tokenizer=tokenizer,       # T5 Tokenizer passed as primary
            unet=unet,
            scheduler=scheduler,
            text_encoder_2=None,       # Explicitly None
            tokenizer_2=None,          # Explicitly None
            image_encoder=None,        # Explicitly None
            feature_extractor=None     # Explicitly None
            # Note: force_zeros_for_empty_prompt is a config flag, not a component attribute here
        )
        self.logger = logger_instance or logging.getLogger(__name__)
        self.tokenizer.model_max_length = 512

        # Define and initialize projection layer *after* super().__init__
        # so self.device and self.dtype are available
        # self.projection_layer = torch.nn.Linear(768, 1280).to(device=self.device, dtype=self.dtype)
        self.projection_layer = torch.nn.Linear(768, 1280) # Define on CPU first
        torch.nn.init.normal_(self.projection_layer.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.projection_layer.bias)

        self.refiner_unet = None  # Managed separately, not a pipeline component

        # --- Clean up unused components from internal registry ---
        # These were added by the base SDXL __init__ but are None and not needed
        components_to_remove = ['text_encoder_2', 'tokenizer_2', 'image_encoder', 'feature_extractor']
        config_keys_to_remove = [] # Store config keys related to removed components

        # 1. Clean up internal _components dictionary
        if hasattr(self, '_components'):
            self.logger.debug(f"Pipeline Init: Before cleanup _components keys: {list(self._components.keys())}")
            for comp_name in components_to_remove:
                if comp_name in self._components:
                    del self._components[comp_name] # Remove from internal dict
            self.logger.debug(f"Pipeline Init: After cleanup _components keys: {list(self._components.keys())}")

        # 2. Clean up attributes (if they exist and are None)
        for comp_name in components_to_remove:
            if hasattr(self, comp_name):
                attr_value = getattr(self, comp_name, 'AttributeMissing')
                if attr_value is None:
                    try:
                        delattr(self, comp_name)
                        self.logger.debug(f"Deleted attribute {comp_name}")
                    except AttributeError:
                         self.logger.debug(f"Attribute {comp_name} not found or couldn't be deleted.")

        # 3. Identify and remove corresponding keys from the pipeline's config *** CRITICAL STEP ***
        if hasattr(self, 'config'):
             self.logger.debug(f"Pipeline Init: Before cleanup config keys: {list(self.config.keys())}")
             # Convert config to dict to safely iterate and modify
             config_dict = self.config.to_dict()
             keys_to_del_from_config = []

             # Find all keys related to the components we want to remove
             for key in config_dict.keys():
                 # Check if the key itself is one of the components or contains its name
                 # (e.g., 'text_encoder_2', or keys related to its config like '_text_encoder_2_name_or_path')
                 if any(comp_name in key for comp_name in components_to_remove):
                      keys_to_del_from_config.append(key)

             # Delete the identified keys from the dictionary
             for key in keys_to_del_from_config:
                 if key in config_dict:
                     del config_dict[key]
                     self.logger.debug(f"Removed key '{key}' from config dict")

             # Overwrite the pipeline's config with the cleaned dictionary
             # We need to be careful about the config type. Let's update the existing object.
             # Directly modifying self.config._internal_dict might be possible but risky.
             # Updating via direct attribute access if config is like a dot-dict:
             original_config_keys = list(self.config.keys())
             for key in original_config_keys:
                 if key not in config_dict and hasattr(self.config, key):
                    try:
                        delattr(self.config, key)
                        self.logger.debug(f"Deleted attribute {key} from config object")
                    except Exception as e:
                        self.logger.warning(f"Could not delete attribute {key} from config object: {e}")
                 elif key in config_dict and hasattr(self.config, key):
                     # Ensure existing keys are updated if necessary (though unlikely here)
                     # setattr(self.config, key, config_dict[key])
                     pass # Usually deletion is sufficient


             self.logger.debug(f"Pipeline Init: After cleanup config keys: {list(self.config.keys())}")
             # Final check - ensure expected keys are still there
             expected_keys = {'vae', 'tokenizer', 'text_encoder', 'scheduler', 'unet'}
             missing_keys = expected_keys - set(self.config.keys())
             if missing_keys:
                 self.logger.warning(f"Config cleanup might have removed expected keys: {missing_keys}")


    def to(self, *args, **kwargs):
        """ Override to ensure projection layer is also moved """
        # Call the original to method first to move registered components
        super_return = super().to(*args, **kwargs)

        # Move the custom projection layer
        if hasattr(self, 'projection_layer') and isinstance(self.projection_layer, torch.nn.Module):
            try:
                self.projection_layer.to(*args, **kwargs)
                self.logger.debug(f"Moved projection_layer to device/dtype specified in .to()")
            except Exception as e:
                self.logger.error(f"Failed to move projection_layer in overridden .to(): {e}")

        # Also ensure refiner_unet is moved if it exists (set_refiner_unet might not catch subsequent .to calls)
        if hasattr(self, 'refiner_unet') and self.refiner_unet is not None:
             try:
                 self.refiner_unet.to(*args, **kwargs)
                 self.logger.debug(f"Moved refiner_unet in overridden .to()")
             except Exception as e:
                 self.logger.error(f"Failed to move refiner_unet in overridden .to(): {e}")

        return super_return

    def set_refiner_unet(self, refiner_unet):
        """Set the refiner UNet and move it to the correct device/dtype."""
        self.refiner_unet = refiner_unet
        if refiner_unet:
            # Move refiner based on the *current* device/dtype of the main unet
            try:
                 target_device = self.unet.device
                 target_dtype = self.unet.dtype
                 self.refiner_unet.to(device=target_device, dtype=target_dtype)
                 self.logger.info(f"Set and moved refiner UNet to device {target_device}, dtype {target_dtype}")
            except Exception as e:
                 self.logger.error(f"Failed to move refiner UNet in set_refiner_unet: {e}")


    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        if isinstance(prompt, str): prompt = [prompt]
        if not all(isinstance(p, str) and p.strip() for p in prompt):
            self.logger.warning(f"Invalid prompts detected: {prompt}. Using default prompt.")
            prompt = ["an abstract representation"] * batch_size
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        ).to(device)
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        self.logger.debug(f"Input IDs stats - min: {input_ids.min().item()}, max: {input_ids.max().item()}, mean: {input_ids.float().mean().item():.4f}")
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            encoder_hidden_states = torch.nan_to_num(encoder_hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
            self.logger.debug(f"T5 output stats - min: {encoder_hidden_states.min().item():.4f}, max: {encoder_hidden_states.max().item():.4f}, mean: {encoder_hidden_states.mean().item():.4f}")
        prompt_embeds = encoder_hidden_states.mean(dim=1)
        prompt_embeds = self.projection_layer(prompt_embeds)
        prompt_embeds = torch.nan_to_num(prompt_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        std = prompt_embeds.std(dim=1, keepdim=True)
        if (std > 1e-6).all():
            prompt_embeds = prompt_embeds / (std.clamp(min=1e-6) * 10.0)
        prompt_embeds = prompt_embeds.unsqueeze(1).expand(-1, 77, -1)
        add_time_ids = torch.tensor([[self.vae.config.sample_size, self.vae.config.sample_size, 0, 0, self.vae.config.sample_size, self.vae.config.sample_size]], dtype=prompt_embeds.dtype).repeat(batch_size, 1).to(device)
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or [""] * batch_size
            if isinstance(negative_prompt, str): negative_prompt = [negative_prompt]
            negative_inputs = self.tokenizer(
                negative_prompt, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
            ).to(device)
            negative_hidden_states = self.text_encoder(negative_inputs.input_ids, attention_mask=negative_inputs.attention_mask).last_hidden_state
            negative_hidden_states = torch.nan_to_num(negative_hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
            negative_embeds = negative_hidden_states.mean(dim=1)
            negative_embeds = self.projection_layer(negative_embeds)
            negative_embeds = torch.nan_to_num(negative_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
            std_neg = negative_embeds.std(dim=1, keepdim=True)
            if (std_neg > 1e-6).all():
                negative_embeds = negative_embeds / (std_neg.clamp(min=1e-6) * 10.0)
            negative_embeds = negative_embeds.unsqueeze(1).expand(-1, 77, -1)
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
            add_time_ids = add_time_ids.repeat(2, 1)
        return prompt_embeds, {"text_embeds": torch.zeros(batch_size, 1280).to(device, prompt_embeds.dtype), "time_ids": add_time_ids}

    def __call__(self, prompt, negative_prompt=None, num_inference_steps=50, guidance_scale=7.5, generator=None, output_type="pil", refiner_steps=10, **kwargs):
        device = self.unet.device
        dtype = self.unet.dtype
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, added_cond_kwargs = self._encode_prompt(
            prompt, device, 1, do_classifier_free_guidance, negative_prompt
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = torch.randn((batch_size, self.unet.in_channels, 1024 // 8, 1024 // 8), device=device, dtype=dtype, generator=generator)
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs
            ).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        if self.refiner_unet:
            for _ in range(refiner_steps):
                noise_pred = self.refiner_unet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds[:batch_size],
                    added_cond_kwargs=added_cond_kwargs
                ).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents.to(dtype=torch.float32)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.permute(0, 2, 3, 1).float().cpu().numpy()
        if output_type == "pil":
            images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
        return images

# --- FinetuneModel Class ---
class FinetuneModel:
    def __init__(self, model_name, output_dir, accelerator: Accelerator, logger_instance=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.logger = logger_instance or logging.getLogger(__name__)
        self.device = accelerator.device
        self.dtype = torch.float16
        self.tokenizer = None
        self.text_encoder = None
        self.unet = None
        self.refiner_unet = None
        self.scheduler = None
        self.vae = None
        self.pipeline = None
        self.image_size = 1024
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.current_epoch = 0
        self._apply_lora_text_flag = False
        self._apply_lora_unet_flag = False
        self._apply_lora_refiner_flag = False
        self.fold_val_losses = []

    def load_model(self):
        try:
            t5_model_name = "t5-base"
            base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

            # Load components individually (onto CPU first)
            self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            self.tokenizer.model_max_length = 512

            # Load models onto CPU initially
            self.text_encoder = T5EncoderModel.from_pretrained(t5_model_name)
            self.vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
            self.refiner_unet = UNet2DConditionModel.from_pretrained(refiner_model_id, subfolder="unet")
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")

            # --- Enable gradient checkpointing ---
            # Use the correct method/attribute for each model type
            self.logger.info("Attempting to enable gradient checkpointing...")
            try:
                # For T5EncoderModel (transformers)
                self.text_encoder.gradient_checkpointing = True
                self.logger.info("Enabled gradient checkpointing for T5 Text Encoder.")
            except AttributeError:
                 self.logger.warning("Could not enable gradient checkpointing for T5 Text Encoder via attribute.")
                 # Add alternative ways if needed, or just log the warning.

            try:
                # For UNet2DConditionModel (diffusers)
                self.unet.enable_gradient_checkpointing()
                self.logger.info("Enabled gradient checkpointing for UNet.")
            except AttributeError:
                 self.logger.warning("Could not enable gradient checkpointing for UNet via method 'enable_gradient_checkpointing'.")

            try:
                 # For Refiner UNet (diffusers)
                self.refiner_unet.enable_gradient_checkpointing()
                self.logger.info("Enabled gradient checkpointing for Refiner UNet.")
            except AttributeError:
                 self.logger.warning("Could not enable gradient checkpointing for Refiner UNet via method 'enable_gradient_checkpointing'.")


            # Instantiate the custom pipeline (components still on CPU)
            self.pipeline = StableDiffusionXLPipelineWithT5(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.scheduler,
                logger_instance=self.logger
            )
            # The pipeline __init__ defines projection_layer but doesn't move it

            # Set the refiner UNet (also still on CPU, set_refiner_unet won't move it yet)
            self.pipeline.set_refiner_unet(self.refiner_unet)

            # --- Move models to target device and set dtype ---
            self.logger.info(f"Moving models to target device: {self.device} and dtype: {self.dtype}")

            # Move VAE separately (often kept in fp32)
            self.vae.to(self.device, dtype=torch.float32)
            self.vae.eval() # Set VAE to eval mode permanently

            # Move the rest using the overridden pipeline.to() method
            # This handles unet, text_encoder, scheduler, tokenizer (if applicable),
            # projection_layer, and refiner_unet.
            self.pipeline.to(self.device, dtype=self.dtype)

            self.logger.info("Model components loaded, pipeline initialized, and models moved to device.")

        except Exception as e:
            self.logger.error(f"Failed to load model components: {e}\n{traceback.format_exc()}")
            raise


    def modify_architecture(self, apply_lora_to_unet=True, apply_lora_to_refiner=True, apply_lora_to_text_encoder=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        self._apply_lora_unet_flag = apply_lora_to_unet
        self._apply_lora_refiner_flag = apply_lora_to_refiner
        self._apply_lora_text_flag = apply_lora_to_text_encoder
        lora_bias = "none"
        unet_target_modules = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=lora_bias, target_modules=unet_target_modules)
        if apply_lora_to_unet and self.unet:
            self.unet = get_peft_model(self.unet, lora_config)
            self.unet.print_trainable_parameters()
            self.logger.info("Applied LoRA to base UNet")
        if apply_lora_to_refiner and self.refiner_unet:
            self.refiner_unet = get_peft_model(self.refiner_unet, lora_config)
            self.refiner_unet.print_trainable_parameters()
            self.logger.info("Applied LoRA to refiner UNet")
        if apply_lora_to_text_encoder and self.text_encoder:
            text_target_modules = ["q", "k", "v", "o"]
            lora_config_text = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=lora_bias, target_modules=text_target_modules)
            self.text_encoder = get_peft_model(self.text_encoder, lora_config_text)
            self.text_encoder.print_trainable_parameters()
            for param in self.pipeline.projection_layer.parameters():
                param.requires_grad = True
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
            self.logger.info("Applied LoRA to T5 text encoder")

    def validate(self, val_dataloader, dataset_path):
        if not all([self.unet, self.refiner_unet, self.pipeline, self.vae, self.scheduler]):
            self.logger.error("Missing required components for validation")
            return float('inf')
        self.unet.eval()
        self.refiner_unet.eval()
        self.text_encoder.eval()
        self.pipeline.projection_layer.eval()
        self.vae.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        image_folder = os.path.join(os.path.dirname(dataset_path), "images")
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                try:
                    image_filenames = batch.get('image')
                    prompts = batch.get('prompt')
                    pixel_values_list = []
                    valid_prompts = []
                    for img_filename, prompt in zip(image_filenames, prompts):
                        try:
                            image_path = os.path.join(image_folder, img_filename)
                            image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
                            image_np = np.array(image).astype(np.float32) / 255.0
                            if np.any(np.isnan(image_np)) or np.any(np.isinf(image_np)):
                                self.logger.warning(f"Val Step {step+1}: NaN/Inf in image {img_filename}")
                                continue
                            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                            pixel_values_list.append(image_tensor)
                            valid_prompts.append(prompt)
                        except Exception as img_err:
                            self.logger.debug(f"Val Skip img {img_filename}: {img_err}")
                    if not pixel_values_list:
                        self.logger.debug(f"Skipping val batch {step+1}: No valid images")
                        continue
                    pixel_values = torch.stack(pixel_values_list).to(self.accelerator.device, dtype=torch.float32)
                    prompts = valid_prompts
                    prompt_embeds, added_cond_kwargs = self.pipeline._encode_prompt(
                        prompts, self.accelerator.device, 1, False
                    )
                    pixel_values_norm = pixel_values * 2.0 - 1.0
                    vae_output = self.vae.encode(pixel_values_norm)
                    latents = vae_output.latent_dist.sample() * self.vae.config.scaling_factor
                    latents = latents.to(dtype=self.dtype)
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        self.logger.warning(f"Val Step {step+1}: NaN/Inf in latents")
                        continue
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                    noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                    noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0)
                    # Base UNet prediction
                    model_pred = self.unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                    if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                        self.logger.warning(f"Val Step {step+1}: NaN/Inf in base model_pred")
                        continue
                    model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1.0, neginf=-1.0)
                    # Refiner UNet prediction
                    refiner_pred = self.refiner_unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                    if torch.isnan(refiner_pred).any() or torch.isinf(refiner_pred).any():
                        self.logger.warning(f"Val Step {step+1}: NaN/Inf in refiner_pred")
                        continue
                    refiner_pred = torch.nan_to_num(refiner_pred, nan=0.0, posinf=1.0, neginf=-1.0)
                    # Combined loss
                    base_loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    refiner_loss = torch.nn.functional.mse_loss(refiner_pred.float(), noise.float(), reduction="mean")
                    val_loss = (base_loss + refiner_loss) / 2.0
                    if torch.isnan(val_loss) or torch.isinf(val_loss):
                        self.logger.warning(f"Val Step {step+1}: Calculated loss is NaN or Inf")
                        continue
                    total_val_loss += self.accelerator.gather(val_loss).mean().item()
                    num_val_batches += 1
                except Exception as val_step_err:
                    self.logger.error(f"Validation step {step+1} failed: {val_step_err}\n{traceback.format_exc()}")
                    continue
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        self.unet.train()
        self.refiner_unet.train()
        self.text_encoder.train()
        self.pipeline.projection_layer.train()
        self.logger.info(f"Validation completed - Avg validation loss: {avg_val_loss:.4f}, Batches: {num_val_batches}")
        return avg_val_loss

    def save_model_state(self, epoch=None, val_loss=None, hyperparameters=None, subdir=None):
        if not self.accelerator.is_main_process:
            return
        output_subdir = subdir if subdir else self.output_dir
        os.makedirs(output_subdir, exist_ok=True)
        save_label = "best"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None and not np.isnan(val_loss) else "N/A"
        save_paths = {}
        if self.unet and isinstance(self.unet, PeftModel):
            unet_path = os.path.join(output_subdir, f"{save_label}_unet_lora")
            self.accelerator.unwrap_model(self.unet).save_pretrained(unet_path)
            save_paths["UNet_LoRA"] = unet_path
            self.logger.info(f"Saved base UNet LoRA to {unet_path}")
        if self.refiner_unet and isinstance(self.refiner_unet, PeftModel):
            refiner_path = os.path.join(output_subdir, f"{save_label}_refiner_unet_lora")
            self.accelerator.unwrap_model(self.refiner_unet).save_pretrained(refiner_path)
            save_paths["Refiner_UNet_LoRA"] = refiner_path
            self.logger.info(f"Saved refiner UNet LoRA to {refiner_path}")
        if self.text_encoder and isinstance(self.text_encoder, PeftModel):
            text_encoder_path = os.path.join(output_subdir, f"{save_label}_text_encoder_lora")
            self.accelerator.unwrap_model(self.text_encoder).save_pretrained(text_encoder_path)
            save_paths["TextEncoder_LoRA"] = text_encoder_path
            self.logger.info(f"Saved TextEncoder LoRA to {text_encoder_path}")
        if self.pipeline.projection_layer:
            proj_layer_path = os.path.join(output_subdir, f"{save_label}_projection_layer.pth")
            torch.save(self.accelerator.get_state_dict(self.pipeline.projection_layer), proj_layer_path)
            save_paths["Projection_Layer"] = proj_layer_path
            self.logger.info(f"Saved Projection Layer to {proj_layer_path}")
        if hyperparameters:
            hyperparameters.update({
                '_apply_lora_text_encoder': self._apply_lora_text_flag,
                '_apply_lora_unet': self._apply_lora_unet_flag,
                '_apply_lora_refiner': self._apply_lora_refiner_flag
            })
            hyperparam_path = os.path.join(output_subdir, f"{save_label}_hyperparameters.json")
            save_data = {'model_name': self.model_name, 'epoch': epoch, 'validation_loss': val_loss_str, 'hyperparameters': hyperparameters}
            with open(hyperparam_path, 'w') as f:
                json.dump(save_data, f, indent=4)
            self.logger.info(f"Saved hyperparameters to {hyperparam_path}")
        if not save_paths:
            self.logger.warning(f"No trainable weights saved for {self.model_name} ({save_label}).")
        else:
            self.logger.info(f"Saved model components: {save_paths}")

    def fine_tune(self, dataset_path, train_val_splits, epochs=1, batch_size=1, learning_rate=5e-8, gradient_accumulation_steps=8, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        self.logger.info(f"Requested learning rate: {learning_rate}")
        # Capping LR - keep if needed
        if learning_rate > 5e-8:
             self.logger.warning(f"Capping learning rate at 5e-8 for stability.")
             learning_rate = 5e-8

        self.fold_val_losses = []
        global_best_avg_val_loss = float('inf')
        global_best_hyperparameters = None
        global_best_epoch = -1

        for fold_idx, (train_dataset, val_dataset) in enumerate(train_val_splits):
            self.best_val_loss = float('inf')
            self.best_epoch = -1
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            image_folder = os.path.join(os.path.dirname(dataset_path), "images")

            # --- Collect parameters to optimize ---
            params_to_optimize = []
            trainable_models = [] # Keep track of models passed to accelerator

            if self._apply_lora_unet_flag and isinstance(self.unet, PeftModel):
                params_to_optimize.extend(p for p in self.unet.parameters() if p.requires_grad)
                trainable_models.append(self.unet)
                self.logger.info("Adding UNet LoRA parameters to optimizer.")
            elif self._apply_lora_unet_flag: # Should be PeftModel, add warning if not
                 self.logger.warning("apply_lora_unet is True, but UNet is not a PeftModel.")
            # If not applying LoRA but want to fine-tune whole UNet (less common/efficient)
            # elif not self._apply_lora_unet_flag and config allows full fine-tune:
            #    params_to_optimize.extend(p for p in self.unet.parameters() if p.requires_grad)
            #    trainable_models.append(self.unet)


            if self._apply_lora_refiner_flag and isinstance(self.refiner_unet, PeftModel):
                params_to_optimize.extend(p for p in self.refiner_unet.parameters() if p.requires_grad)
                trainable_models.append(self.refiner_unet)
            elif self._apply_lora_refiner_flag:
                 self.logger.warning("apply_lora_refiner is True, but Refiner UNet is not a PeftModel.")

            if self._apply_lora_text_flag and isinstance(self.text_encoder, PeftModel):
                params_to_optimize.extend(p for p in self.text_encoder.parameters() if p.requires_grad)
                trainable_models.append(self.text_encoder)
                # Also ensure projection layer is trainable if text encoder LoRA is active
                if hasattr(self.pipeline, 'projection_layer'):
                    for param in self.pipeline.projection_layer.parameters():
                        param.requires_grad = True
                    params_to_optimize.extend(p for p in self.pipeline.projection_layer.parameters() if p.requires_grad)
                    # Add projection layer only if it wasn't already added via another condition
                    if self.pipeline.projection_layer not in trainable_models:
                         trainable_models.append(self.pipeline.projection_layer)
                    self.logger.info("Adding Text Encoder LoRA and Projection Layer parameters to optimizer.")
            elif hasattr(self.pipeline, 'projection_layer'): # If not training TE LoRA, ensure projection layer grads are off
                 for param in self.pipeline.projection_layer.parameters():
                     param.requires_grad = False


            # Always include projection layer parameters if it exists and requires grad
            # if self.pipeline.projection_layer:
            #      # Ensure requires_grad is True if training it (should be by default)
            #      for param in self.pipeline.projection_layer.parameters():
            #          param.requires_grad = True
            #      params_to_optimize.extend(p for p in self.pipeline.projection_layer.parameters() if p.requires_grad)
            #      trainable_models.append(self.pipeline.projection_layer) # Add projection layer

            # Deduplicate parameters (important!)
            params_to_optimize = list({id(p): p for p in params_to_optimize}.values())
            self.logger.info(f"Number of parameters to optimize: {len(params_to_optimize)}")

            if not params_to_optimize:
                self.logger.error("Optimizer params list is empty! No models marked for training (LoRA or otherwise).")
                continue # Skip fold if nothing to train

            # Ensure unique parameters if models share layers (unlikely here)
            params_to_optimize = list({id(p): p for p in params_to_optimize}.values())
            self.logger.info(f"Number of parameters to optimize: {len(params_to_optimize)}")
            if not trainable_models:
                 self.logger.error(f"No trainable components identified for Fold {fold_idx+1}. Skipping.")
                 continue

            optimizer = AdamW8bit(params_to_optimize, lr=learning_rate)

            # --- Prepare components with Accelerator ---
            # Include all models/layers being trained, optimizer, and dataloaders
            prepare_list = trainable_models + [optimizer, train_dataloader, val_dataloader]
            prepared_components = self.accelerator.prepare(*prepare_list)

            # Unpack prepared components carefully based on what was passed in
            num_models = len(trainable_models)
            prepared_models = prepared_components[:num_models]
            self.optimizer, train_dataloader, val_dataloader = prepared_components[num_models:]

            # Update model references (important!)
            # This assumes the order in trainable_models is preserved by prepare
            model_map = {type(orig).__name__: prep for orig, prep in zip(trainable_models, prepared_models)}

            if self.unet in trainable_models: self.unet = model_map.get('PeftModel', self.unet) # Assumes LoRA UNet is PeftModel
            if self.refiner_unet in trainable_models: self.refiner_unet = model_map.get('PeftModel', self.refiner_unet) # Assumes LoRA Refiner is PeftModel
            if self.text_encoder in trainable_models: self.text_encoder = model_map.get('PeftModel', self.text_encoder) # Assumes LoRA TE is PeftModel
            if self.pipeline.projection_layer in trainable_models: self.pipeline.projection_layer = model_map.get('Linear', self.pipeline.projection_layer) # Projection is Linear

            # Models *not* trained but used (like VAE) should still be moved to device
            self.pipeline.to(self.accelerator.device) # Ensure pipeline components are on device
            self.vae.to(self.accelerator.device)
            self.vae.eval() # Keep VAE in eval mode


            hyperparameters = {
                'model_name': self.model_name,
                'text_encoder': 't5-base',
                'epochs': epochs,
                'batch_size': batch_size * self.accelerator.num_processes, # Adjust effective batch size
                'learning_rate': learning_rate,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout
            }

            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
            max_train_steps = epochs * num_update_steps_per_epoch
            global_step = 0

            self.logger.info(f"***** Running training for Fold {fold_idx+1} *****")
            self.logger.info(f"  Num examples = {len(train_dataset)}")
            self.logger.info(f"  Num Epochs = {epochs}")
            self.logger.info(f"  Instantaneous batch size per device = {batch_size}")
            self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {batch_size * self.accelerator.num_processes * gradient_accumulation_steps}")
            self.logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
            self.logger.info(f"  Total optimization steps = {max_train_steps}")

            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                # Set models to train mode (accelerator might handle this, but explicit is fine)
                for model in prepared_models: model.train()

                train_loss_epoch = 0.0
                num_train_batches_epoch = 0
                num_skipped_steps = 0

                for step, batch in enumerate(train_dataloader):
                    # Check if batch is smaller than expected due to dataset size & num_processes
                    is_final_batch = (step == len(train_dataloader) - 1)

                    try:
                        image_filenames = batch.get('image')
                        prompts = batch.get('prompt')
                        if image_filenames is None or prompts is None:
                             self.logger.warning(f"Train Step {step+1}: Batch missing 'image' or 'prompt'. Skipping.")
                             continue

                        # --- Image Loading and Preprocessing ---
                        pixel_values_list = []
                        valid_prompts = []
                        for img_filename, prompt in zip(image_filenames, prompts):
                            # Basic check for valid prompt
                            if not isinstance(prompt, str) or not prompt.strip():
                                self.logger.warning(f"Train Step {step+1}: Invalid or empty prompt for image {img_filename}. Skipping item.")
                                continue
                            try:
                                image_path = os.path.join(image_folder, img_filename)
                                image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
                                image_np = np.array(image).astype(np.float32) / 255.0
                                if np.any(np.isnan(image_np)) or np.any(np.isinf(image_np)):
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in image {img_filename}. Skipping item.")
                                    continue
                                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1) # C, H, W
                                pixel_values_list.append(image_tensor)
                                valid_prompts.append(prompt)
                            except FileNotFoundError:
                                self.logger.warning(f"Train Step {step+1}: Image file not found {image_path}. Skipping item.")
                            except Exception as img_err:
                                self.logger.warning(f"Train Step {step+1}: Error loading image {img_filename}: {img_err}. Skipping item.")

                        if not pixel_values_list:
                            self.logger.debug(f"Skipping train batch {step+1}: No valid image/prompt pairs.")
                            continue

                        # Stack valid images and convert to target device/dtype
                        pixel_values = torch.stack(pixel_values_list).to(self.accelerator.device, dtype=torch.float32) # Keep images fp32 for VAE
                        prompts = valid_prompts # Use only prompts corresponding to valid images

                        # --- Training Step ---
                        # Use accelerator.accumulate context manager for gradient accumulation
                        with self.accelerator.accumulate(*prepared_models): # Pass all models being trained
                            # Use autocast for mixed precision
                            with torch.cuda.amp.autocast(enabled=self.accelerator.mixed_precision == 'fp16'):
                                # 1. Encode Prompts (using pipeline's method)
                                # Ensure T5 encoder is in training mode if LoRA applied
                                if self.text_encoder in prepared_models: self.text_encoder.train()
                                # Ensure projection layer is in training mode
                                if self.pipeline.projection_layer in prepared_models: self.pipeline.projection_layer.train()

                                prompt_embeds, added_cond_kwargs = self.pipeline._encode_prompt(
                                    prompts, self.accelerator.device, 1, False # is_train=True? No CFG needed
                                )
                                # Ensure embeds are on correct device/dtype
                                prompt_embeds = prompt_embeds.to(dtype=self.dtype)
                                for k, v in added_cond_kwargs.items():
                                    added_cond_kwargs[k] = v.to(dtype=self.dtype)


                                # 2. Encode Images to Latents (VAE in eval mode, no gradients)
                                # VAE requires fp32 input typically
                                pixel_values_norm = pixel_values * 2.0 - 1.0
                                with torch.no_grad(): # Ensure no gradients for VAE
                                    vae_output = self.vae.encode(pixel_values_norm.to(self.vae.dtype)) # Use VAE's expected dtype
                                    latents = vae_output.latent_dist.sample() * self.vae.config.scaling_factor
                                    latents = latents.to(dtype=self.dtype) # Convert latents to training dtype

                                if torch.isnan(latents).any() or torch.isinf(latents).any():
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf detected in latents. Skipping step.")
                                    # Consider zeroing gradients accumulated so far for this step if possible?
                                    self.optimizer.zero_grad() # Zero grad to prevent issue propagation
                                    continue

                                # 3. Sample Noise and Timesteps
                                noise = torch.randn_like(latents)
                                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()

                                # 4. Add Noise to Latents
                                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                                noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0) # Clip/replace invalid numbers

                                # 5. Predict Noise using UNets (Base and Refiner)
                                # Ensure UNets are in training mode
                                if self.unet in prepared_models: self.unet.train()
                                if self.refiner_unet in prepared_models: self.refiner_unet.train()

                                # Base UNet prediction
                                model_pred = self.unet(
                                    sample=noisy_latents,
                                    timestep=timesteps,
                                    encoder_hidden_states=prompt_embeds,
                                    added_cond_kwargs=added_cond_kwargs
                                ).sample
                                if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in base model_pred. Skipping step.")
                                    self.optimizer.zero_grad()
                                    continue
                                model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1.0, neginf=-1.0)


                                # Refiner UNet prediction (using same inputs for simplicity in training)
                                refiner_pred = torch.tensor(0.0, device=latents.device) # Default if not training refiner
                                refiner_loss = torch.tensor(0.0, device=latents.device)
                                loss_count = 1.0 # Start with base loss

                                if self.refiner_unet in prepared_models: # Only compute if refiner is being trained
                                    refiner_pred = self.refiner_unet(
                                        sample=noisy_latents, # Use same noisy latents
                                        timestep=timesteps,   # Use same timesteps
                                        encoder_hidden_states=prompt_embeds,
                                        added_cond_kwargs=added_cond_kwargs
                                    ).sample
                                    if torch.isnan(refiner_pred).any() or torch.isinf(refiner_pred).any():
                                         self.logger.warning(f"Train Step {step+1}: NaN/Inf in refiner_pred. Skipping step.")
                                         self.optimizer.zero_grad()
                                         continue
                                    refiner_pred = torch.nan_to_num(refiner_pred, nan=0.0, posinf=1.0, neginf=-1.0)
                                    refiner_loss = torch.nn.functional.mse_loss(refiner_pred.float(), noise.float(), reduction="mean")
                                    loss_count += 1.0


                                # 6. Calculate Loss
                                # Target is the original noise
                                base_loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                                # Combine losses (average if both are computed)
                                loss = (base_loss + refiner_loss) / loss_count


                                if torch.isnan(loss) or torch.isinf(loss):
                                    self.logger.warning(f"Train Step {step+1}: Calculated loss is NaN or Inf. Skipping step.")
                                    self.optimizer.zero_grad()
                                    continue

                            # --- Backpropagation and Optimization ---
                            # Scale loss for gradient accumulation
                            # accelerator.backward handles mixed precision scaling automatically
                            self.accelerator.backward(loss) # Removed division by grad_accum steps, accelerator handles it? Check docs. Re-add if needed.
                            # Let's assume backward needs the scaled loss if accumulate context doesn't do it
                            # self.accelerator.backward(loss / gradient_accumulation_steps)

                            # Optimizer step occurs only when gradients are synchronized
                            if self.accelerator.sync_gradients:
                                valid_gradients = True
                                grad_norm = torch.tensor(0.0, device=self.device) # Initialize grad_norm
                                try:
                                    # Clip gradients only for the parameters being optimized
                                    grad_norm = self.accelerator.clip_grad_norm_(params_to_optimize, 1.0) # Use 1.0 as max_norm
                                    # Check for NaN/Inf in grad_norm itself
                                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                        self.logger.warning(f"Train Step {step+1}: NaN/Inf gradient norm detected ({grad_norm.item()}). Skipping optimizer step.")
                                        valid_gradients = False
                                    else:
                                         self.logger.debug(f"Train Step {step+1}: Grad norm: {grad_norm.item():.4f}")
                                except Exception as clip_err:
                                    # Catch potential errors during clipping itself (e.g., empty param list?)
                                    self.logger.warning(f"Train Step {step+1}: Error during gradient clipping: {clip_err}. Skipping optimizer step.")
                                    valid_gradients = False

                                if valid_gradients:
                                    self.optimizer.step() # Perform optimizer step
                                    self.optimizer.zero_grad() # Reset gradients
                                else:
                                    # Gradients were invalid, ensure they are zeroed before next accumulation
                                    self.optimizer.zero_grad()
                                    num_skipped_steps += 1
                                global_step += 1 # Increment global step only on optimizer step

                            # --- Logging ---
                            # Gather loss across processes for logging
                            avg_loss = self.accelerator.gather(loss.repeat(batch_size)).mean() # Use gather on the unscaled loss
                            train_loss_epoch += avg_loss.item() # Accumulate gathered loss
                            num_train_batches_epoch += 1

                            if self.accelerator.is_main_process and global_step % 50 == 0: # Log every 50 optimizer steps
                                self.logger.info(f"Fold {fold_idx+1}, Epoch {self.current_epoch}, Step {global_step}/{max_train_steps}, Train Loss: {avg_loss.item():.4f}, Skipped Steps: {num_skipped_steps}")


                    # --- Error Handling for Batch ---
                    except Exception as e:
                        self.logger.error(f"Training step {step+1} (Fold {fold_idx+1}, Epoch {self.current_epoch}) failed: {e}\n{traceback.format_exc()}")
                        if "out of memory" in str(e).lower():
                            self.logger.error("OOM Error detected during training step. Consider reducing batch size or enabling more memory optimization techniques.")
                            # Optional: break epoch or attempt recovery
                        # Ensure gradients are zeroed if an error occurred mid-step
                        if self.accelerator.sync_gradients:
                             self.optimizer.zero_grad()
                        continue # Continue to next batch

                # --- End of Epoch ---
                avg_train_loss_epoch = train_loss_epoch / num_train_batches_epoch if num_train_batches_epoch > 0 else float('nan')
                self.logger.info(f"Fold {fold_idx+1}, Epoch {self.current_epoch} Finished - Avg Train Loss: {avg_train_loss_epoch:.4f}, Total Skipped Steps: {num_skipped_steps}")

                # --- Validation ---
                if self.accelerator.is_main_process: # Perform validation only on main process
                    self.logger.info(f"Running validation for Fold {fold_idx+1}, Epoch {self.current_epoch}...")
                    # Pass prepared models (potentially wrapped by accelerator) to validate
                    avg_val_loss = self.validate(val_dataloader, dataset_path) # Validate uses models in eval mode internally

                    if not np.isnan(avg_val_loss) and avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self.best_epoch = self.current_epoch
                        save_dir = os.path.join(self.output_dir, f"fold_{fold_idx+1}_best_model") # Save best model per fold
                        self.logger.info(f"*** New best validation loss for Fold {fold_idx+1}: {avg_val_loss:.4f} at Epoch {self.current_epoch} ***")
                        self.save_model_state(epoch=self.current_epoch, val_loss=avg_val_loss, hyperparameters=hyperparameters, subdir=save_dir)
                    else:
                         self.logger.info(f"Validation loss ({avg_val_loss:.4f}) did not improve from best ({self.best_val_loss:.4f})")


                # Wait for all processes before starting next epoch or finishing fold
                self.accelerator.wait_for_everyone()
                gc.collect()
                torch.cuda.empty_cache()

            # --- End of Fold ---
            self.fold_val_losses.append(self.best_val_loss) # Store best val loss for this fold
            if self.accelerator.is_main_process:
                self.logger.info(f"Fold {fold_idx+1} finished. Best Validation Loss: {self.best_val_loss:.4f} at Epoch {self.best_epoch}")
                if self.best_val_loss < global_best_avg_val_loss:
                    global_best_avg_val_loss = self.best_val_loss
                    global_best_hyperparameters = hyperparameters # Store the hyperparams that led to the best fold
                    global_best_epoch = self.best_epoch # Store the epoch from the best fold

            # Cleanup for next fold - potentially reload fresh model weights?
            # Depending on setup, LoRA weights might persist. Need to reset or reload.
            # Simplest is to re-initialize FinetuneModel for each hyperparam config,
            # but for k-fold with *same* hyperparams, need to reset weights between folds.
            # Current code re-uses the model state. Add reloading if needed.


        # --- End of K-Fold Cross-Validation ---
        valid_losses = [loss for loss in self.fold_val_losses if loss != float('inf') and not np.isnan(loss)]
        avg_kfold_val_loss = np.mean(valid_losses) if valid_losses else float('inf')
        self.logger.info(f"K-Fold Training Finished for current hyperparameters - Avg Best Validation Loss across {len(valid_losses)} valid folds: {avg_kfold_val_loss:.4f}")

        # Return the average best loss across folds for this hyperparameter set
        return avg_kfold_val_loss, None, global_best_hyperparameters, global_best_epoch # Return hyperparams associated with the single best fold


# --- Main Execution ---
def run_finetune(config_path):
    config = load_config(config_path)
    if config is None:
        print("Exiting due to config load failure.")
        return

    # --- Accelerator Setup ---
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 8)
    mixed_precision = config.get("mixed_precision", 'fp16') # Ensure config can specify this
    output_dir = config.get("base_output_dir", "./output")
    project_dir = os.path.join(output_dir, "logs")

    # Ensure log directory exists before initializing Accelerator with TensorBoard
    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)
        print(f"Created project directory for logs: {project_dir}")

    # Try installing tensorboard if missing
    try:
        import tensorboard
        log_with = "tensorboard"
        print("TensorBoard found. Logging enabled.")
    except ImportError:
        log_with = None
        print("TensorBoard not found. Skipping TensorBoard logging.")
        print("You can install it using: pip install tensorboard")


    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with,
        project_dir=project_dir
    )

    # Setup logging AFTER accelerator init to potentially use accelerator.is_main_process
    # Basic logging setup was already done, enhance or use accelerator's logger
    global logger # Access the global logger
    logger = get_logger(__name__, log_level="INFO") # Use accelerator's logger setup
    logger.info(f"Logging to {LOG_FILE} and accelerator handlers (if any).") # Log file setup remains
    logger.info(accelerator.state, main_process_only=False) # Log state on all processes


    # --- Dataset Loading ---
    dataset_path = config.get("dataset_path", "/home/iris/Documents/deep_learning/data/finetune_dataset/coco/dataset.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset JSON path not found: {dataset_path}")
        return
    logger.info(f"Loading dataset from: {dataset_path}")

    # Assume load_dataset returns a standard PyTorch Dataset or compatible object
    try:
        full_dataset = load_dataset(dataset_path) # Make sure load_dataset is correctly implemented
        if full_dataset is None or len(full_dataset) == 0:
            logger.error("load_dataset returned None or empty dataset.")
            return
        logger.info(f"Successfully loaded dataset with {len(full_dataset)} entries.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}\n{traceback.format_exc()}")
        return


    # --- K-Fold Setup ---
    k_folds = config.get("k_folds", 5)
    if k_folds < 2:
        logger.warning(f"k_folds set to {k_folds}. Disabling cross-validation. Using 80/20 train/val split.")
        from sklearn.model_selection import train_test_split
        indices = list(range(len(full_dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        train_val_splits = [(Subset(full_dataset, train_idx), Subset(full_dataset, val_idx))]
    else:
        logger.info(f"Setting up K-Fold cross-validation with k={k_folds}")
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        try:
             train_val_splits = [(Subset(full_dataset, train_idx), Subset(full_dataset, val_idx))
                                 for train_idx, val_idx in kf.split(range(len(full_dataset)))]
             logger.info(f"Created {len(train_val_splits)} train/validation splits.")
        except Exception as e:
            logger.error(f"Failed to create K-Fold splits: {e}\n{traceback.format_exc()}")
            return


    # --- Output Directory ---
    base_output_dir = config.get("base_output_dir", "/home/iris/Documents/deep_learning/experiments/sdxl_t5_refiner")
    if accelerator.is_main_process:
        os.makedirs(base_output_dir, exist_ok=True)
        logger.info(f"Base output directory: {base_output_dir}")


    # --- Hyperparameter Grid ---
    # Ensure default values are lists for product
    param_grid = {
        'learning_rate': config.get("learning_rate", [5e-8]),
        'lora_r': config.get("lora_r", [8]),
        'apply_lora_unet': config.get("apply_lora_unet", [True]),
        'apply_lora_refiner': config.get("apply_lora_refiner", [True]),
        'apply_lora_text_encoder': config.get("apply_lora_text_encoder", [False]),
        'epochs': config.get("epochs", [5]),
        'batch_size': config.get("batch_size", [1]), # This is per-device batch size
        'lora_alpha': config.get("lora_alpha", [16]),
        'lora_dropout': config.get("lora_dropout", [0.1])
    }
    # Convert single values to lists for product
    for key in param_grid:
        if not isinstance(param_grid[key], list):
            param_grid[key] = [param_grid[key]]

    keys, values = zip(*param_grid.items())
    hyperparam_configs = [dict(zip(keys, v)) for v in product(*values)]
    logger.info(f"Generated {len(hyperparam_configs)} hyperparameter configurations to test.")


    # --- Training Loop ---
    performance_records = []
    model_name = "sdxl_t5_refiner" # Model identifier
    overall_best_avg_kfold_loss = float('inf')
    best_performing_config_info = None

    for idx, hyperparams in enumerate(hyperparam_configs):
        config_name = f"hyperparam_config_{idx}"
        config_output_dir = os.path.join(base_output_dir, config_name)
        if accelerator.is_main_process:
            os.makedirs(config_output_dir, exist_ok=True)

        logger.info(f"--- Running {config_name} ({idx+1}/{len(hyperparam_configs)}) ---")
        logger.info(f"Hyperparameters: {hyperparams}")

        # Instantiate the model trainer FOR EACH hyperparameter config
        # This ensures fresh model weights are loaded each time
        finetuner = FinetuneModel(model_name, config_output_dir, accelerator, logger_instance=logger)

        try:
            # 1. Load base model weights
            logger.info("Loading base model...")
            finetuner.load_model()

            # 2. Modify architecture (Apply LoRA adapters)
            logger.info("Modifying architecture (applying LoRA if configured)...")
            finetuner.modify_architecture(
                apply_lora_to_unet=hyperparams['apply_lora_unet'],
                apply_lora_to_refiner=hyperparams['apply_lora_refiner'],
                apply_lora_to_text_encoder=hyperparams['apply_lora_text_encoder'],
                lora_r=hyperparams['lora_r'],
                lora_alpha=hyperparams['lora_alpha'],
                lora_dropout=hyperparams['lora_dropout']
            )

            # 3. Run Fine-tuning with K-Fold Cross-Validation
            logger.info("Starting K-Fold fine-tuning...")
            avg_kfold_val_loss, _, best_fold_hyperparams, best_fold_epoch = finetuner.fine_tune(
                dataset_path=dataset_path,
                train_val_splits=train_val_splits, # Pass the generated splits
                epochs=hyperparams['epochs'],
                batch_size=hyperparams['batch_size'], # Per-device batch size
                learning_rate=float(hyperparams['learning_rate']),
                gradient_accumulation_steps=gradient_accumulation_steps,
                lora_r=hyperparams['lora_r'],
                lora_alpha=hyperparams['lora_alpha'],
                lora_dropout=hyperparams['lora_dropout']
            )

            # Record performance on the main process
            if accelerator.is_main_process:
                 record = {
                    'config_idx': idx,
                    'config_name': config_name,
                    'hyperparameters': hyperparams,
                    'avg_kfold_val_loss': avg_kfold_val_loss if not np.isnan(avg_kfold_val_loss) else 'NaN',
                    'fold_losses': finetuner.fold_val_losses, # Best loss from each fold
                    # 'best_fold_epoch': best_fold_epoch # Epoch corresponding to the single best fold overall
                 }
                 performance_records.append(record)
                 logger.info(f"Finished run for {config_name}. Avg K-Fold Val Loss: {avg_kfold_val_loss:.4f}")

                 # Check if this config is the best overall based on avg k-fold loss
                 if not np.isnan(avg_kfold_val_loss) and avg_kfold_val_loss < overall_best_avg_kfold_loss:
                     overall_best_avg_kfold_loss = avg_kfold_val_loss
                     best_performing_config_info = record # Save the entire record
                     logger.info(f"*** New overall best performance found! Config {idx}, Avg K-Fold Loss: {avg_kfold_val_loss:.4f} ***")
                     # Note: Saving of the "best" model happens per-fold inside fine_tune based on that fold's val loss.
                     # Here we just track which hyperparam config performed best on average.

        except Exception as e:
            logger.error(f"Run FAILED for {config_name}: {e}\n{traceback.format_exc()}")
            # Record failure on main process
            if accelerator.is_main_process:
                performance_records.append({
                    'config_idx': idx,
                    'config_name': config_name,
                    'hyperparameters': hyperparams,
                    'avg_kfold_val_loss': 'FAILED',
                    'fold_losses': [],
                    'error': str(e)
                })

        finally:
            # Clean up GPU memory before next hyperparameter run
            del finetuner
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"--- Finished {config_name} ---")
            accelerator.wait_for_everyone() # Sync before next loop iteration


    # --- Save Summary ---
    if accelerator.is_main_process:
        summary_path = os.path.join(base_output_dir, "hyperparameter_performance_summary.json")
        logger.info(f"Saving performance summary to {summary_path}")
        try:
            # Convert numpy types to standard types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer): return int(obj)
                elif isinstance(obj, np.floating): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
                return obj

            serializable_records = convert_numpy_types(performance_records)
            with open(summary_path, 'w') as f:
                json.dump(serializable_records, f, indent=4)

            if best_performing_config_info:
                logger.info(f"--- Overall Best Performing Configuration ---")
                logger.info(f"  Config Index: {best_performing_config_info['config_idx']}")
                logger.info(f"  Config Name: {best_performing_config_info['config_name']}")
                logger.info(f"  Avg K-Fold Val Loss: {best_performing_config_info['avg_kfold_val_loss']:.4f}")
                logger.info(f"  Hyperparameters: {best_performing_config_info['hyperparameters']}")
                logger.info(f"  Best model weights saved within fold directories inside: {os.path.join(base_output_dir, best_performing_config_info['config_name'])}")
            else:
                logger.warning("No configuration completed successfully or achieved a valid validation loss.")

        except Exception as e:
             logger.error(f"Failed to save performance summary: {e}")


    logger.info("Finetuning script completed.")


if __name__ == "__main__":
    # Ensure the config path is correct
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found at {config_path}")
    else:
        print(f"Starting finetuning using config: {config_path}")
        run_finetune(config_path)