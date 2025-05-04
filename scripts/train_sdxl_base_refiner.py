import os
import torch
import logging
import yaml
import traceback
import numpy as np
import json
import math
import gc
import warnings
from accelerate import Accelerator
from accelerate.logging import get_logger
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    DiffusionPipeline,
    ConfigMixin,
)
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.utils import BaseOutput
from diffusers.configuration_utils import FrozenDict
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil

# Suppress FutureWarning from diffusers (if applicable)
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.models.transformers.transformer_2d")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from bitsandbytes.optim import AdamW8bit
except ImportError:
    print("Warning: bitsandbytes not found. Using torch.optim.AdamW.")
    AdamW8bit = torch.optim.AdamW

try:
    from src.utils.dataset import load_dataset, CocoFinetuneDataset
except ImportError:
    print("Warning: Could not import dataset utilities. Using dummy dataset.")
    class CocoFinetuneDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return {"image": f"dummy_{idx}.jpg", "prompt": "a dummy prompt"}
    def load_dataset(path, splits=None):
        print(f"Warning: Using dummy load_dataset for path: {path}")
        dummy_data = [{"id": i, "image": f"dummy_{i}.jpg", "prompt": f"dummy prompt {i}"} for i in range(200)]
        return CocoFinetuneDataset(dummy_data)

# --- Configuration ---
LOG_FILE = '/home/iris/Documents/deep_learning/src/logs/train_sdxl_with_refiner.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    filemode='w',
)

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        return None
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def load_hyperparam_config(hyperparam_config_path, logger):
    try:
        with open(hyperparam_config_path, 'r') as f:
            configs = json.load(f)
        best_config = None
        best_loss = float('inf')
        for config in configs:
            loss = config.get('avg_kfold_val_loss', 'NaN')
            if loss != 'NaN' and loss != 'FAILED' and isinstance(loss, (int, float)) and loss < best_loss:
                best_loss = loss
                best_config = config
        if best_config is None:
            logger.error("No valid configuration found with a finite avg_kfold_val_loss.")
            print("Error: No valid configuration found in hyperparameter performance summary.")
            return None
        default_hyperparams = {
            "learning_rate": 1e-7,
            "lora_r": 8,
            "apply_lora_unet": True,
            "apply_lora_refiner": True,
            "apply_lora_text_encoder": True,
            "epochs": 20,
            "batch_size": 1,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "validation_split": 0.2,
            "early_stopping_patience": 5,
            "generation_frequency": 1,
            "perceptual_loss_weight": 0.1
        }
        hyperparameters = best_config.get("hyperparameters", {})
        for key, value in default_hyperparams.items():
            if key not in hyperparameters:
                logger.warning(f"Missing hyperparameter '{key}' in best config. Using default: {value}")
                hyperparameters[key] = value
        if hyperparameters.get("apply_lora_text_encoder") is False:
            logger.warning("apply_lora_text_encoder is False in config, overriding to True to train projection layers.")
            hyperparameters["apply_lora_text_encoder"] = True
        hyperparam_config = {
            "config_idx": best_config.get("config_idx", 0),
            "config_name": best_config.get("config_name", "hyperparam_config_0"),
            "hyperparameters": hyperparameters,
            "avg_kfold_val_loss": best_config.get("avg_kfold_val_loss", float('inf')),
            "fold_losses": best_config.get("fold_losses", [])
        }
        required_keys = default_hyperparams.keys()
        missing_keys = [key for key in required_keys if key not in hyperparameters]
        if missing_keys:
            logger.error(f"Best hyperparameter config missing required keys: {missing_keys}. Using defaults for these.")
        return hyperparam_config
    except FileNotFoundError:
        logger.error(f"Hyperparameter configuration file not found: {hyperparam_config_path}")
        print(f"Error: Hyperparameter configuration file not found: {hyperparam_config_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading hyperparameter configuration: {e}\n{traceback.format_exc()}")
        print(f"Error loading hyperparameter configuration: {e}")
        return None

# --- Custom Pipeline ---
class StableDiffusionXLPipelineWithT5(DiffusionPipeline, ConfigMixin):
    config_name = "pipeline_config.json"
    _optional_components = ["refiner_unet"]

    def __init__(self,
                 vae,
                 text_encoder,
                 tokenizer,
                 unet,
                 scheduler,
                 projection_layer=None,
                 pool_projection_layer=None,
                 refiner_unet=None,
                 refiner_pool_projection_layer=None,
                 logger_instance=None):
        super().__init__()
        self.logger = logger_instance or get_logger(__name__)

        self.register_to_config(
            _class_name=self.__class__.__name__,
            _diffusers_version="Unknown",
            vae=(vae.__module__, vae.__class__.__name__),
            text_encoder=(text_encoder.__module__, text_encoder.__class__.__name__),
            tokenizer=(tokenizer.__module__, tokenizer.__class__.__name__),
            unet=(unet.__module__, unet.__class__.__name__),
            scheduler=(scheduler.__module__, scheduler.__class__.__name__),
            refiner_unet=(refiner_unet.__module__, refiner_unet.__class__.__name__) if refiner_unet else None,
        )

        self.pipeline_config = {
            "image_size": 512,
            "t5_hidden_size": 768,
            "unet_cross_attn_dim": 2048,
            "base_pooled_embed_dim": 1280,
            "refiner_pooled_embed_dim": 2560
        }

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            refiner_unet=refiner_unet,
        )

        if hasattr(self.tokenizer, 'model_max_length'):
            self.tokenizer.model_max_length = 512
        else:
            self.logger.warning("Tokenizer does not have 'model_max_length' attribute.")

        t5_hidden_size = self.pipeline_config.get("t5_hidden_size", 768)
        unet_cross_attn_dim = self.pipeline_config.get("unet_cross_attn_dim", 2048)
        base_pooled_embed_dim = self.pipeline_config.get("base_pooled_embed_dim", 1280)
        refiner_pooled_embed_dim = self.pipeline_config.get("refiner_pooled_embed_dim", 2560)

        if projection_layer is None:
            self.logger.info(f"Initializing Projection Layer T5({t5_hidden_size}) -> SDXL UNet({unet_cross_attn_dim})")
            self.projection_layer = torch.nn.Linear(t5_hidden_size, unet_cross_attn_dim)
            torch.nn.init.normal_(self.projection_layer.weight, mean=0.0, std=0.02)  # Changed from 0.001 to 0.02
            torch.nn.init.zeros_(self.projection_layer.bias)
        else:
            self.projection_layer = projection_layer

        if pool_projection_layer is None:
            self.logger.info(f"Initializing Pool Projection Layer T5({t5_hidden_size}) -> SDXL Base AddEmbs({base_pooled_embed_dim})")
            self.logger.info(f"Initializing Pool Projection Layer T5({t5_hidden_size}) -> SDXL Base AddEmbs({base_pooled_embed_dim})")
            self.pool_projection_layer = torch.nn.Linear(t5_hidden_size, base_pooled_embed_dim)
            torch.nn.init.normal_(self.pool_projection_layer.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(self.pool_projection_layer.bias)
        else:
            self.pool_projection_layer = pool_projection_layer

        if refiner_unet is not None:
            if refiner_pool_projection_layer is None:
                self.logger.info(f"Initializing Refiner Pool Projection Layer T5({t5_hidden_size}) -> SDXL Refiner AddEmbs({refiner_pooled_embed_dim})")
                self.refiner_pool_projection_layer = torch.nn.Linear(t5_hidden_size, refiner_pooled_embed_dim)
                torch.nn.init.normal_(self.pool_projection_layer.weight, mean=0.0, std=0.02)
                torch.nn.init.zeros_(self.pool_projection_layer.bias)
            else:
                self.refiner_pool_projection_layer = refiner_pool_projection_layer
            if self.refiner_pool_projection_layer.out_features != refiner_pooled_embed_dim:
                raise ValueError(f"refiner_pool_projection_layer output dimension {self.refiner_pool_projection_layer.out_features} does not match expected {refiner_pooled_embed_dim}")
        else:
            self.refiner_pool_projection_layer = None

        if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'block_out_channels'):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8
            self.logger.warning(f"Could not determine vae_scale_factor, using default: {self.vae_scale_factor}")

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if latents is None:
            latents = torch.randn(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device=device, dtype=dtype)

        latents = latents * self.scheduler.init_noise_sigma
        # latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
        self.logger.debug(f"Prepared latents with shape: {latents.shape}, device: {latents.device}, dtype: {latents.dtype}, mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
        return latents

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        if isinstance(prompt, str): prompt = [prompt]

        tokenizer_max_length = getattr(self.tokenizer, 'model_max_length', 512)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        with torch.no_grad():
            encoder_hidden_states_t5 = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask.to(device)
            )[0]
        # encoder_hidden_states_t5 = torch.nan_to_num(encoder_hidden_states_t5, nan=0.0, posinf=1.0, neginf=-1.0)
        self.logger.debug(f"T5 hidden states - mean: {encoder_hidden_states_t5.mean().item():.4f}, std: {encoder_hidden_states_t5.std().item():.4f}")

        if not hasattr(self, 'projection_layer') or self.projection_layer is None:
            raise AttributeError("projection_layer is required.")
        prompt_embeds = self.projection_layer(encoder_hidden_states_t5.to(dtype=self.projection_layer.weight.dtype))
        # prompt_embeds = torch.nan_to_num(prompt_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        self.logger.debug(f"Prompt embeds after projection - mean: {prompt_embeds.mean().item():.4f}, std: {prompt_embeds.std().item():.4f}")


        pooled_output_t5 = encoder_hidden_states_t5.mean(dim=1)
        if not hasattr(self, 'pool_projection_layer') or self.pool_projection_layer is None:
            raise AttributeError("pool_projection_layer is required.")
        base_text_embeds = self.pool_projection_layer(pooled_output_t5.to(dtype=self.pool_projection_layer.weight.dtype))
        # base_text_embeds = torch.nan_to_num(base_text_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        self.logger.debug(f"Base text_embeds shape: {base_text_embeds.shape}, mean: {base_text_embeds.mean().item():.4f}, std: {base_text_embeds.std().item():.4f}")
                
        if torch.isnan(prompt_embeds).any() or torch.isinf(prompt_embeds).any():
            self.logger.warning("NaN/Inf in prompt_embeds after projection.")
        base_text_embeds = self.pool_projection_layer(pooled_output_t5.to(dtype=self.pool_projection_layer.weight.dtype))
        self.logger.debug(f"Base text_embeds - mean: {base_text_embeds.mean().item():.4f}, std: {base_text_embeds.std().item():.4f}, min: {base_text_embeds.min().item():.4f}, max: {base_text_embeds.max().item():.4f}")
        if torch.isnan(base_text_embeds).any() or torch.isinf(base_text_embeds).any():
            self.logger.warning("NaN/Inf in base_text_embeds after projection.")

        refiner_text_embeds = None
        if self.refiner_unet is not None and self.refiner_pool_projection_layer is not None:
            refiner_text_embeds = self.refiner_pool_projection_layer(pooled_output_t5.to(dtype=self.refiner_pool_projection_layer.weight.dtype))
            # refiner_text_embeds = torch.nan_to_num(refiner_text_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
            expected_refiner_dim = self.pipeline_config["refiner_pooled_embed_dim"]
            if refiner_text_embeds.shape[-1] != expected_refiner_dim:
                raise ValueError(f"refiner_text_embeds dimension {refiner_text_embeds.shape[-1]} does not match expected {expected_refiner_dim}")
            self.logger.debug(f"Refiner text_embeds shape: {refiner_text_embeds.shape}, mean: {refiner_text_embeds.mean().item():.4f}, std: {refiner_text_embeds.std().item():.4f}")
        else:
            self.logger.debug("No refiner UNet or refiner_pool_projection_layer; using base text_embeds only.")

        image_size = self.pipeline_config.get("image_size", 512)
        original_size = (image_size, image_size)
        crops_coords_top_left = (0, 0)
        target_size = (image_size, image_size)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=prompt_embeds.dtype, device=device)
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            if isinstance(negative_prompt, str): negative_prompt = [negative_prompt] * batch_size

            uncond_tokens = self.tokenizer(
                negative_prompt, padding="max_length", max_length=tokenizer_max_length, truncation=True, return_tensors="pt",
            )
            uncond_input_ids = uncond_tokens.input_ids
            uncond_attention_mask = uncond_tokens.attention_mask

            with torch.no_grad():
                uncond_encoder_hidden_states_t5 = self.text_encoder(
                    uncond_input_ids.to(device), attention_mask=uncond_attention_mask.to(device)
                )[0]
            # uncond_encoder_hidden_states_t5 = torch.nan_to_num(uncond_encoder_hidden_states_t5, nan=0.0, posinf=1.0, neginf=-1.0)

            uncond_prompt_embeds = self.projection_layer(uncond_encoder_hidden_states_t5.to(dtype=self.projection_layer.weight.dtype))
            # uncond_prompt_embeds = torch.nan_to_num(uncond_prompt_embeds, nan=0.0, posinf=1.0, neginf=-1.0)

            uncond_pooled_output_t5 = uncond_encoder_hidden_states_t5.mean(dim=1)
            uncond_base_text_embeds = self.pool_projection_layer(uncond_pooled_output_t5.to(dtype=self.pool_projection_layer.weight.dtype))
            # uncond_base_text_embeds = torch.nan_to_num(uncond_base_text_embeds, nan=0.0, posinf=1.0, neginf=-1.0)

            uncond_refiner_text_embeds = None
            if self.refiner_unet is not None and self.refiner_pool_projection_layer is not None:
                uncond_refiner_text_embeds = self.refiner_pool_projection_layer(uncond_pooled_output_t5.to(dtype=self.refiner_pool_projection_layer.weight.dtype))
                # uncond_refiner_text_embeds = torch.nan_to_num(uncond_refiner_text_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
                if uncond_refiner_text_embeds.shape[-1] != expected_refiner_dim:
                    raise ValueError(f"uncond_refiner_text_embeds dimension {uncond_refiner_text_embeds.shape[-1]} does not match expected {expected_refiner_dim}")

            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
            base_text_embeds = torch.cat([uncond_base_text_embeds, base_text_embeds])
            if uncond_refiner_text_embeds is not None:
                refiner_text_embeds = torch.cat([uncond_refiner_text_embeds, refiner_text_embeds])
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        added_cond_kwargs = {
            "text_embeds": base_text_embeds,
            "time_ids": add_time_ids,
            "refiner_text_embeds": refiner_text_embeds if refiner_text_embeds is not None else base_text_embeds
        }
        self.logger.debug(f"Prompt embeds shape: {prompt_embeds.shape}, Base text_embeds shape: {base_text_embeds.shape}, Refiner text_embeds shape: {refiner_text_embeds.shape if refiner_text_embeds is not None else 'None'}")
        return prompt_embeds, added_cond_kwargs
    
    def debug_latents_to_image(self, latents, step, debug_dir):
        latents = latents.to(dtype=torch.float32, device=self.device)
        self.vae.to(dtype=torch.float32, device=self.device)
        latents_scaled = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents_scaled).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image_pil = self.numpy_to_pil(image)[0]
        output_path = os.path.join(debug_dir, f"latent_debug_step_{step}.png")
        image_pil.save(output_path)
        self.logger.info(f"Saved latent debug image at step {step} to {output_path}")


    @torch.no_grad()
    def __call__(self, prompt, negative_prompt=None, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1, generator=None, output_type="pil", refiner_steps=10, height=None, width=None, latents=None, return_dict=True, debug_dir=None, **kwargs):
        device = self.device
        dtype = self.unet.dtype if hasattr(self.unet, 'dtype') else torch.float32

        height = height or self.pipeline_config.get("image_size", 512)
        width = width or self.pipeline_config.get("image_size", 512)

        batch_size = len(prompt) if isinstance(prompt, list) else 1
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, added_cond_kwargs = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=negative_prompt
        )
        prompt_embeds = prompt_embeds.to(dtype)
        for k, v in added_cond_kwargs.items():
            if isinstance(v, torch.Tensor):
                added_cond_kwargs[k] = v.to(dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )


        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": added_cond_kwargs["text_embeds"], "time_ids": added_cond_kwargs["time_ids"]}
                ).sample

                if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                    self.logger.warning(f"Step {i+1}: NaN/Inf in noise_pred. Skipping update.")
                    continue

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    self.logger.warning(f"Step {i+1}: NaN/Inf in latents after update. Resetting latents.")
                    latents = torch.zeros_like(latents)

                if debug_dir and i % 10 == 0:
                    latents_np = latents.cpu().numpy()
                    latents_path = os.path.join(debug_dir, f"latents_step_{i}.npy")
                    np.save(latents_path, latents_np)
                    self.logger.info(f"Saved latents at step {i} to {latents_path}, mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
                    self.debug_latents_to_image(latents, i, "/home/iris/Documents/deep_learning/experiments/trained_sdxl_t5_refiner/debug")

                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                gc.collect()
                torch.cuda.empty_cache()

        if self.refiner_unet is not None and refiner_steps > 0:
            self.logger.info("Running refiner UNet for additional denoising steps.")
            self.scheduler.set_timesteps(refiner_steps, device=device)
            refiner_timesteps = self.scheduler.timesteps
            with self.progress_bar(total=refiner_steps) as progress_bar:
                for i, t in enumerate(refiner_timesteps):
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    noise_pred = self.refiner_unet(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": added_cond_kwargs["refiner_text_embeds"], "time_ids": added_cond_kwargs["time_ids"]}
                    ).sample

                    if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                        self.logger.warning(f"Refiner Step {i+1}: NaN/Inf in noise_pred. Skipping update.")
                        continue

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        self.logger.warning(f"Refiner Step {i+1}: NaN/Inf in latents after update. Resetting latents.")
                        latents = torch.zeros_like(latents)

                    if i == len(refiner_timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    gc.collect()
                    torch.cuda.empty_cache()

        image = latents
        if output_type != "latent":
            self.vae.to(dtype=torch.float32, device=self.device)
            latents = latents.to(dtype=torch.float32, device=self.device)
            self.logger.info(f"Latents before VAE decode - mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}, min: {latents.min().item():.4f}, max: {latents.max().item():.4f}")
            
            latents = latents / self.vae.config.scaling_factor #typically 0.18215 for SDXL instead of self.vae_scale_factor 
            self.vae.to(device=latents.device)
            with torch.no_grad():
                image = self.vae.decode(latents).sample
            self.logger.info(f"Decoded image - mean: {image.mean().item():.4f}, std: {image.std().item():.4f}, min: {image.min().item():.4f}, max: {image.max().item():.4f}")
            if debug_dir:
                image_np = image.cpu().numpy()
                image_path = os.path.join(debug_dir, "decoded_image.npy")
                np.save(image_path, image_np)
                self.logger.info(f"Saved decoded image to {image_path}")
            # Normalize to [0, 1]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()

            gc.collect()
            torch.cuda.empty_cache()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return BaseOutput(images=image)

# --- FinetuneModel Class ---
class FinetuneModel:
    def __init__(self, model_name, output_dir, accelerator: Accelerator, logger_instance=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.logger = logger_instance or get_logger(__name__)
        self.device = accelerator.device
        # if accelerator.mixed_precision == 'fp16':
        #     self.dtype = torch.float16

        if accelerator.mixed_precision == 'no':
            self.dtype = torch.float32
        else:
            self.dtype = torch.float32
        self.logger.info(f"Using dtype: {self.dtype} based on accelerator state.")

        self.tokenizer = None
        self.text_encoder = None
        self.unet = None
        self.refiner_unet = None
        self.scheduler = None
        self.vae = None
        self.pipeline = None
        self.image_size = 512
        self.current_epoch = 0
        self._apply_lora_unet_flag = False
        self._apply_lora_refiner_flag = False
        self._apply_lora_text_flag = False
        self.optimizer = None
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.perceptual_loss_fn = None
        self.logger.info("Perceptual loss disabled to reduce memory usage.")

    def load_model(self):
        try:
            t5_model_name = "t5-base"
            base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

            self.logger.info("Loading tokenizer...")
            self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            gc.collect()
            torch.cuda.empty_cache()

            self.logger.info("Loading text encoder...")
            self.text_encoder = T5EncoderModel.from_pretrained(t5_model_name).to(self.device, dtype=self.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            self.logger.info("Loading VAE...")
            self.vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae").to(self.device, dtype=torch.float32)
            self.vae.eval()
            gc.collect()
            torch.cuda.empty_cache()

            self.logger.info("Loading UNet...")
            self.unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet").to(self.device, dtype=self.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            self.logger.info("Loading Refiner UNet...")
            self.refiner_unet = UNet2DConditionModel.from_pretrained(refiner_model_id, subfolder="unet").to(self.device, dtype=self.dtype)
            # self.refiner_unet = None
            gc.collect()
            torch.cuda.empty_cache()

            self.logger.info("Loading scheduler...")
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")
            gc.collect()
            torch.cuda.empty_cache()

            self.logger.info("Attempting to enable gradient checkpointing...")
            try:
                self.text_encoder.gradient_checkpointing_enable()
                self.logger.info("Enabled gradient checkpointing for T5 Text Encoder.")
            except AttributeError:
                self.logger.warning("Could not enable gradient checkpointing for T5.")
            try:
                self.unet.enable_gradient_checkpointing()
                self.logger.info("Enabled gradient checkpointing for UNet.")
            except AttributeError:
                self.logger.warning("Could not enable gradient checkpointing for UNet.")
            try:
                self.refiner_unet.enable_gradient_checkpointing()
                self.logger.info("Enabled gradient checkpointing for Refiner UNet.")
            except AttributeError:
                self.logger.warning("Could not enable gradient checkpointing for Refiner UNet.")

            self.logger.info("Initializing pipeline...")
            self.pipeline = StableDiffusionXLPipelineWithT5(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.scheduler,
                refiner_unet=self.refiner_unet,
                logger_instance=self.logger
            )

            self.logger.info(f"Moving models to target device: {self.device} and dtype: {self.dtype}")
            if hasattr(self.pipeline, 'projection_layer') and self.pipeline.projection_layer is not None:
                self.pipeline.projection_layer.to(self.device, dtype=self.dtype)
                self.logger.info("Moved projection_layer to device and dtype.")
            if hasattr(self.pipeline, 'pool_projection_layer') and self.pipeline.pool_projection_layer is not None:
                self.pipeline.pool_projection_layer.to(self.device, dtype=self.dtype)
                self.logger.info("Moved pool_projection_layer to device and dtype.")
            if hasattr(self.pipeline, 'refiner_pool_projection_layer') and self.pipeline.refiner_pool_projection_layer is not None:
                self.pipeline.refiner_pool_projection_layer.to(self.device, dtype=self.dtype)
                self.logger.info("Moved refiner_pool_projection_layer to device and dtype.")

            self.unet = self.pipeline.unet
            self.text_encoder = self.pipeline.text_encoder
            self.vae = self.pipeline.vae
            self.scheduler = self.pipeline.scheduler
            self.tokenizer = self.pipeline.tokenizer
            self.refiner_unet = self.pipeline.refiner_unet

            self.logger.info("Model components loaded, pipeline initialized, and models moved to device.")
        except Exception as e:
            self.logger.error(f"Failed to load model components: {e}\n{traceback.format_exc()}")
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def modify_architecture(self, apply_lora_unet=True, apply_lora_refiner=True, apply_lora_text_encoder=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        self._apply_lora_unet_flag = apply_lora_unet
        self._apply_lora_refiner_flag = apply_lora_refiner
        self._apply_lora_text_flag = apply_lora_text_encoder

        lora_bias = "none"
        unet_target_modules = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
        lora_config_unet = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=lora_bias, target_modules=unet_target_modules)

        if apply_lora_unet and self.unet:
            self.logger.info("Applying LoRA to base UNet...")
            self.unet = get_peft_model(self.unet, lora_config_unet)
            self.pipeline.unet = self.unet
            self.unet.print_trainable_parameters()
            gc.collect()
            torch.cuda.empty_cache()
            
        if apply_lora_refiner and self.refiner_unet:
            self.logger.info("Applying LoRA to refiner UNet...")
            self.refiner_unet = get_peft_model(self.refiner_unet, lora_config_unet)
            self.pipeline.refiner_unet = self.refiner_unet
            self.refiner_unet.print_trainable_parameters()
            gc.collect()
            torch.cuda.empty_cache()

        if apply_lora_text_encoder and self.text_encoder:
            self.logger.info("Applying LoRA to T5 text encoder...")
            text_target_modules = ["q", "k", "v", "o"]
            lora_config_text = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=lora_bias, target_modules=text_target_modules)
            self.text_encoder = get_peft_model(self.text_encoder, lora_config_text)
            self.pipeline.text_encoder = self.text_encoder
            self.text_encoder.print_trainable_parameters()
            if hasattr(self.pipeline, 'projection_layer'):
                self.logger.info("Setting projection layer requires_grad=True")
                for param in self.pipeline.projection_layer.parameters():
                    param.requires_grad = True
            if hasattr(self.pipeline, 'pool_projection_layer'):
                self.logger.info("Setting pool projection layer requires_grad=True")
                for param in self.pipeline.pool_projection_layer.parameters():
                    param.requires_grad = True
            if hasattr(self.pipeline, 'refiner_pool_projection_layer'):
                self.logger.info("Setting refiner pool projection layer requires_grad=True")
                for param in self.pipeline.refiner_pool_projection_layer.parameters():
                    param.requires_grad = True
            gc.collect()
            torch.cuda.empty_cache()

    def save_model_state(self, epoch=None, train_loss=None, val_loss=None, hyperparameters=None, subdir=None, is_best=False):
        if not self.accelerator.is_main_process:
            return
        output_subdir = subdir if subdir else self.output_dir
        os.makedirs(output_subdir, exist_ok=True)
        train_loss_str = f"{train_loss:.4f}" if train_loss is not None and not np.isnan(train_loss) else "N/A"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None and not np.isnan(val_loss) else "N/A"
        if is_best:
            save_label = f"best_model_epoch_{epoch}_train_loss_{train_loss_str}_val_loss_{val_loss_str}"
        else:
            save_label = f"epoch_{epoch}_train_loss_{train_loss_str}_val_loss_{val_loss_str}"
        save_paths = {}

        if isinstance(self.unet, PeftModel):
            unet_path = os.path.join(output_subdir, f"{save_label}_unet_lora")
            try:
                self.accelerator.unwrap_model(self.unet).save_pretrained(unet_path)
                save_paths["UNet_LoRA"] = unet_path
                self.logger.info(f"Saved base UNet LoRA to {unet_path}")
            except Exception as e:
                self.logger.error(f"Failed to save UNet LoRA: {e}")

        if self.refiner_unet and isinstance(self.refiner_unet, PeftModel):
            refiner_path = os.path.join(output_subdir, f"{save_label}_refiner_unet_lora")
            try:
                self.accelerator.unwrap_model(self.refiner_unet).save_pretrained(refiner_path)
                save_paths["Refiner_UNet_LoRA"] = refiner_path
                self.logger.info(f"Saved refiner UNet LoRA to {refiner_path}")
            except Exception as e:
                self.logger.error(f"Failed to save Refiner UNet LoRA: {e}")

        if isinstance(self.text_encoder, PeftModel):
            text_encoder_path = os.path.join(output_subdir, f"{save_label}_text_encoder_lora")
            try:
                self.accelerator.unwrap_model(self.text_encoder).save_pretrained(text_encoder_path)
                save_paths["TextEncoder_LoRA"] = text_encoder_path
                self.logger.info(f"Saved TextEncoder LoRA to {text_encoder_path}")
            except Exception as e:
                self.logger.error(f"Failed to save Text Encoder LoRA: {e}")

        if hasattr(self.pipeline, 'projection_layer') and self._apply_lora_text_flag:
            proj_layer_path = os.path.join(output_subdir, f"{save_label}_projection_layer.pth")
            try:
                torch.save(self.accelerator.get_state_dict(self.pipeline.projection_layer), proj_layer_path)
                save_paths["Projection_Layer"] = proj_layer_path
                self.logger.info(f"Saved Projection Layer to {proj_layer_path}")
            except Exception as e:
                self.logger.error(f"Failed to save Projection Layer: {e}")

        if hasattr(self.pipeline, 'pool_projection_layer') and self._apply_lora_text_flag:
            pool_proj_layer_path = os.path.join(output_subdir, f"{save_label}_pool_projection_layer.pth")
            try:
                torch.save(self.accelerator.get_state_dict(self.pipeline.pool_projection_layer), pool_proj_layer_path)
                save_paths["Pool_Projection_Layer"] = pool_proj_layer_path
                self.logger.info(f"Saved Pool Projection Layer to {pool_proj_layer_path}")
            except Exception as e:
                self.logger.error(f"Failed to save Pool Projection Layer: {e}")

        if hasattr(self.pipeline, 'refiner_pool_projection_layer') and self._apply_lora_text_flag:
            refiner_pool_proj_layer_path = os.path.join(output_subdir, f"{save_label}_refiner_pool_projection_layer.pth")
            try:
                torch.save(self.accelerator.get_state_dict(self.pipeline.refiner_pool_projection_layer), refiner_pool_proj_layer_path)
                save_paths["Refiner_Pool_Projection_Layer"] = refiner_pool_proj_layer_path
                self.logger.info(f"Saved Refiner Pool Projection Layer to {refiner_pool_proj_layer_path}")
            except Exception as e:
                self.logger.error(f"Failed to save Refiner Pool Projection Layer: {e}")

        if hyperparameters:
            try:
                hyperparameters.update({
                    '_apply_lora_text_encoder': self._apply_lora_text_flag,
                    '_apply_lora_unet': self._apply_lora_unet_flag,
                    '_apply_lora_refiner': self._apply_lora_refiner_flag
                })
                hyperparam_path = os.path.join(output_subdir, f"{save_label}_hyperparameters.json")
                save_data = {'model_name': self.model_name, 'epoch': epoch, 'train_loss': train_loss_str, 'val_loss': val_loss_str, 'hyperparameters': hyperparameters}
                with open(hyperparam_path, 'w') as f:
                    json.dump(save_data, f, indent=4)
                self.logger.info(f"Saved hyperparameters to {hyperparam_path}")
            except Exception as e:
                self.logger.error(f"Failed to save hyperparameters: {e}")

        if not save_paths:
            self.logger.warning(f"No trainable weights saved for epoch {epoch}.")
        else:
            self.logger.info(f"Saved model components for epoch {epoch}: {save_paths}")

    def generate_validation_images(self, prompts, output_dir, epoch, use_pretrained=False):
        if not self.accelerator.is_main_process:
            return
        os.makedirs(output_dir, exist_ok=True)
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        self.logger.info(f"Generating validation images for epoch {epoch} (use_pretrained={use_pretrained})...")

        original_unet = self.pipeline.unet
        if use_pretrained:
            self.logger.info("Loading pre-trained UNet for validation image generation...")
            # Move fine-tuned UNet to CPU to free GPU memory
            self.pipeline.unet.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            self.pipeline.unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet").to(self.device, dtype=self.dtype)
            gc.collect()
            torch.cuda.empty_cache()

        for idx, prompt in enumerate(prompts[:3]):
            try:
                generator = torch.Generator(device=self.device).manual_seed(42 + idx)
                with torch.no_grad():
                    # Move text encoder to GPU for prompt encoding
                    self.text_encoder.to(self.device)
                    image = self.pipeline(
                        prompt=prompt,
                        negative_prompt="blurry, low quality, distorted",
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        refiner_steps=10,
                        height=self.image_size,
                        width=self.image_size,
                        generator=generator,
                        output_type="pil",
                        debug_dir=debug_dir
                    ).images[0]
                    # Move text encoder back to CPU
                    self.text_encoder.to("cpu")
                    torch.cuda.empty_cache()
                suffix = "pretrained" if use_pretrained else "finetuned"
                output_path = os.path.join(output_dir, f"val_image_epoch_{epoch}_prompt_{idx}_{suffix}.png")
                image.save(output_path)
                self.logger.info(f"Saved validation image: {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to generate validation image for prompt {idx}: {e}")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        if use_pretrained:
            # Move pretrained UNet to CPU and restore fine-tuned UNet
            self.pipeline.unet.to("cpu")
            self.pipeline.unet = original_unet.to(self.device, dtype=self.dtype)
            self.logger.info("Restored fine-tuned UNet after validation image generation.")
            gc.collect()
            torch.cuda.empty_cache()

    def validate(self, val_dataloader, image_folder, perceptual_loss_weight=0.1):
        self.logger.info("Running validation...")
        val_loss = 0.0
        num_val_batches = 0

        for model_component in [self.unet, self.refiner_unet, self.text_encoder]:
            if model_component:
                model_component.eval()

        for batch in val_dataloader:
            try:
                image_filenames = batch.get('image')
                prompts = batch.get('prompt')
                if image_filenames is None or prompts is None:
                    self.logger.warning("Validation batch missing 'image' or 'prompt'. Skipping.")
                    continue

                pixel_values_list = []
                valid_prompts = []
                for img_filename, prompt in zip(image_filenames, prompts):
                    if not isinstance(prompt, str) or not prompt.strip():
                        self.logger.warning(f"Validation: Invalid or empty prompt for image {img_filename}. Skipping item.")
                        continue
                    try:
                        image_path = os.path.join(image_folder, img_filename)
                        image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
                        image_np = np.array(image).astype(np.float32) / 255.0
                        if np.any(np.isnan(image_np)) or np.any(np.isinf(image_np)):
                            self.logger.warning(f"Validation: NaN/Inf in image {img_filename}. Skipping item.")
                            continue
                        #Ensures input images are strictly in [0, 1] and free of NaN/Inf before VAE encoding.
                        if np.any(image_np < 0) or np.any(image_np > 1):
                            self.logger.warning(f"Image {img_filename} has values outside [0, 1]. Clamping.")
                            image_np = np.clip(image_np, 0, 1)
                        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                        pixel_values_list.append(image_tensor)
                        valid_prompts.append(prompt)
                    except FileNotFoundError:
                        self.logger.warning(f"Validation: Image file not found {image_path}. Skipping item.")
                    except Exception as img_err:
                        self.logger.warning(f"Validation: Error loading image {img_filename}: {img_err}. Skipping item.")

                if not pixel_values_list:
                    self.logger.debug("Skipping validation batch: No valid image/prompt pairs.")
                    continue

                pixel_values = torch.stack(pixel_values_list).to(self.device, dtype=torch.float32)
                prompts = valid_prompts

                with torch.no_grad():
                    # Move text encoder to GPU for prompt encoding
                    self.text_encoder.to(self.device)
                    prompt_embeds, added_cond_kwargs = self.pipeline._encode_prompt(
                        prompts, self.device, 1, False
                    )
                    # Move text encoder back to CPU to free GPU memory
                    self.text_encoder.to("cpu")
                    torch.cuda.empty_cache()
                    prompt_embeds = prompt_embeds.to(self.dtype)
                    for k, v in added_cond_kwargs.items():
                        added_cond_kwargs[k] = v.to(self.dtype)

                    pixel_values_norm = pixel_values * 2.0 - 1.0
                    pixel_values_norm = torch.clamp(pixel_values_norm, -1.0, 1.0)
                    self.logger.debug(f"Pixel values norm - shape: {pixel_values_norm.shape}, dtype: {pixel_values_norm.dtype}, mean: {pixel_values_norm.mean().item():.4f}, std: {pixel_values_norm.std().item():.4f}, min: {pixel_values_norm.min().item():.4f}, max: {pixel_values_norm.max().item():.4f}")

                    # Move VAE to GPU for encoding
                    self.vae.to(self.device)
                    self.logger.debug(f"VAE device before encoding: {next(self.vae.parameters()).device}, VAE dtype: {self.vae.dtype}")
                    vae_output = self.vae.encode(pixel_values_norm.to(self.vae.dtype))
                    # Move VAE back to CPU to free GPU memory
                    self.vae.to("cpu")
                    torch.cuda.empty_cache()

                    latents = vae_output.latent_dist.sample()
                    # latents = latents * self.pipeline.vae_scale_factor
                    latents = latents * self.vae.config.scaling_factor
                    latents = torch.clamp(latents, -1e6, 1e6)
                    # latents = torch.nan_to_num(latents, nan=0.0, posinf=1e6, neginf=-1e6)
                    latents = latents.to(self.dtype)

                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        self.logger.warning("Validation: NaN/Inf in latents after clamping. Skipping batch.")
                        continue

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                    noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                    # noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0)

                    model_pred = self.unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                    if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                        self.logger.warning("Validation: NaN/Inf in model_pred. Skipping batch.")
                        continue
                    # model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1.0, neginf=-1.0)

                    mse_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    if torch.isnan(mse_loss) or torch.isinf(mse_loss):
                        self.logger.warning("Validation: MSE loss is NaN/Inf. Skipping batch.")
                        continue

                    total_loss = mse_loss

                    avg_loss = self.accelerator.gather(total_loss.repeat(latents.shape[0])).mean()
                    val_loss += avg_loss.item()
                    num_val_batches += 1

                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Validation step failed: {e}\n{traceback.format_exc()}")
                continue

        for model_component in [self.unet, self.refiner_unet, self.text_encoder]:
            if model_component:
                model_component.train()

        if num_val_batches == 0:
            self.logger.warning("No valid validation batches processed. Returning infinity for validation loss.")
            return float('inf')
        avg_val_loss = val_loss / num_val_batches
        self.logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def fine_tune(self, dataset_path, dataset, epochs=1, batch_size=1, learning_rate=1e-7, gradient_accumulation_steps=4, lora_r=8, lora_alpha=16, lora_dropout=0.1, validation_split=0.2, early_stopping_patience=2, generation_frequency=1, perceptual_loss_weight=0.1):
        self.logger.info(f"Requested learning rate: {learning_rate}")
        if learning_rate > 5e-5:
            self.logger.warning(f"Capping learning rate at 1e-8 for stability.")
            learning_rate = 5e-5

        try:
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            train_indices, val_indices = train_test_split(indices, test_size=validation_split, random_state=42)
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            self.logger.info(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False) #Eliminates GPU memory buffering for images
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
            image_folder = os.path.join(os.path.dirname(dataset_path), "images") if dataset_path else "images"

            params_to_optimize = []
            trainable_components = []
            projection_layer_added = False
            pool_projection_layer_added = False
            refiner_pool_projection_layer_added = False

            if self._apply_lora_unet_flag:
                params_to_optimize.extend(p for p in self.unet.parameters() if p.requires_grad)
                trainable_components.append(self.unet)
                self.logger.info("Adding UNet LoRA parameters.")

            if self._apply_lora_refiner_flag and self.refiner_unet is not None:
                params_to_optimize.extend(p for p in self.refiner_unet.parameters() if p.requires_grad)
                trainable_components.append(self.refiner_unet)
                self.logger.info("Adding Refiner UNet LoRA parameters.")

            if self._apply_lora_text_flag:
                params_to_optimize.extend(p for p in self.text_encoder.parameters() if p.requires_grad)
                trainable_components.append(self.text_encoder)
                self.logger.info("Adding Text Encoder LoRA parameters.")
                if hasattr(self.pipeline, 'projection_layer'):
                    self.logger.info("Setting projection layer requires_grad=True.")
                    for param in self.pipeline.projection_layer.parameters():
                        param.requires_grad = True
                    if not projection_layer_added:
                        params_to_optimize.extend(p for p in self.pipeline.projection_layer.parameters() if p.requires_grad)
                        trainable_components.append(self.pipeline.projection_layer)
                        projection_layer_added = True
                        self.logger.info("Adding Projection Layer parameters.")
                if hasattr(self.pipeline, 'pool_projection_layer'):
                    self.logger.info("Setting pool projection layer requires_grad=True.")
                    for param in self.pipeline.pool_projection_layer.parameters():
                        param.requires_grad = True
                    if not pool_projection_layer_added:
                        params_to_optimize.extend(p for p in self.pipeline.pool_projection_layer.parameters() if p.requires_grad)
                        trainable_components.append(self.pipeline.pool_projection_layer)
                        pool_projection_layer_added = True
                        self.logger.info("Adding Pool Projection Layer parameters.")
                if hasattr(self.pipeline, 'refiner_pool_projection_layer'):
                    self.logger.info("Setting refiner pool projection layer requires_grad=True.")
                    for param in self.pipeline.refiner_pool_projection_layer.parameters():
                        param.requires_grad = True
                    if not refiner_pool_projection_layer_added:
                        params_to_optimize.extend(p for p in self.pipeline.refiner_pool_projection_layer.parameters() if p.requires_grad)
                        trainable_components.append(self.pipeline.refiner_pool_projection_layer)
                        refiner_pool_projection_layer_added = True
                        self.logger.info("Adding Refiner Pool Projection Layer parameters.")
            elif not self._apply_lora_text_flag:
                if hasattr(self.pipeline, 'projection_layer'):
                    for param in self.pipeline.projection_layer.parameters():
                        param.requires_grad = False
                if hasattr(self.pipeline, 'pool_projection_layer'):
                    for param in self.pipeline.pool_projection_layer.parameters():
                        param.requires_grad = False
                if hasattr(self.pipeline, 'refiner_pool_projection_layer'):
                    for param in self.pipeline.refiner_pool_projection_layer.parameters():
                        param.requires_grad = False

            params_to_optimize = list({id(p): p for p in params_to_optimize}.values())
            self.logger.info(f"Total unique parameters to optimize: {len(params_to_optimize)}")

            if not params_to_optimize:
                self.logger.error("Optimizer params list is empty! Check LoRA flags / config.")
                return float('inf'), None, None, -1

            self.optimizer = AdamW8bit(params_to_optimize, lr=learning_rate)

            self.logger.info(f"Preparing {len(trainable_components)} model components, optimizer, and dataloaders with Accelerator.")
            prepare_list = trainable_components + [self.optimizer, train_dataloader, val_dataloader]
            prepared_components = self.accelerator.prepare(*prepare_list)

            num_models = len(trainable_components)
            prepared_models_tuple = prepared_components[:num_models]
            self.optimizer, train_dataloader, val_dataloader = prepared_components[num_models:]

            for i, original_component in enumerate(trainable_components):
                prepared_component = prepared_models_tuple[i]
                if original_component is self.unet:
                    self.unet = prepared_component
                    self.pipeline.unet = prepared_component
                elif original_component is self.refiner_unet:
                    self.refiner_unet = prepared_component
                    self.pipeline.refiner_unet = prepared_component
                elif original_component is self.text_encoder:
                    self.text_encoder = prepared_component
                    self.pipeline.text_encoder = prepared_component
                elif original_component is self.pipeline.projection_layer:
                    self.pipeline.projection_layer = prepared_component
                elif original_component is self.pipeline.pool_projection_layer:
                    self.pipeline.pool_projection_layer = prepared_component
                elif original_component is self.pipeline.refiner_pool_projection_layer:
                    self.pipeline.refiner_pool_projection_layer = prepared_component

            final_train_loss = float('inf')
            final_val_loss = float('inf')
            final_epoch = -1
            final_hyperparameters = None

            validation_prompts = [
                "A serene landscape with mountains and a river, sharp focus, highly detailed, 8k",
                "A futuristic cityscape at night with neon lights, intricate details, high resolution",
                "A cozy cabin in a snowy forest, clear, masterpiece, 8k"
            ]

            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                self.logger.info(f"--- Training Epoch {self.current_epoch} ---")

                max_train_steps = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
                global_step = 0
                train_loss_epoch = 0.0
                num_train_batches_epoch = 0
                num_skipped_steps = 0

                for model_component in prepared_models_tuple:
                    model_component.train()

                for step, batch in enumerate(train_dataloader):
                    try:
                        image_filenames = batch.get('image')
                        prompts = batch.get('prompt')
                        if image_filenames is None or prompts is None:
                            self.logger.warning(f"Train Step {step+1}: Batch missing 'image' or 'prompt'. Skipping.")
                            continue

                        pixel_values_list = []
                        valid_prompts = []
                        for img_filename, prompt in zip(image_filenames, prompts):
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
                                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                                pixel_values_list.append(image_tensor)
                                valid_prompts.append(prompt)
                            except FileNotFoundError:
                                self.logger.warning(f"Train Step {step+1}: Image file not found {image_path}. Skipping item.")
                            except Exception as img_err:
                                self.logger.warning(f"Train Step {step+1}: Error loading image {img_filename}: {img_err}. Skipping item.")

                        if not pixel_values_list:
                            self.logger.debug(f"Skipping train batch {step+1}: No valid image/prompt pairs.")
                            continue

                        pixel_values = torch.stack(pixel_values_list).to(self.device, dtype=torch.float32)
                        prompts = valid_prompts

                        with self.accelerator.accumulate(*prepared_models_tuple):
                            # Move T5 to GPU for prompt encoding
                            self.text_encoder.to(self.device)
                            with torch.amp.autocast(device_type='cuda', enabled=self.accelerator.mixed_precision in ['fp16', 'bf16']):
                                prompt_embeds, added_cond_kwargs = self.pipeline._encode_prompt(
                                    prompts, self.device, 1, False
                                )
                                prompt_embeds = prompt_embeds.to(self.dtype)
                                for k, v in added_cond_kwargs.items():
                                    added_cond_kwargs[k] = v.to(self.dtype)

                                # Move T5 to CPU after encoding
                                self.text_encoder.to("cpu")
                                torch.cuda.empty_cache()

                                if torch.isnan(prompt_embeds).any() or torch.isinf(prompt_embeds).any():
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in prompt_embeds. Skipping.")
                                    self.optimizer.zero_grad()
                                    continue
                                if torch.isnan(added_cond_kwargs["text_embeds"]).any() or torch.isinf(added_cond_kwargs["text_embeds"]).any():
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in added_cond_kwargs['text_embeds']. Skipping.")
                                    self.optimizer.zero_grad()
                                    continue

                                embed_reg_loss = 1e-4 * torch.norm(prompt_embeds, p=2)
                                embed_reg_loss += 1e-4 * torch.norm(added_cond_kwargs["text_embeds"], p=2)

                                if torch.isnan(embed_reg_loss) or torch.isinf(embed_reg_loss):
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in embed_reg_loss. Skipping.")
                                    self.optimizer.zero_grad()
                                    continue

                                pixel_values_norm = pixel_values * 2.0 - 1.0
                                pixel_values_norm = torch.clamp(pixel_values_norm, -1.0, 1.0)
                                with torch.no_grad():
                                    # Move VAE to GPU for encoding
                                    self.vae.to(self.device)
                                    pixel_values_norm = pixel_values_norm.to(torch.float32)
                                    self.logger.debug(f"Pixel values norm - mean: {pixel_values_norm.mean().item():.4f}, std: {pixel_values_norm.std().item():.4f}, min: {pixel_values_norm.min().item():.4f}, max: {pixel_values_norm.max().item():.4f}")
                                    if torch.isnan(pixel_values_norm).any() or torch.isinf(pixel_values_norm).any():
                                        self.logger.warning(f"Train Step {step+1}: NaN/Inf in pixel_values_norm. Skipping.")
                                        self.optimizer.zero_grad()
                                        continue

                                    vae_output = self.vae.encode(pixel_values_norm.to(torch.float32))
                                    self.logger.debug(f"latent_dist type: {type(vae_output.latent_dist)}, attributes: {dir(vae_output.latent_dist)}")
                                    self.logger.debug(f"VAE output latent_dist mean: {vae_output.latent_dist.mean.mean().item():.4f}, std: {vae_output.latent_dist.std.mean().item():.4f}")

                                    # Disable autocast for VAE encoding
                                    """Forces the VAE encoding to run in float32, bypassing any float16 casting from the autocast context.
                                        If this resolves the NaN/Inf, the issue is due to float16 precision in the VAE."""
                                    with torch.cuda.amp.autocast(enabled=False):
                                        vae_output = self.vae.encode(pixel_values_norm)
                                    self.logger.debug(f"VAE output latent_dist mean: {vae_output.latent_dist.mean.mean().item():.4f}, std: {vae_output.latent_dist.std.mean().item():.4f}")

                                    if torch.isnan(vae_output.latent_dist.mean).any() or torch.isinf(vae_output.latent_dist.mean).any():
                                        self.logger.warning(f"Train Step {step+1}: NaN/Inf in VAE latent_dist mean. Skipping.")
                                        self.optimizer.zero_grad()
                                        continue

                                    latents = vae_output.latent_dist.sample()
                                    self.logger.debug(f"Latents after sampling - mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}, min: {latents.min().item():.4f}, max: {latents.max().item():.4f}")
                                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                                        self.logger.warning(f"Train Step {step+1}: NaN/Inf in latents after sampling. Skipping.")
                                        self.optimizer.zero_grad()
                                        continue

                                    latents = torch.clamp(latents, -100, 100)  # Early clamping to prevent extreme values
                                    latents = latents * self.vae.config.scaling_factor
                                    self.logger.debug(f"Latents after scaling - mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}, min: {latents.min().item():.4f}, max: {latents.max().item():.4f}")
                                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                                        self.logger.warning(f"Train Step {step+1}: NaN/Inf in latents after scaling. Skipping.")
                                        self.optimizer.zero_grad()
                                        continue

                                    latents = torch.clamp(latents, -1e6, 1e6)
                                    self.logger.debug(f"Latents after clamping - mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}, min: {latents.min().item():.4f}, max: {latents.max().item():.4f}")
                                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                                        self.logger.warning(f"Train Step {step+1}: NaN/Inf in latents after clamping. Skipping.")
                                        self.optimizer.zero_grad()
                                        continue

                                    # latents = torch.nan_to_num(latents, nan=0.0, posinf=1e6, neginf=-1e6)
                                    latents = latents.to(self.dtype) # Convert to bfloat16 only after clamping
                                    self.logger.debug(f"Latents after type conversion - mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}, min: {latents.min().item():.4f}, max: {latents.max().item():.4f}")
                                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                                        self.logger.warning(f"Train Step {step+1}: NaN/Inf in latents after type conversion. Skipping.")
                                        self.optimizer.zero_grad()
                                        continue

                                    # Move VAE to CPU after encoding
                                    self.vae.to("cpu")
                                    torch.cuda.empty_cache()
                                                    
                                if torch.isnan(latents).any() or torch.isinf(latents).any():
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in latents after clamping. Skipping.")
                                    self.optimizer.zero_grad()
                                    continue

                                noise = torch.randn_like(latents)
                                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                                # noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0)

                                model_pred = self.unet(
                                    sample=noisy_latents,
                                    timestep=timesteps,
                                    encoder_hidden_states=prompt_embeds,
                                    added_cond_kwargs=added_cond_kwargs
                                ).sample
                                if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in base model_pred. Skipping.")
                                    self.optimizer.zero_grad()
                                    continue
                                # model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1.0, neginf=-1.0)

                                mse_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                                if torch.isnan(mse_loss) or torch.isinf(mse_loss):
                                    self.logger.warning(f"Train Step {step+1}: MSE loss is NaN/Inf. Skipping.")
                                    self.optimizer.zero_grad()
                                    continue

                                total_loss = mse_loss + embed_reg_loss

                            self.accelerator.backward(total_loss)

                            if self.accelerator.sync_gradients:
                                valid_gradients = True
                                try:
                                    grad_norm = self.accelerator.clip_grad_norm_(params_to_optimize, 5.0)
                                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                        self.logger.warning(f"Train Step {step+1}: NaN/Inf gradient norm ({grad_norm.item()}). Skipping step.")
                                        valid_gradients = False
                                    else:
                                        self.logger.debug(f"Train Step {step+1}: Gradient norm before clipping: {grad_norm.item():.4f}")
                                except Exception as clip_err:
                                    self.logger.warning(f"Train Step {step+1}: Error during gradient clipping: {clip_err}. Skipping step.")
                                    valid_gradients = False

                                if valid_gradients:
                                    self.optimizer.step()
                                    self.optimizer.zero_grad()
                                else:
                                    self.optimizer.zero_grad()
                                    num_skipped_steps += 1
                                global_step += 1

                            avg_loss = self.accelerator.gather(total_loss.repeat(latents.shape[0])).mean()
                            train_loss_epoch += avg_loss.item()
                            num_train_batches_epoch += 1

                            if self.accelerator.is_main_process and (global_step % 50 == 0 or step == len(train_dataloader) - 1):
                                self.logger.info(f"Epoch {self.current_epoch}, Step {global_step}/{max_train_steps}, Train Loss: {avg_loss.item():.4f}")

                        gc.collect()
                        torch.cuda.empty_cache()

                    except Exception as e:
                        self.logger.error(f"Training step {step+1} failed: {e}\n{traceback.format_exc()}")
                        try:
                            if self.accelerator.sync_gradients:
                                self.optimizer.zero_grad()
                        except Exception:
                            pass
                        continue

                avg_train_loss_epoch = train_loss_epoch / num_train_batches_epoch if num_train_batches_epoch > 0 else float('nan')
                self.logger.info(f"Epoch {self.current_epoch} Finished - Avg Train Loss: {avg_train_loss_epoch:.4f}, Skipped Opt Steps: {num_skipped_steps}")

                val_loss = self.validate(val_dataloader, image_folder, perceptual_loss_weight)

                if self.current_epoch % generation_frequency == 0:
                    val_image_dir = os.path.join(self.output_dir, f"val_images_epoch_{self.current_epoch}")
                    self.generate_validation_images(validation_prompts, val_image_dir, self.current_epoch, use_pretrained=False)
                    self.generate_validation_images(validation_prompts, val_image_dir, self.current_epoch, use_pretrained=True)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    save_dir = os.path.join(self.output_dir, "best_model")
                    self.logger.info(f"New best validation loss: {val_loss:.4f}. Saving best model...")
                    self.save_model_state(
                        epoch=self.current_epoch,
                        train_loss=avg_train_loss_epoch,
                        val_loss=val_loss,
                        hyperparameters={
                            'model_name': self.model_name,
                            'text_encoder': 't5-base',
                            'epochs': epochs,
                            'batch_size': batch_size * self.accelerator.num_processes,
                            'learning_rate': learning_rate,
                            'gradient_accumulation_steps': gradient_accumulation_steps,
                            'lora_r': lora_r,
                            'lora_alpha': lora_alpha,
                            'lora_dropout': lora_dropout,
                            'validation_split': validation_split,
                            'early_stopping_patience': early_stopping_patience,
                            'generation_frequency': generation_frequency,
                            'perceptual_loss_weight': perceptual_loss_weight
                        },
                        subdir=save_dir,
                        is_best=True
                    )
                else:
                    self.epochs_no_improve += 1
                    self.logger.info(f"No improvement in validation loss for {self.epochs_no_improve} epochs.")

                if self.epochs_no_improve >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {self.epochs_no_improve} epochs without improvement.")
                    break

                save_dir = os.path.join(self.output_dir, f"checkpoint_epoch_{self.current_epoch}")
                self.save_model_state(
                    epoch=self.current_epoch,
                    train_loss=avg_train_loss_epoch,
                    val_loss=val_loss,
                    hyperparameters={
                        'model_name': self.model_name,
                        'text_encoder': 't5-base',
                        'epochs': epochs,
                        'batch_size': batch_size * self.accelerator.num_processes,
                        'learning_rate': learning_rate,
                        'gradient_accumulation_steps': gradient_accumulation_steps,
                        'lora_r': lora_r,
                        'lora_alpha': lora_alpha,
                        'lora_dropout': lora_dropout,
                        'validation_split': validation_split,
                        'early_stopping_patience': early_stopping_patience,
                        'generation_frequency': generation_frequency,
                        'perceptual_loss_weight': perceptual_loss_weight
                    },
                    subdir=save_dir
                )

                final_train_loss = avg_train_loss_epoch
                final_val_loss = val_loss
                final_epoch = self.current_epoch
                final_hyperparameters = {
                    'model_name': self.model_name,
                    'text_encoder': 't5-base',
                    'epochs': epochs,
                    'batch_size': batch_size * self.accelerator.num_processes,
                    'learning_rate': learning_rate,
                    'gradient_accumulation_steps': gradient_accumulation_steps,
                    'lora_r': lora_r,
                    'lora_alpha': lora_alpha,
                    'lora_dropout': lora_dropout,
                    'validation_split': validation_split,
                    'early_stopping_patience': early_stopping_patience,
                    'generation_frequency': generation_frequency,
                    'perceptual_loss_weight': perceptual_loss_weight
                }

                self.accelerator.wait_for_everyone()
                gc.collect()
                torch.cuda.empty_cache()

            self.logger.info(f"Training Finished - Final Train Loss: {final_train_loss:.4f}, Final Val Loss: {final_val_loss:.4f} at Epoch {final_epoch}")
            return final_train_loss, final_val_loss, final_hyperparameters, final_epoch

        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}\n{traceback.format_exc()}")
            return float('inf'), float('inf'), None, -1

def run_finetune(config_path, hyperparam_config_path):
    accelerator = Accelerator(
        gradient_accumulation_steps=16,  # Reduced to 4 to minimize numerical issues in FP16 (fallback if BF16 is unsupported)
        mixed_precision= "no", #'fp16',  # in pretrained validation to prevent compatibility ikssue. previously Switch to BF16 for better numerical stability (requires compatible hardware)
        log_with='tensorboard' if 'tensorboard' in globals() else None,
        project_dir=os.path.join(
            "/home/iris/Documents/deep_learning/experiments/trained_sdxl_t5_refiner", "logs"
        )
    )
    logger = get_logger(__name__, log_level="INFO")
    print(f"Logging to {LOG_FILE}")

    config = load_config(config_path)
    if config is None:
        print("Exiting due to config load failure.")
        return

    hyperparam_config = load_hyperparam_config(hyperparam_config_path, logger)
    if hyperparam_config is None:
        print("Exiting due to hyperparameter config load failure.")
        return

    base_output_dir = config.get("base_output_dir_sdxl", "/home/iris/Documents/deep_learning/experiments/trained_sdxl_t5_refiner")
    project_dir = os.path.join(base_output_dir, "logs")

    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)
        print(f"Created project directory for logs: {project_dir}")

    try:
        import tensorboard
        print("TensorBoard found. Logging enabled.")
    except ImportError:
        print("TensorBoard not found. Skipping TensorBoard logging.")

    logger.info(f"Logging to {LOG_FILE} and accelerator handlers (if any).")
    logger.info(accelerator.state, main_process_only=False)

    dataset_path = config.get("dataset_path", "/home/iris/Documents/deep_learning/data/finetune_dataset/coco/dataset.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset JSON path not found: {dataset_path}")
        return
    logger.info(f"Loading dataset from: {dataset_path}")

    try:
        full_dataset = load_dataset(dataset_path)
        if full_dataset is None or len(full_dataset) == 0:
            logger.error("load_dataset returned None or empty dataset.")
            return
        logger.info(f"Successfully loaded dataset with {len(full_dataset)} entries.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}\n{traceback.format_exc()}")
        return

    if accelerator.is_main_process:
        os.makedirs(base_output_dir, exist_ok=True)
        logger.info(f"Base output directory: {base_output_dir}")

    config_name = hyperparam_config["config_name"]
    config_output_dir = os.path.join(base_output_dir, config_name)
    if accelerator.is_main_process:
        if os.path.exists(config_output_dir):
            shutil.rmtree(config_output_dir)
            logger.info(f"Cleared previous training directory: {config_output_dir}")
        os.makedirs(config_output_dir, exist_ok=True)

    logger.info(f"--- Running {config_name} ---")
    logger.info(f"Hyperparameters: {hyperparam_config['hyperparameters']}")

    model_name = "sdxl_t5_refiner"
    finetuner = FinetuneModel(model_name, config_output_dir, accelerator, logger_instance=logger)
    try:
        logger.info("Loading base model from scratch...")
        finetuner.load_model()
        logger.info("Modifying architecture (applying LoRA if configured)...")
        finetuner.modify_architecture(
            apply_lora_unet=hyperparam_config['hyperparameters']['apply_lora_unet'],
            apply_lora_refiner=hyperparam_config['hyperparameters']['apply_lora_refiner'],
            apply_lora_text_encoder=hyperparam_config['hyperparameters']['apply_lora_text_encoder'],
            lora_r=hyperparam_config['hyperparameters']['lora_r'],
            lora_alpha=hyperparam_config['hyperparameters']['lora_alpha'],
            lora_dropout=hyperparam_config['hyperparameters']['lora_dropout']
        )
        logger.info("Starting fine-tuning on full dataset...")
        final_train_loss, final_val_loss, final_hyperparams, final_epoch = finetuner.fine_tune(
            dataset_path=dataset_path,
            dataset=full_dataset,
            epochs=hyperparam_config['hyperparameters']['epochs'],
            batch_size=hyperparam_config['hyperparameters']['batch_size'],
            learning_rate=float(hyperparam_config['hyperparameters']['learning_rate']),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            lora_r=hyperparam_config['hyperparameters']['lora_r'],
            lora_alpha=hyperparam_config['hyperparameters']['lora_alpha'],
            lora_dropout=hyperparam_config['hyperparameters']['lora_dropout'],
            validation_split=hyperparam_config['hyperparameters']['validation_split'],
            early_stopping_patience=hyperparam_config['hyperparameters']['early_stopping_patience'],
            generation_frequency=hyperparam_config['hyperparameters']['generation_frequency'],
            perceptual_loss_weight=hyperparam_config['hyperparameters']['perceptual_loss_weight']
        )
        if accelerator.is_main_process:
            performance_record = {
                'config_idx': hyperparam_config['config_idx'],
                'config_name': config_name,
                'hyperparameters': hyperparam_config['hyperparameters'],
                'final_train_loss': float(final_train_loss) if not np.isnan(final_train_loss) else 'NaN',
                'final_val_loss': float(final_val_loss) if not np.isnan(final_val_loss) else 'NaN',
            }
            summary_path = os.path.join(base_output_dir, f"{config_name}_performance.json")
            logger.info(f"Saving performance summary to {summary_path}")
            with open(summary_path, 'w') as f:
                json.dump(performance_record, f, indent=4)
            logger.info(f"Finished run for {config_name}. Final Train Loss: {final_train_loss:.4f}, Final Val Loss: {final_val_loss:.4f}")
    except Exception as e:
        logger.error(f"Run FAILED for {config_name}: {e}\n{traceback.format_exc()}")
        performance_record = {
            'config_idx': hyperparam_config['config_idx'],
            'config_name': config_name,
            'hyperparameters': hyperparam_config['hyperparameters'],
            'final_train_loss': 'FAILED',
            'final_val_loss': 'FAILED',
            'error': str(e)
        }
        if accelerator.is_main_process:
            summary_path = os.path.join(base_output_dir, f"{config_name}_performance.json")
            with open(summary_path, 'w') as f:
                json.dump(performance_record, f, indent=4)
    finally:
        del finetuner
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"--- Finished {config_name} ---")
        accelerator.wait_for_everyone()

    logger.info("Finetuning script completed.")

if __name__ == "__main__":
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    hyperparam_config_path = "/home/iris/Documents/deep_learning/experiments/sdxl_t5_refiner/hyperparameter_performance_summary.json"
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found at {config_path}")
    elif not os.path.exists(hyperparam_config_path):
        print(f"ERROR: Hyperparameter configuration file not found at {hyperparam_config_path}")
    else:
        print(f"Starting finetuning using config: {config_path} and hyperparam config: {hyperparam_config_path}")
        run_finetune(config_path, hyperparam_config_path)