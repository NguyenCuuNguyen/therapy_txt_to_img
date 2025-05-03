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
logger = get_logger(__name__, log_level="INFO")
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
        logger.error(f"Error loading configuration: {e}\n{traceback.format_exc()}")
        print(f"Error loading configuration: {e}")
        return None

# --- Custom Pipeline ---
class StableDiffusionXLPipelineWithT5(DiffusionPipeline, ConfigMixin):
    config_name = "pipeline_config.json"
    _optional_components = ["refiner_unet"]

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: T5EncoderModel,
                 tokenizer: T5Tokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: DPMSolverMultistepScheduler,
                 projection_layer: torch.nn.Module = None,
                 pool_projection_layer: torch.nn.Module = None,
                 refiner_unet: UNet2DConditionModel = None,
                 logger_instance = None):
        super().__init__()
        self.logger = logger_instance or get_logger(__name__)

        # Initialize configuration for modules only
        try:
            diffusers_version = "Unknown"
            import diffusers
            diffusers_version = diffusers.__version__
        except ImportError:
            pass

        self.register_to_config(
            _class_name=self.__class__.__name__,
            _diffusers_version=diffusers_version,
            vae=(vae.__module__, vae.__class__.__name__),
            text_encoder=(text_encoder.__module__, text_encoder.__class__.__name__),
            tokenizer=(tokenizer.__module__, tokenizer.__class__.__name__),
            unet=(unet.__module__, unet.__class__.__name__),
            scheduler=(scheduler.__module__, scheduler.__class__.__name__),
            refiner_unet=(refiner_unet.__module__, refiner_unet.__class__.__name__) if refiner_unet else None,
        )

        # Store non-module configuration parameters separately
        self.pipeline_config = {
            "image_size": 1024,
            "t5_hidden_size": 768,
            "unet_cross_attn_dim": 2048,
            "pooled_embed_dim": 1280,
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
        pooled_embed_dim = self.pipeline_config.get("pooled_embed_dim", 1280)

        if projection_layer is None:
            self.logger.info(f"Initializing Projection Layer T5({t5_hidden_size}) -> SDXL UNet({unet_cross_attn_dim})")
            self.projection_layer = torch.nn.Linear(t5_hidden_size, unet_cross_attn_dim)
            torch.nn.init.normal_(self.projection_layer.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(self.projection_layer.bias)
        else:
            self.projection_layer = projection_layer

        if pool_projection_layer is None:
            self.logger.info(f"Initializing Pool Projection Layer T5({t5_hidden_size}) -> SDXL AddEmbs({pooled_embed_dim})")
            self.pool_projection_layer = torch.nn.Linear(t5_hidden_size, pooled_embed_dim)
            torch.nn.init.normal_(self.pool_projection_layer.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(self.pool_projection_layer.bias)
        else:
            self.pool_projection_layer = pool_projection_layer

        if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'block_out_channels'):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8
            self.logger.warning(f"Could not determine vae_scale_factor, using default: {self.vae_scale_factor}")

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
        encoder_hidden_states_t5 = torch.nan_to_num(encoder_hidden_states_t5, nan=0.0, posinf=1.0, neginf=-1.0)

        if not hasattr(self, 'projection_layer') or self.projection_layer is None:
            raise AttributeError("projection_layer is required.")
        prompt_embeds = self.projection_layer(encoder_hidden_states_t5.to(dtype=self.projection_layer.weight.dtype))
        prompt_embeds = torch.nan_to_num(prompt_embeds, nan=0.0, posinf=1.0, neginf=-1.0)

        pooled_output_t5 = encoder_hidden_states_t5.mean(dim=1)
        if not hasattr(self, 'pool_projection_layer') or self.pool_projection_layer is None:
            raise AttributeError("pool_projection_layer is required.")
        text_embeds = self.pool_projection_layer(pooled_output_t5.to(dtype=self.pool_projection_layer.weight.dtype))
        text_embeds = torch.nan_to_num(text_embeds, nan=0.0, posinf=1.0, neginf=-1.0)

        image_size = self.pipeline_config.get("image_size", 1024)
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
            uncond_encoder_hidden_states_t5 = torch.nan_to_num(uncond_encoder_hidden_states_t5, nan=0.0, posinf=1.0, neginf=-1.0)

            uncond_prompt_embeds = self.projection_layer(uncond_encoder_hidden_states_t5.to(dtype=self.projection_layer.weight.dtype))
            uncond_prompt_embeds = torch.nan_to_num(uncond_prompt_embeds, nan=0.0, posinf=1.0, neginf=-1.0)

            uncond_pooled_output_t5 = uncond_encoder_hidden_states_t5.mean(dim=1)
            uncond_text_embeds = self.pool_projection_layer(uncond_pooled_output_t5.to(dtype=self.pool_projection_layer.weight.dtype))
            uncond_text_embeds = torch.nan_to_num(uncond_text_embeds, nan=0.0, posinf=1.0, neginf=-1.0)

            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
            text_embeds = torch.cat([uncond_text_embeds, text_embeds])
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": add_time_ids}
        return prompt_embeds, added_cond_kwargs

    @torch.no_grad()
    def __call__(self, prompt, negative_prompt=None, num_inference_steps=50, guidance_scale=7.5, generator=None, output_type="pil", refiner_steps=10,
                 height=None, width=None, latents=None, return_dict=True, **kwargs):
        device = self.device
        dtype = torch.float32
        if hasattr(self.unet, 'dtype'):
            dtype = self.unet.dtype

        height = height or self.pipeline_config.get("image_size", 1024)
        width = width or self.pipeline_config.get("image_size", 1024)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, added_cond_kwargs = self._encode_prompt(
            prompt, device, 1, do_classifier_free_guidance, negative_prompt=negative_prompt
        )
        prompt_embeds = prompt_embeds.to(dtype)
        for k, v in added_cond_kwargs.items():
            if isinstance(v, torch.Tensor):
                added_cond_kwargs[k] = v.to(dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size, num_channels_latents, height, width, dtype, device, generator, latents,
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
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if self.refiner_unet is not None and refiner_steps > 0:
            self.logger.warning("Refiner UNet step skipped: Requires separate encoding/projection logic not implemented here.")

        image = latents
        if not output_type == "latent":
            self.vae.to(dtype=torch.float32)
            latents = latents.to(dtype=torch.float32)
            latents = latents / self.vae_scale_factor
            self.vae.to(device=latents.device)
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()

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
        if accelerator.mixed_precision == 'fp16':
            self.dtype = torch.float16
        elif accelerator.mixed_precision == 'bf16':
            self.dtype = torch.bfloat16
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
        self.image_size = 1024
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.current_epoch = 0
        self._apply_lora_unet_flag = False
        self._apply_lora_refiner_flag = False
        self._apply_lora_text_flag = False
        self.fold_val_losses = []
        self.optimizer = None

    def load_model(self):
        try:
            t5_model_name = "t5-base"
            base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

            self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            self.text_encoder = T5EncoderModel.from_pretrained(t5_model_name)
            self.vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
            self.refiner_unet = UNet2DConditionModel.from_pretrained(refiner_model_id, subfolder="unet")
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")

            self.logger.info("Attempting to enable gradient checkpointing...")
            try:
                self.text_encoder.gradient_checkpointing = True
                self.logger.info("Enabled gradient checkpointing for T5 Text Encoder.")
            except AttributeError: self.logger.warning("Could not enable GC for T5.")
            try:
                self.unet.enable_gradient_checkpointing()
                self.logger.info("Enabled gradient checkpointing for UNet.")
            except AttributeError: self.logger.warning("Could not enable GC for UNet.")
            try:
                self.refiner_unet.enable_gradient_checkpointing()
                self.logger.info("Enabled gradient checkpointing for Refiner UNet.")
            except AttributeError: self.logger.warning("Could not enable GC for Refiner UNet.")

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
            self.vae.to(self.device, dtype=torch.float32)
            self.vae.eval()
            self.pipeline.to(self.device, dtype=self.dtype)

            # Manually move non-registered Linear layers
            if hasattr(self.pipeline, 'projection_layer') and self.pipeline.projection_layer is not None:
                self.pipeline.projection_layer.to(self.device)
                for param in self.pipeline.projection_layer.parameters():
                    param.data = param.data.to(dtype=self.dtype)
                self.logger.info("Manually moved projection_layer to device and dtype.")
            if hasattr(self.pipeline, 'pool_projection_layer') and self.pipeline.pool_projection_layer is not None:
                self.pipeline.pool_projection_layer.to(self.device)
                for param in self.pipeline.pool_projection_layer.parameters():
                    param.data = param.data.to(dtype=self.dtype)
                self.logger.info("Manually moved pool_projection_layer to device and dtype.")

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

        if apply_lora_refiner and self.refiner_unet:
            self.logger.info("Applying LoRA to refiner UNet...")
            self.refiner_unet = get_peft_model(self.refiner_unet, lora_config_unet)
            self.pipeline.refiner_unet = self.refiner_unet
            self.refiner_unet.print_trainable_parameters()

        if apply_lora_text_encoder and self.text_encoder:
            self.logger.info("Applying LoRA to T5 text encoder...")
            text_target_modules = ["q", "k", "v", "o"]
            lora_config_text = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=lora_bias, target_modules=text_target_modules)
            self.text_encoder = get_peft_model(self.text_encoder, lora_config_text)
            self.pipeline.text_encoder = self.text_encoder
            self.text_encoder.print_trainable_parameters()
            if hasattr(self.pipeline, 'projection_layer'):
                self.logger.info("Setting projection layer requires_grad=True")
                for param in self.pipeline.projection_layer.parameters(): param.requires_grad = True
            if hasattr(self.pipeline, 'pool_projection_layer'):
                self.logger.info("Setting pool projection layer requires_grad=True")
                for param in self.pipeline.pool_projection_layer.parameters(): param.requires_grad = True

    def validate(self, val_dataloader, dataset_path):
        component_check_list = [self.unet, self.pipeline, self.vae, self.scheduler, self.text_encoder]
        if not all(component_check_list):
            self.logger.error("Missing required components for validation")
            return float('inf')

        self.unet.eval()
        if self.refiner_unet: self.refiner_unet.eval()
        self.text_encoder.eval()
        if hasattr(self.pipeline, 'projection_layer'): self.pipeline.projection_layer.eval()
        if hasattr(self.pipeline, 'pool_projection_layer'): self.pipeline.pool_projection_layer.eval()
        self.vae.eval()

        total_val_loss = 0.0
        num_val_batches = 0
        image_folder = os.path.join(os.path.dirname(dataset_path), "images") if dataset_path else "images"

        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                try:
                    image_filenames = batch.get('image')
                    prompts = batch.get('prompt')
                    if image_filenames is None or prompts is None: continue

                    pixel_values_list = []
                    valid_prompts = []
                    for img_filename, prompt in zip(image_filenames, prompts):
                        if not isinstance(prompt, str) or not prompt.strip(): continue
                        try:
                            image_path = os.path.join(image_folder, img_filename)
                            image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
                            image_np = np.array(image).astype(np.float32) / 255.0
                            if np.any(np.isnan(image_np)) or np.any(np.isinf(image_np)): continue
                            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                            pixel_values_list.append(image_tensor)
                            valid_prompts.append(prompt)
                        except Exception as img_err:
                            self.logger.debug(f"Val Skip img {img_filename}: {img_err}")

                    if not pixel_values_list: continue
                    pixel_values = torch.stack(pixel_values_list).to(self.accelerator.device, dtype=torch.float32)
                    prompts = valid_prompts

                    prompt_embeds, added_cond_kwargs = self.pipeline._encode_prompt(
                        prompts, self.accelerator.device, 1, False
                    )
                    prompt_embeds = prompt_embeds.to(dtype=self.dtype)
                    for k,v in added_cond_kwargs.items(): added_cond_kwargs[k] = v.to(dtype=self.dtype)

                    pixel_values_norm = pixel_values * 2.0 - 1.0
                    pixel_values_norm = torch.clamp(pixel_values_norm, -1.0, 1.0)  # Ensure valid input range
                    vae_output = self.vae.encode(pixel_values_norm.to(self.vae.dtype))
                    latents = vae_output.latent_dist.sample() * self.pipeline.vae_scale_factor
                    latents = torch.clamp(latents, -1e6, 1e6)  # Prevent extreme values
                    latents = torch.nan_to_num(latents, nan=0.0, posinf=1e6, neginf=-1e6)
                    latents = latents.to(dtype=self.dtype)
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        self.logger.warning(f"Validation Step {step+1}: NaN/Inf in latents after clamping. Skipping.")
                        continue

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                    noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                    noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0)

                    model_pred = self.unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                    if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                        self.logger.warning(f"Validation Step {step+1}: NaN/Inf in base model_pred. Skipping.")
                        continue
                    model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1.0, neginf=-1.0)
                    base_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    val_loss = base_loss

                    if torch.isnan(val_loss) or torch.isinf(val_loss):
                        self.logger.warning(f"Validation Step {step+1}: Calculated loss is NaN/Inf. Skipping.")
                        continue

                    gathered_loss = self.accelerator.gather(val_loss.repeat(latents.shape[0]))
                    total_val_loss += gathered_loss.mean().item() * latents.shape[0]
                    num_val_batches += latents.shape[0]

                except Exception as val_step_err:
                    self.logger.error(f"Validation step {step+1} failed: {val_step_err}\n{traceback.format_exc()}")
                    continue

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')

        if isinstance(self.unet, PeftModel): self.unet.train()
        if self.refiner_unet and isinstance(self.refiner_unet, PeftModel): self.refiner_unet.train()
        if isinstance(self.text_encoder, PeftModel): self.text_encoder.train()
        if hasattr(self.pipeline, 'projection_layer') and self._apply_lora_text_flag:
            self.pipeline.projection_layer.train()
        if hasattr(self.pipeline, 'pool_projection_layer') and self._apply_lora_text_flag:
            self.pipeline.pool_projection_layer.train()

        self.logger.info(f"Validation completed - Avg validation loss: {avg_val_loss:.4f}, Items: {num_val_batches}")
        return avg_val_loss

    def save_model_state(self, epoch=None, val_loss=None, hyperparameters=None, subdir=None):
        if not self.accelerator.is_main_process: return
        output_subdir = subdir if subdir else self.output_dir
        os.makedirs(output_subdir, exist_ok=True)
        save_label = "best"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None and not np.isnan(val_loss) else "N/A"
        save_paths = {}

        if isinstance(self.unet, PeftModel):
            unet_path = os.path.join(output_subdir, f"{save_label}_unet_lora")
            try:
                self.accelerator.unwrap_model(self.unet).save_pretrained(unet_path)
                save_paths["UNet_LoRA"] = unet_path
                self.logger.info(f"Saved base UNet LoRA to {unet_path}")
            except Exception as e: self.logger.error(f"Failed to save UNet LoRA: {e}")

        if self.refiner_unet and isinstance(self.refiner_unet, PeftModel):
            refiner_path = os.path.join(output_subdir, f"{save_label}_refiner_unet_lora")
            try:
                self.accelerator.unwrap_model(self.refiner_unet).save_pretrained(refiner_path)
                save_paths["Refiner_UNet_LoRA"] = refiner_path
                self.logger.info(f"Saved refiner UNet LoRA to {refiner_path}")
            except Exception as e: self.logger.error(f"Failed to save Refiner UNet LoRA: {e}")

        if isinstance(self.text_encoder, PeftModel):
            text_encoder_path = os.path.join(output_subdir, f"{save_label}_text_encoder_lora")
            try:
                self.accelerator.unwrap_model(self.text_encoder).save_pretrained(text_encoder_path)
                save_paths["TextEncoder_LoRA"] = text_encoder_path
                self.logger.info(f"Saved TextEncoder LoRA to {text_encoder_path}")
            except Exception as e: self.logger.error(f"Failed to save Text Encoder LoRA: {e}")

        if hasattr(self.pipeline, 'projection_layer') and self._apply_lora_text_flag:
            proj_layer_path = os.path.join(output_subdir, f"{save_label}_projection_layer.pth")
            try:
                torch.save(self.accelerator.get_state_dict(self.pipeline.projection_layer), proj_layer_path)
                save_paths["Projection_Layer"] = proj_layer_path
                self.logger.info(f"Saved Projection Layer to {proj_layer_path}")
            except Exception as e: self.logger.error(f"Failed to save Projection Layer: {e}")

        if hasattr(self.pipeline, 'pool_projection_layer') and self._apply_lora_text_flag:
            pool_proj_layer_path = os.path.join(output_subdir, f"{save_label}_pool_projection_layer.pth")
            try:
                torch.save(self.accelerator.get_state_dict(self.pipeline.pool_projection_layer), pool_proj_layer_path)
                save_paths["Pool_Projection_Layer"] = pool_proj_layer_path
                self.logger.info(f"Saved Pool Projection Layer to {pool_proj_layer_path}")
            except Exception as e: self.logger.error(f"Failed to save Pool Projection Layer: {e}")

        if hyperparameters:
            try:
                hyperparameters.update({
                    '_apply_lora_text_encoder': self._apply_lora_text_flag,
                    '_apply_lora_unet': self._apply_lora_unet_flag,
                    '_apply_lora_refiner': self._apply_lora_refiner_flag
                })
                hyperparam_path = os.path.join(output_subdir, f"{save_label}_hyperparameters.json")
                save_data = {'model_name': self.model_name, 'epoch': epoch, 'validation_loss': val_loss_str, 'hyperparameters': hyperparameters}
                with open(hyperparam_path, 'w') as f: json.dump(save_data, f, indent=4)
                self.logger.info(f"Saved hyperparameters to {hyperparam_path}")
            except Exception as e: self.logger.error(f"Failed to save hyperparameters: {e}")

        if not save_paths: self.logger.warning(f"No trainable weights saved for epoch {epoch}.")
        else: self.logger.info(f"Saved model components for epoch {epoch}: {save_paths}")

    def fine_tune(self, dataset_path, train_val_splits, epochs=1, batch_size=1, learning_rate=5e-8, gradient_accumulation_steps=8, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        self.logger.info(f"Requested learning rate: {learning_rate}")
        if learning_rate > 5e-8:
            self.logger.warning(f"Capping learning rate at 5e-8 for stability.")
            learning_rate = 5e-8

        self.fold_val_losses = []
        global_best_avg_val_loss = float('inf')
        global_best_hyperparameters = None
        global_best_epoch = -1

        for fold_idx, (train_dataset, val_dataset) in enumerate(train_val_splits):
            self.logger.info(f"--- Starting Fold {fold_idx+1}/{len(train_val_splits)} ---")
            self.best_val_loss = float('inf')
            self.best_epoch = -1
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            image_folder = os.path.join(os.path.dirname(dataset_path), "images") if dataset_path else "images"

            params_to_optimize = []
            trainable_components = []
            projection_layer_added = False
            pool_projection_layer_added = False

            if self._apply_lora_unet_flag and isinstance(self.unet, PeftModel):
                params_to_optimize.extend(p for p in self.unet.parameters() if p.requires_grad)
                trainable_components.append(self.unet)
                self.logger.info("Adding UNet LoRA parameters.")

            if self._apply_lora_refiner_flag and self.refiner_unet is not None and isinstance(self.refiner_unet, PeftModel):
                params_to_optimize.extend(p for p in self.refiner_unet.parameters() if p.requires_grad)
                trainable_components.append(self.refiner_unet)
                self.logger.info("Adding Refiner UNet LoRA parameters.")

            if self._apply_lora_text_flag and isinstance(self.text_encoder, PeftModel):
                params_to_optimize.extend(p for p in self.text_encoder.parameters() if p.requires_grad)
                trainable_components.append(self.text_encoder)
                self.logger.info("Adding Text Encoder LoRA parameters.")
                if hasattr(self.pipeline, 'projection_layer'):
                    self.logger.info("Setting projection layer requires_grad=True.")
                    for param in self.pipeline.projection_layer.parameters(): param.requires_grad = True
                    if not projection_layer_added:
                        params_to_optimize.extend(p for p in self.pipeline.projection_layer.parameters() if p.requires_grad)
                        trainable_components.append(self.pipeline.projection_layer)
                        projection_layer_added = True
                        self.logger.info("Adding Projection Layer parameters.")
                if hasattr(self.pipeline, 'pool_projection_layer'):
                    self.logger.info("Setting pool projection layer requires_grad=True.")
                    for param in self.pipeline.pool_projection_layer.parameters(): param.requires_grad = True
                    if not pool_projection_layer_added:
                        params_to_optimize.extend(p for p in self.pipeline.pool_projection_layer.parameters() if p.requires_grad)
                        trainable_components.append(self.pipeline.pool_projection_layer)
                        pool_projection_layer_added = True
                        self.logger.info("Adding Pool Projection Layer parameters.")
            elif not self._apply_lora_text_flag:
                if hasattr(self.pipeline, 'projection_layer'):
                    for param in self.pipeline.projection_layer.parameters(): param.requires_grad = False
                if hasattr(self.pipeline, 'pool_projection_layer'):
                    for param in self.pipeline.pool_projection_layer.parameters(): param.requires_grad = False

            params_to_optimize = list({id(p): p for p in params_to_optimize}.values())
            self.logger.info(f"Total unique parameters to optimize: {len(params_to_optimize)}")

            if not params_to_optimize:
                self.logger.error("Optimizer params list is empty! Check LoRA flags / config.")
                continue

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

            max_train_steps = epochs * math.ceil(len(train_dataloader) / gradient_accumulation_steps)
            global_step = 0
            self.logger.info(f"Fold {fold_idx+1}: Total optimization steps = {max_train_steps}")

            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                for model_component in prepared_models_tuple:
                    model_component.train()

                train_loss_epoch = 0.0
                num_train_batches_epoch = 0
                num_skipped_steps = 0

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

                        pixel_values = torch.stack(pixel_values_list).to(self.accelerator.device, dtype=torch.float32)
                        prompts = valid_prompts

                        with self.accelerator.accumulate(*prepared_models_tuple):
                            with torch.cuda.amp.autocast(enabled=self.accelerator.mixed_precision == 'fp16'):
                                prompt_embeds, added_cond_kwargs = self.pipeline._encode_prompt(
                                    prompts, self.accelerator.device, 1, False
                                )
                                prompt_embeds = prompt_embeds.to(dtype=self.dtype)
                                for k,v in added_cond_kwargs.items(): added_cond_kwargs[k] = v.to(dtype=self.dtype)

                                pixel_values_norm = pixel_values * 2.0 - 1.0
                                pixel_values_norm = torch.clamp(pixel_values_norm, -1.0, 1.0)  # Ensure valid input range
                                with torch.no_grad():
                                    vae_output = self.vae.encode(pixel_values_norm.to(self.vae.dtype))
                                    latents = vae_output.latent_dist.sample()
                                    latents = latents * self.pipeline.vae_scale_factor
                                    latents = torch.clamp(latents, -1e6, 1e6)  # Prevent extreme values
                                    latents = torch.nan_to_num(latents, nan=0.0, posinf=1e6, neginf=-1e6)
                                    latents = latents.to(dtype=self.dtype)
                                if torch.isnan(latents).any() or torch.isinf(latents).any():
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in latents after clamping. Skipping.")
                                    self.optimizer.zero_grad()
                                    continue

                                noise = torch.randn_like(latents)
                                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                                noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0)

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
                                model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1.0, neginf=-1.0)

                                base_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                                loss = base_loss

                                if torch.isnan(loss) or torch.isinf(loss):
                                    self.logger.warning(f"Train Step {step+1}: Calculated loss is NaN/Inf. Skipping.")
                                    self.optimizer.zero_grad()
                                    continue

                            self.accelerator.backward(loss)

                            if self.accelerator.sync_gradients:
                                valid_gradients = True
                                try:
                                    grad_norm = self.accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                        self.logger.warning(f"Train Step {step+1}: NaN/Inf gradient norm ({grad_norm.item()}). Skipping step.")
                                        valid_gradients = False
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

                            avg_loss = self.accelerator.gather(loss.repeat(latents.shape[0])).mean()
                            train_loss_epoch += avg_loss.item()
                            num_train_batches_epoch += 1

                            if self.accelerator.is_main_process and (global_step % 50 == 0 or step == len(train_dataloader) - 1):
                                self.logger.info(f"Fold {fold_idx+1}, Epoch {self.current_epoch}, Step {global_step}/{max_train_steps}, Train Loss: {avg_loss.item():.4f}")

                    except Exception as e:
                        self.logger.error(f"Training step {step+1} failed: {e}\n{traceback.format_exc()}")
                        try:
                            if self.accelerator.sync_gradients: self.optimizer.zero_grad()
                        except Exception: pass
                        continue

                avg_train_loss_epoch = train_loss_epoch / num_train_batches_epoch if num_train_batches_epoch > 0 else float('nan')
                self.logger.info(f"Fold {fold_idx+1}, Epoch {self.current_epoch} Finished - Avg Train Loss: {avg_train_loss_epoch:.4f}, Skipped Opt Steps: {num_skipped_steps}")

                if self.accelerator.is_main_process:
                    self.logger.info(f"Running validation for Fold {fold_idx+1}, Epoch {self.current_epoch}...")
                    avg_val_loss = self.validate(val_dataloader, dataset_path)
                    if not np.isnan(avg_val_loss) and avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self.best_epoch = self.current_epoch
                        save_dir = os.path.join(self.output_dir, f"fold_{fold_idx+1}_best_model")
                        self.logger.info(f"*** New best validation loss for Fold {fold_idx+1}: {avg_val_loss:.4f} at Epoch {self.current_epoch} ***")
                        self.save_model_state(epoch=self.current_epoch, val_loss=avg_val_loss, hyperparameters={
                            'model_name': self.model_name,
                            'text_encoder': 't5-base',
                            'epochs': epochs,
                            'batch_size': batch_size * self.accelerator.num_processes,
                            'learning_rate': learning_rate,
                            'gradient_accumulation_steps': gradient_accumulation_steps,
                            'lora_r': lora_r,
                            'lora_alpha': lora_alpha,
                            'lora_dropout': lora_dropout
                        }, subdir=save_dir)
                    else:
                        self.logger.info(f"Validation loss ({avg_val_loss:.4f}) did not improve from best ({self.best_val_loss:.4f})")

                self.accelerator.wait_for_everyone()
                gc.collect()
                torch.cuda.empty_cache()

            self.fold_val_losses.append(self.best_val_loss)
            if self.accelerator.is_main_process:
                self.logger.info(f"Fold {fold_idx+1} finished. Best Validation Loss: {self.best_val_loss:.4f} at Epoch {self.best_epoch}")
                if self.best_val_loss < global_best_avg_val_loss:
                    global_best_avg_val_loss = self.best_val_loss
                    global_best_hyperparameters = {
                        'model_name': self.model_name,
                        'text_encoder': 't5-base',
                        'epochs': epochs,
                        'batch_size': batch_size * self.accelerator.num_processes,
                        'learning_rate': learning_rate,
                        'gradient_accumulation_steps': gradient_accumulation_steps,
                        'lora_r': lora_r,
                        'lora_alpha': lora_alpha,
                        'lora_dropout': lora_dropout
                    }
                    global_best_epoch = self.best_epoch

        valid_losses = [loss for loss in self.fold_val_losses if loss != float('inf') and not np.isnan(loss)]
        avg_kfold_val_loss = np.mean(valid_losses) if valid_losses else float('inf')
        self.logger.info(f"K-Fold Training Finished - Avg Best Validation Loss across {len(valid_losses)} valid folds: {avg_kfold_val_loss:.4f}")

        return avg_kfold_val_loss, None, global_best_hyperparameters, global_best_epoch

# --- Main Execution ---
def run_finetune(config_path):
    config = load_config(config_path)
    if config is None:
        print("Exiting due to config load failure.")
        return

    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 8)
    mixed_precision = config.get("mixed_precision", 'fp16')
    output_dir = config.get("base_output_dir", "./output")
    project_dir = os.path.join(output_dir, "logs")

    if not os.path.exists(project_dir):
        os.makedirs(project_dir, exist_ok=True)
        print(f"Created project directory for logs: {project_dir}")

    try:
        import tensorboard
        log_with = "tensorboard"
        print("TensorBoard found. Logging enabled.")
    except ImportError:
        log_with = None
        print("TensorBoard not found. Skipping TensorBoard logging.")

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with,
        project_dir=project_dir
    )

    global logger
    logger = get_logger(__name__, log_level="INFO")
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

    base_output_dir = config.get("base_output_dir_sdxl", "/home/iris/Documents/deep_learning/experiments/sdxl_t5_refiner")
    if accelerator.is_main_process:
        os.makedirs(base_output_dir, exist_ok=True)
        logger.info(f"Base output directory: {base_output_dir}")

    param_grid = {
        'learning_rate': config.get("learning_rate", [5e-8]),
        'lora_r': config.get("lora_r", [8]),
        'apply_lora_unet': config.get("apply_lora_unet", [True]),
        'apply_lora_refiner': config.get("apply_lora_refiner", [True]),
        'apply_lora_text_encoder': config.get("apply_lora_text_encoder", [False]),
        'epochs': config.get("epochs", [5]),
        'batch_size': config.get("batch_size", [1]),
        'lora_alpha': config.get("lora_alpha", [16]),
        'lora_dropout': config.get("lora_dropout", [0.1])
    }
    for key in param_grid:
        if not isinstance(param_grid[key], list):
            param_grid[key] = [param_grid[key]]

    keys, values = zip(*param_grid.items())
    hyperparam_configs = [dict(zip(keys, v)) for v in product(*values)]
    logger.info(f"Generated {len(hyperparam_configs)} hyperparameter configurations to test.")

    performance_records = []
    model_name = "sdxl_t5_refiner"
    overall_best_avg_kfold_loss = float('inf')
    best_performing_config_info = None

    for idx, hyperparams in enumerate(hyperparam_configs):
        config_name = f"hyperparam_config_{idx}"
        config_output_dir = os.path.join(base_output_dir, config_name)
        if accelerator.is_main_process:
            os.makedirs(config_output_dir, exist_ok=True)

        logger.info(f"--- Running {config_name} ({idx+1}/{len(hyperparam_configs)}) ---")
        logger.info(f"Hyperparameters: {hyperparams}")

        finetuner = FinetuneModel(model_name, config_output_dir, accelerator, logger_instance=logger)
        try:
            logger.info("Loading base model...")
            finetuner.load_model()
            logger.info("Modifying architecture (applying LoRA if configured)...")
            finetuner.modify_architecture(
                apply_lora_unet=hyperparams['apply_lora_unet'],
                apply_lora_refiner=hyperparams['apply_lora_refiner'],
                apply_lora_text_encoder=hyperparams['apply_lora_text_encoder'],
                lora_r=hyperparams['lora_r'],
                lora_alpha=hyperparams['lora_alpha'],
                lora_dropout=hyperparams['lora_dropout']
            )
            logger.info("Starting K-Fold fine-tuning...")
            avg_kfold_val_loss, _, best_fold_hyperparams, best_fold_epoch = finetuner.fine_tune(
                dataset_path=dataset_path,
                train_val_splits=train_val_splits,
                epochs=hyperparams['epochs'],
                batch_size=hyperparams['batch_size'],
                learning_rate=float(hyperparams['learning_rate']),
                gradient_accumulation_steps=gradient_accumulation_steps,
                lora_r=hyperparams['lora_r'],
                lora_alpha=hyperparams['lora_alpha'],
                lora_dropout=hyperparams['lora_dropout']
            )
            if accelerator.is_main_process:
                record = {
                    'config_idx': idx,
                    'config_name': config_name,
                    'hyperparameters': hyperparams,
                    'avg_kfold_val_loss': avg_kfold_val_loss if not np.isnan(avg_kfold_val_loss) else 'NaN',
                    'fold_losses': finetuner.fold_val_losses,
                }
                performance_records.append(record)
                logger.info(f"Finished run for {config_name}. Avg K-Fold Val Loss: {avg_kfold_val_loss:.4f}")
                if not np.isnan(avg_kfold_val_loss) and avg_kfold_val_loss < overall_best_avg_kfold_loss:
                    overall_best_avg_kfold_loss = avg_kfold_val_loss
                    best_performing_config_info = record
                    logger.info(f"*** New overall best performance found! Config {idx}, Avg K-Fold Loss: {avg_kfold_val_loss:.4f} ***")
        except Exception as e:
            logger.error(f"Run FAILED for {config_name}: {e}\n{traceback.format_exc()}")
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
            del finetuner
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"--- Finished {config_name} ---")
            accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        summary_path = os.path.join(base_output_dir, "hyperparameter_performance_summary.json")
        logger.info(f"Saving performance summary to {summary_path}")
        try:
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
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found at {config_path}")
    else:
        print(f"Starting finetuning using config: {config_path}")
        run_finetune(config_path)