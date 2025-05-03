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
class StableDiffusionXLPipelineWithT5(StableDiffusionXLPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, logger_instance=None):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            text_encoder_2=None,
            tokenizer_2=None
        )
        self.logger = logger_instance or logging.getLogger(__name__)
        self.tokenizer.model_max_length = 512
        self.projection_layer = torch.nn.Linear(768, 1280).to(device=self.unet.device, dtype=self.unet.dtype)
        torch.nn.init.normal_(self.projection_layer.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.projection_layer.bias)
        self.refiner_unet = None  # Managed separately, not a pipeline component
        # Explicitly register only the required components
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            text_encoder_2=None,
            tokenizer_2=None
        )

    def set_refiner_unet(self, refiner_unet):
        """Set the refiner UNet without registering it as a pipeline component."""
        self.refiner_unet = refiner_unet
        if refiner_unet:
            self.refiner_unet.to(device=self.unet.device, dtype=self.unet.dtype)

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
            self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            self.tokenizer.model_max_length = 512
            self.text_encoder = T5EncoderModel.from_pretrained(t5_model_name, torch_dtype=self.dtype).to(self.device)
            self.text_encoder.gradient_checkpointing_enable()
            self.vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float32).to(self.device)
            self.vae.eval()
            self.unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=self.dtype).to(self.device)
            self.unet.enable_gradient_checkpointing()
            self.refiner_unet = UNet2DConditionModel.from_pretrained(refiner_model_id, subfolder="unet", torch_dtype=self.dtype).to(self.device)
            self.refiner_unet.enable_gradient_checkpointing()
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")
            self.pipeline = StableDiffusionXLPipelineWithT5(
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=self.scheduler,
                logger_instance=self.logger
            )
            self.pipeline.set_refiner_unet(self.refiner_unet)
            self.pipeline.to(self.device)
            self.pipeline.projection_layer.to(self.device, dtype=self.dtype)
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
            params_to_optimize = []
            if self._apply_lora_unet_flag and isinstance(self.unet, PeftModel):
                params_to_optimize.extend(self.unet.parameters())
            if self._apply_lora_refiner_flag and isinstance(self.refiner_unet, PeftModel):
                params_to_optimize.extend(self.refiner_unet.parameters())
            if self._apply_lora_text_flag and isinstance(self.text_encoder, PeftModel):
                params_to_optimize.extend(self.text_encoder.parameters())
            if self.pipeline.projection_layer:
                params_to_optimize.extend(self.pipeline.projection_layer.parameters())
            if not params_to_optimize:
                self.logger.error("Optimizer params list is empty!")
                continue
            params_to_optimize = list({id(p): p for p in params_to_optimize}.values())
            optimizer = AdamW8bit(params_to_optimize, lr=learning_rate)
            prepare_list = [self.unet, self.refiner_unet, optimizer, train_dataloader, val_dataloader]
            prepared_components = self.accelerator.prepare(*prepare_list)
            self.unet, self.refiner_unet, self.optimizer, train_dataloader, val_dataloader = prepared_components
            self.pipeline.to(self.accelerator.device)
            self.vae.to(self.accelerator.device)
            self.vae.eval()
            hyperparameters = {
                'model_name': self.model_name,
                'text_encoder': 't5-base',
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                'lora_dropout': lora_dropout
            }
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
            max_train_steps = epochs * num_update_steps_per_epoch
            global_step = 0
            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                self.unet.train()
                self.refiner_unet.train()
                self.text_encoder.train()
                self.pipeline.projection_layer.train()
                train_loss_epoch = 0.0
                num_train_batches_epoch = 0
                num_skipped_steps = 0
                for step, batch in enumerate(train_dataloader):
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
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in image {img_filename}")
                                    continue
                                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                                pixel_values_list.append(image_tensor)
                                valid_prompts.append(prompt)
                            except Exception as img_err:
                                self.logger.debug(f"Train Skip img {img_filename}: {img_err}")
                        if not pixel_values_list:
                            self.logger.debug(f"Skipping train batch {step+1}: No valid images")
                            continue
                        pixel_values = torch.stack(pixel_values_list).to(self.accelerator.device, dtype=torch.float32)
                        prompts = valid_prompts
                        with self.accelerator.accumulate(self.unet, self.refiner_unet):
                            with torch.cuda.amp.autocast():
                                prompt_embeds, added_cond_kwargs = self.pipeline._encode_prompt(
                                    prompts, self.accelerator.device, 1, False
                                )
                                pixel_values_norm = pixel_values * 2.0 - 1.0
                                vae_output = self.vae.encode(pixel_values_norm)
                                latents = vae_output.latent_dist.sample() * self.vae.config.scaling_factor
                                latents = latents.to(dtype=self.dtype)
                                if torch.isnan(latents).any() or torch.isinf(latents).any():
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in latents")
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
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in base model_pred")
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
                                    self.logger.warning(f"Train Step {step+1}: NaN/Inf in refiner_pred")
                                    continue
                                refiner_pred = torch.nan_to_num(refiner_pred, nan=0.0, posinf=1.0, neginf=-1.0)
                                # Combined loss
                                base_loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                                refiner_loss = torch.nn.functional.mse_loss(refiner_pred.float(), noise.float(), reduction="mean")
                                loss = (base_loss + refiner_loss) / 2.0
                                if torch.isnan(loss) or torch.isinf(loss):
                                    self.logger.warning(f"Train Step {step+1}: Calculated loss is NaN or Inf")
                                    continue
                            self.accelerator.backward(loss / gradient_accumulation_steps)
                            if self.accelerator.sync_gradients:
                                grad_norm = torch.tensor(float('inf'), device=self.device)
                                valid_gradients = True
                                try:
                                    grad_norm = self.accelerator.clip_grad_norm_(params_to_optimize, 0.5)
                                    self.logger.debug(f"Train Step {step+1}: Clipped gradient norm: {grad_norm.item():.4f}")
                                except ValueError:
                                    self.logger.warning(f"Train Step {step+1}: Skipping optimizer step due to NaN/Inf gradients")
                                    valid_gradients = False
                                if valid_gradients and not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                                    self.optimizer.step()
                                    self.optimizer.zero_grad()
                                else:
                                    num_skipped_steps += 1
                            train_loss_epoch += self.accelerator.gather(loss).mean().item() * gradient_accumulation_steps
                            num_train_batches_epoch += 1
                            if self.accelerator.is_main_process and global_step % 25 == 0:
                                self.logger.info(f"Fold {fold_idx+1}, Epoch {self.current_epoch}, Step {global_step}/{max_train_steps}, Train Loss: {loss.item():.4f}, Skipped Steps: {num_skipped_steps}")
                            global_step += 1
                    except Exception as e:
                        self.logger.error(f"Training step {step+1} failed: {e}\n{traceback.format_exc()}")
                        if "out of memory" in str(e).lower():
                            self.logger.error("OOM Error detected. Try smaller batch size.")
                        continue
                avg_train_loss_epoch = train_loss_epoch / num_train_batches_epoch if num_train_batches_epoch > 0 else float('nan')
                self.logger.info(f"Fold {fold_idx+1}, Epoch {self.current_epoch} - Avg Train Loss: {avg_train_loss_epoch:.4f}, Skipped Steps: {num_skipped_steps}/{num_train_batches_epoch}")
                avg_val_loss = self.validate(val_dataloader, dataset_path)
                if self.accelerator.is_main_process and avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.best_epoch = self.current_epoch
                    save_dir = os.path.join(self.output_dir, f"fold_{fold_idx+1}_epoch_{self.current_epoch}_loss_{avg_val_loss:.4f}")
                    self.save_model_state(epoch=self.current_epoch, val_loss=avg_val_loss, hyperparameters=hyperparameters, subdir=save_dir)
                gc.collect()
                torch.cuda.empty_cache()
            self.fold_val_losses.append(self.best_val_loss)
            if self.best_val_loss < global_best_avg_val_loss:
                global_best_avg_val_loss = self.best_val_loss
                global_best_hyperparameters = hyperparameters
                global_best_epoch = self.best_epoch
        valid_losses = [loss for loss in self.fold_val_losses if loss != float('inf')]
        avg_val_loss = np.mean(valid_losses) if valid_losses else float('inf')
        self.logger.info(f"Training Finished - Avg validation loss: {avg_val_loss:.4f} across {len(valid_losses)} valid folds")
        return avg_val_loss, None, global_best_hyperparameters, global_best_epoch

# --- Main Execution ---
def run_finetune(config_path):
    config = load_config(config_path)
    if config is None:
        return
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        mixed_precision='fp16',
        log_with="tensorboard",
        project_dir=os.path.join(config.get("base_output_dir", "./output"), "logs")
    )
    logger.info(accelerator.state, main_process_only=True)
    dataset_path = config.get("dataset_path", "/home/iris/Documents/deep_learning/data/finetune_dataset/coco/dataset.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    base_output_dir = config.get("base_output_dir", "/home/iris/Documents/deep_learning/experiments/sdxl_t5_refiner")
    if accelerator.is_main_process:
        os.makedirs(base_output_dir, exist_ok=True)
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
    full_dataset = load_dataset(dataset_path)
    if full_dataset is None:
        logger.error("Failed to load dataset")
        return
    k_folds = config.get("k_folds", 5)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    train_val_splits = [(Subset(full_dataset, train_idx), Subset(full_dataset, val_idx)) for train_idx, val_idx in kf.split(range(len(full_dataset)))]
    performance_records = []
    model_name = "sdxl_t5_refiner"
    overall_best_val_loss = float('inf')
    best_config = None
    for idx, hyperparams in enumerate(hyperparam_configs):
        config_name = f"hyperparam_config_{idx}"
        logger.info(f"--- Running Hyperparameter Config {idx+1}/{len(hyperparam_configs)} ({config_name}) ---")
        finetuner = FinetuneModel(model_name, base_output_dir, accelerator, logger_instance=logger)
        try:
            finetuner.load_model()
            finetuner.modify_architecture(
                apply_lora_to_unet=hyperparams['apply_lora_unet'],
                apply_lora_to_refiner=hyperparams['apply_lora_refiner'],
                apply_lora_to_text_encoder=hyperparams['apply_lora_text_encoder'],
                lora_r=hyperparams['lora_r'],
                lora_alpha=hyperparams['lora_alpha'],
                lora_dropout=hyperparams['lora_dropout']
            )
            avg_val_loss, _, hyperparameters, best_epoch = finetuner.fine_tune(
                dataset_path=dataset_path,
                train_val_splits=train_val_splits,
                epochs=hyperparams['epochs'],
                batch_size=hyperparams['batch_size'],
                learning_rate=float(hyperparams['learning_rate']),
                gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
                lora_r=hyperparams['lora_r'],
                lora_alpha=hyperparams['lora_alpha'],
                lora_dropout=hyperparams['lora_dropout']
            )
            performance_records.append({
                'config_idx': idx,
                'hyperparameters': hyperparams,
                'avg_val_loss': avg_val_loss,
                'fold_losses': finetuner.fold_val_losses,
                'epoch': best_epoch
            })
            if accelerator.is_main_process and not np.isnan(avg_val_loss) and avg_val_loss < overall_best_val_loss:
                overall_best_val_loss = avg_val_loss
                best_config = {
                    'index': idx,
                    'hyperparameters': hyperparams,
                    'avg_val_loss': avg_val_loss,
                    'epoch': best_epoch,
                    'fold_losses': finetuner.fold_val_losses
                }
                save_dir = os.path.join(base_output_dir, f"config_{idx}_loss_{avg_val_loss:.4f}")
                os.makedirs(save_dir, exist_ok=True)
                finetuner.save_model_state(
                    epoch=best_epoch,
                    val_loss=avg_val_loss,
                    hyperparameters=hyperparams,
                    subdir=save_dir
                )
        except Exception as e:
            logger.error(f"Run FAILED for {config_name}: {e}\n{traceback.format_exc()}")
        finally:
            del finetuner
            gc.collect()
            torch.cuda.empty_cache()
    if best_config and accelerator.is_main_process:
        summary_path = os.path.join(base_output_dir, "performance_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(performance_records, f, indent=4)
        logger.info(f"Final best configuration: Config {best_config['index']}, Avg Val Loss: {best_config['avg_val_loss']:.4f}, Epoch: {best_config['epoch']}")
    logger.info("Finetuning script completed.")

if __name__ == "__main__":
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    run_finetune(config_path)