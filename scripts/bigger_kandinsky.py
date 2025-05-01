import os
import torch
import logging
import yaml
import traceback
import numpy as np
import json
import math
import shutil
import copy
import torch.nn as nn
import warnings
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    UnCLIPPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    KandinskyV22PriorPipeline,
    UNet2DConditionModel,
)
from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer, AutoTokenizer,
    CLIPImageProcessor
)
from torch.utils.data import DataLoader
from PIL import Image
import gc
from src.utils.dataset import load_dataset, CocoFinetuneDataset
from bitsandbytes.optim import AdamW8bit
import numpy as np

# Suppress torch.cuda.amp.GradScaler deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")

# Configure logging
logging.basicConfig(
    filename='/home/iris/Documents/deep_learning/src/logs/bigger_kandinsky.log',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s',
    filemode='w',
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def summarize_prompt(prompt, max_tokens=512):
    """Summarize string prompt for DeepFloyd IF only if it exceeds max_tokens."""
    from transformers import pipeline
    
    try:
        if not prompt or not isinstance(prompt, str):
            logger.warning(f"Invalid prompt: {prompt}, returning empty string")
            return ""
        
        token_count = len(prompt.split())
        if token_count <= max_tokens:
            return prompt
        
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
        input_length = token_count
        max_length = max(10, input_length // 2)
        summary = summarizer(prompt, max_length=max_length, min_length=5, do_sample=False)[0]['summary_text']
        logger.debug(f"Summary for prompt '{prompt}': {summary}")
        return summary[:max_tokens] if summary else prompt[:max_tokens]
    except Exception as e:
        logger.error(f"Failed to summarize prompt: {e}")
        return prompt[:max_tokens] if prompt else ""

def select_top_topics(prompt, max_topics=10):
    """Return string prompt as-is for SDXL, Kandinsky, Karlo."""
    if not prompt or not isinstance(prompt, str):
        logger.warning(f"Invalid prompt: {prompt}, returning empty string")
        return ""
    return prompt

class FinetuneModel:
    def __init__(self, model_name, output_dir, accelerator: Accelerator, logger_instance=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.logger = logger_instance or logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.tokenizer = None
        self.tokenizer_2 = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.unet = None
        self.scheduler = None
        self.vae = None
        self.projection_layer = None
        self.image_size = 1024
        self.best_val_loss = -1
        self.best_epoch = -1
        self.current_epoch = 0
        self._apply_lora_text_flag = False
        self._apply_lora_unet_flag = False

    def load_model(self):
        """Load model components, replacing CLIP with T5 for specified models."""
        self.logger.info(f"Loading model: {self.model_name}")
        t5_model_name = "google/flan-t5-base"
        self.logger.info(f"Using Text Encoder: {t5_model_name}")

        try:
            if self.model_name in ["sdxl", "kandinsky"]:
                self.logger.info(f"Loading Tokenizer for {t5_model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                self.logger.info(f"Loading Text Encoder {t5_model_name}...")
                t5_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
                self.text_encoder = T5EncoderModel.from_pretrained(t5_model_name, torch_dtype=t5_dtype)
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                self.text_encoder.eval()
                self.text_encoder.to(self.device)
                self.logger.info(f"Loaded and froze Text Encoder {t5_model_name} on {self.device}.")

            if self.model_name == "sdxl":
                self.image_size = 1024
                base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                self.logger.info(f"Loading SDXL UNet/VAE/Scheduler from {base_model_id}...")
                self.vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float32).to(self.device)
                self.vae.eval()
                self.unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=self.dtype, variant="fp16").to(self.device)
                self.scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")
                self.tokenizer_2 = None
                self.text_encoder_2 = None
                t5_hidden_size = self.text_encoder.config.d_model
                unet_cross_attn_dim = self.unet.config.cross_attention_dim
                self.projection_layer = nn.Linear(t5_hidden_size, unet_cross_attn_dim).to(self.device).to(self.dtype)
                self.logger.info(f"Created projection layer: Linear({t5_hidden_size}, {unet_cross_attn_dim})")

            elif self.model_name == "kandinsky":
                self.image_size = 512
                decoder_model_id = "kandinsky-community/kandinsky-2-2-decoder"
                self.logger.info(f"Loading Kandinsky UNet/VAE/Scheduler from {decoder_model_id}...")
                self.vae = VQModel.from_pretrained(decoder_model_id, subfolder="movq", torch_dtype=torch.float32).to(self.device)
                self.vae.eval()
                self.unet = UNet2DConditionModel.from_pretrained(decoder_model_id, subfolder="unet", torch_dtype=self.dtype).to(self.device)
                self.scheduler = DPMSolverMultistepScheduler.from_pretrained(decoder_model_id, subfolder="scheduler")
                t5_hidden_size = self.text_encoder.config.d_model
                unet_cross_attn_dim = self.unet.config.cross_attention_dim
                self.projection_layer = nn.Linear(t5_hidden_size, unet_cross_attn_dim).to(self.device).to(self.dtype)
                self.logger.info(f"Created projection layer: Linear({t5_hidden_size}, {unet_cross_attn_dim})")

            elif self.model_name == "karlo":
                self.image_size = 64
                clip_id = "openai/clip-vit-base-patch32"
                self.text_encoder = CLIPTextModel.from_pretrained(clip_id, torch_dtype=self.dtype).to(self.device)
                self.tokenizer = CLIPTokenizer.from_pretrained(clip_id)
                self.unet, self.scheduler, self.vae, self.projection_layer = None, None, None, None
                self.logger.warning("Karlo loading unchanged, uses CLIP.")

            else:
                raise ValueError(f"Unknown model_name: {self.model_name}")

            if self.unet:
                for param in self.unet.parameters():
                    param.requires_grad = False
                self.unet.train()

            self.logger.info(f"Finished loading components for {self.model_name} on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}\n{traceback.format_exc()}")
            raise

    def modify_architecture(self, apply_lora_to_unet=True):
        """Applies LoRA to the UNet if specified. Skips Text Encoder LoRA."""
        self.logger.info(f"Configuring LoRA (Apply UNet: {apply_lora_to_unet})")
        self._apply_lora_text_flag = False
        self._apply_lora_unet_flag = apply_lora_to_unet

        if apply_lora_to_unet and self.unet:
            lora_r = 8; lora_alpha = 16; lora_dropout = 0.1; lora_bias = "none"
            unet_target_modules = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2.proj"]
            lora_config_unet = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=lora_bias, target_modules=unet_target_modules)
            try:
                self.unet = get_peft_model(self.unet, lora_config_unet)
                self.logger.info(f"Applied LoRA to UNet ({self.model_name})")
                self.unet.print_trainable_parameters()
            except Exception as e:
                self.logger.error(f"Failed to apply LoRA to UNet: {e}\n{traceback.format_exc()}")
        elif self.unet:
            self.logger.info(f"Skipped applying LoRA to UNet (Flag was {apply_lora_to_unet}).")

        self.logger.info("Skipping LoRA application for T5 Text Encoder (training projection layer instead).")

    def validate(self, val_dataloader, dataset_path):
        """Runs validation using T5 encoder and projection layer."""
        self.logger.info(f"--- Running Validation Epoch {self.current_epoch} for {self.model_name} ---")
        if not all([self.unet, self.text_encoder, self.projection_layer, self.tokenizer, self.scheduler]): 
            self.logger.error("Missing required components for validation")
            return -1
        if self.model_name in ["sdxl", "kandinsky"] and not self.vae: 
            self.logger.error("VAE missing for SDXL/Kandinsky")
            return -1

        if self.unet: self.unet.eval()
        if self.projection_layer: self.projection_layer.eval()

        total_val_loss = 0.0; num_val_batches = 0
        image_folder = os.path.join(os.path.dirname(dataset_path), "images")

        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                try:
                    image_filenames = batch['image']; prompts = batch['prompt']
                    if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts): 
                        self.logger.debug(f"Skipping batch {step+1}: Invalid prompts")
                        continue
                    pixel_values_list = []; valid_prompts = []
                    for i, (img_filename, prompt) in enumerate(zip(image_filenames, prompts)):
                        try:
                            image_path = os.path.join(image_folder, img_filename)
                            image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
                            image_np = np.array(image).astype(np.float32) / 255.0
                            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                            pixel_values_list.append(image_tensor)
                            valid_prompts.append(prompt)
                        except Exception as img_err:
                            self.logger.debug(f"Val Skip img {img_filename}: {img_err}")
                    if not pixel_values_list: 
                        self.logger.debug(f"Skipping batch {step+1}: No valid images")
                        continue
                    pixel_values = torch.stack(pixel_values_list).to(self.accelerator.device, dtype=torch.float32)
                    prompts = valid_prompts

                    added_cond_kwargs = {}; target_values = None; encoder_hidden_states = None; noisy_input = None; latents = None

                    try:
                        t5_max_length = self.tokenizer.model_max_length
                        inputs = self.tokenizer(prompts, padding="max_length", max_length=t5_max_length, truncation=True, return_tensors="pt").to(self.accelerator.device)
                        t5_outputs = self.text_encoder(inputs.input_ids)
                        projected_embeddings = self.projection_layer(t5_outputs.last_hidden_state.to(dtype=self.projection_layer.weight.dtype))
                        if torch.isnan(projected_embeddings).any() or torch.isinf(projected_embeddings).any(): 
                            self.logger.debug(f"Skipping batch {step+1}: Invalid embeddings")
                            continue
                        encoder_hidden_states = projected_embeddings
                    except Exception as text_err:
                        self.logger.error(f"Val Text/Projection failed in batch {step+1}: {text_err}")
                        continue

                    if self.model_name in ["sdxl", "kandinsky"]:
                        pixel_values_norm = pixel_values * 2.0 - 1.0
                        try:
                            input_vae = pixel_values_norm.to(dtype=torch.float32)
                            vae_output = self.vae.encode(input_vae)
                            if isinstance(self.vae, AutoencoderKL):
                                latents = vae_output.latent_dist.sample()
                            elif isinstance(self.vae, VQModel):
                                latents = vae_output.latents
                            else:
                                self.logger.debug(f"Skipping batch {step+1}: Unknown VAE type")
                                continue
                        except Exception as e:
                            self.logger.error(f"Val VAE encode fail in batch {step+1}: {e}")
                            continue
                        if latents is None or torch.isnan(latents).any() or torch.isinf(latents).any(): 
                            self.logger.debug(f"Skipping batch {step+1}: Invalid latents")
                            continue
                        scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)
                        latents = (latents.float() * scaling_factor).to(dtype=self.dtype)
                        if torch.isnan(latents).any() or torch.isinf(latents).any(): 
                            self.logger.debug(f"Skipping batch {step+1}: Invalid scaled latents")
                            continue
                        noise = torch.randn_like(latents)
                        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.accelerator.device).long()
                        noisy_input = self.scheduler.add_noise(latents, noise, timesteps)

                    if noisy_input is None or torch.isnan(noisy_input).any() or torch.isinf(noisy_input).any(): 
                        self.logger.debug(f"Skipping batch {step+1}: Invalid noisy input")
                        continue
                    target_values = noise

                    if self.unet and encoder_hidden_states is not None:
                        unet_dtype = self.unet.dtype
                        unet_args = {
                            "sample": noisy_input.to(dtype=unet_dtype),
                            "timestep": timesteps,
                            "encoder_hidden_states": encoder_hidden_states.to(dtype=unet_dtype),
                            "added_cond_kwargs": {}
                        }
                        if self.model_name == "kandinsky":
                            image_embed_dim = 1280
                            batch_size_current = noisy_input.shape[0]
                            zero_image_embeds = torch.zeros((batch_size_current, image_embed_dim), dtype=unet_dtype, device=self.accelerator.device)
                            unet_args["added_cond_kwargs"]["image_embeds"] = zero_image_embeds

                        unet_output = self.unet(**unet_args)
                        model_pred = unet_output.sample if hasattr(unet_output, "sample") else unet_output
                        if model_pred.shape[1] != target_values.shape[1]:
                            if model_pred.shape[1] == target_values.shape[1] * 2:
                                model_pred = model_pred[:, :target_values.shape[1], :, :]
                            else:
                                self.logger.debug(f"Skipping batch {step+1}: Mismatched prediction shape")
                                continue
                        if torch.isnan(model_pred).any() or torch.isinf(model_pred).any(): 
                            self.logger.debug(f"Skipping batch {step+1}: Invalid model prediction")
                            continue

                        val_loss = torch.nn.functional.mse_loss(model_pred.float(), target_values.float(), reduction="mean")
                        if not torch.isnan(val_loss):
                            gathered_loss = self.accelerator.gather(val_loss)
                            total_val_loss += gathered_loss.mean().item()
                            num_val_batches += 1
                        else:
                            self.logger.warning(f"Val Step {step+1}: Calculated loss is NaN.")

                except Exception as val_step_err:
                    self.logger.error(f"Validation step {step+1} failed: {val_step_err}\n{traceback.format_exc()}")
                    continue

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else -1
        self.logger.info(f"--- Epoch {self.current_epoch} Validation Summary ---")
        self.logger.info(f"Average Validation Loss: {avg_val_loss:.4f} ({num_val_batches} batches)")

        if self.unet: self.unet.train()
        if self.projection_layer: self.projection_layer.train()

        return avg_val_loss

    def save_model_state(self, epoch=None, val_loss=None, hyperparameters=None):
        """Saves the trained LoRA weights and projection layer using Accelerator."""
        if not self.accelerator.is_main_process: 
            self.logger.debug("Not main process, skipping save")
            return
        if epoch is None: 
            epoch = self.current_epoch

        save_label = "best"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None and not np.isnan(val_loss) and not np.isinf(val_loss) else "N/A"
        self.logger.info(f"Saving model state for {self.model_name} as '{save_label}' (Epoch: {epoch}, Val Loss: {val_loss_str})")

        output_subdir = self.output_dir
        try:
            os.makedirs(output_subdir, exist_ok=True)
            self.logger.info(f"Created output directory: {output_subdir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory {output_subdir}: {e}")
            return

        save_paths = {}

        if self.unet and isinstance(self.unet, PeftModel):
            try:
                unwrapped_unet = self.accelerator.unwrap_model(self.unet)
                unet_path = os.path.join(output_subdir, "best_unet_lora")
                unwrapped_unet.save_pretrained(unet_path)
                save_paths["UNet_LoRA"] = unet_path
                self.logger.info(f"Saved UNet LoRA weights to {unet_path}")
            except Exception as e:
                self.logger.error(f"Failed to save UNet LoRA: {e}")

        if self.projection_layer:
            try:
                unwrapped_proj = self.accelerator.unwrap_model(self.projection_layer)
                proj_path = os.path.join(output_subdir, "best_projection_layer.pth")
                self.accelerator.save(unwrapped_proj.state_dict(), proj_path)
                save_paths["ProjectionLayer"] = proj_path
                self.logger.info(f"Saved Projection Layer state_dict to {proj_path}")
            except Exception as e:
                self.logger.error(f"Failed to save Projection Layer: {e}")

        if hyperparameters:
            hyperparameters['_apply_lora_text_encoder'] = self._apply_lora_text_flag
            hyperparameters['_apply_lora_unet'] = self._apply_lora_unet_flag
            hyperparam_path = os.path.join(output_subdir, "best_hyperparameters.json")
            save_data = {
                'model_name': self.model_name,
                'epoch': epoch,
                'validation_loss': val_loss_str,
                'hyperparameters': hyperparameters
            }
            try:
                with open(hyperparam_path, 'w') as f:
                    json.dump(save_data, f, indent=4, default=str)
                self.logger.info(f"Saved hyperparameters to {hyperparam_path}")
            except Exception as e:
                self.logger.error(f"Failed to save hyperparameters: {e}")

        if not save_paths:
            self.logger.warning(f"No trainable weights were saved for {self.model_name} ({save_label}).")

    def fine_tune(self, dataset_path, train_val_splits, epochs=1, batch_size=1, learning_rate=1e-5, gradient_accumulation_steps=1):
        """Fine-tune the UNet LoRA adapters and T5 projection layer using k-fold cross-validation."""
        self.logger.info(f"Starting T5-UNet fine-tuning with {len(train_val_splits)} folds...")
        fold_val_losses = []
        global_best_avg_val_loss = -1
        global_best_model_state = None
        global_best_hyperparameters = None
        global_best_epoch = -1

        for fold_idx, (train_dataset, val_dataset) in enumerate(train_val_splits):
            self.logger.info(f"Training fold {fold_idx + 1}/{len(train_val_splits)}")
            self.best_val_loss = -1
            self.best_epoch = -1
            fold_best_model_state = None

            try:
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                data_dir = os.path.dirname(dataset_path)
                image_folder = os.path.join(data_dir, "images")

                params_to_optimize = []
                models_to_prep = []
                if self.projection_layer:
                    params_to_optimize.extend(self.projection_layer.parameters())
                    models_to_prep.append(self.projection_layer)
                    self.logger.info("Added Projection Layer params.")
                if self.unet and isinstance(self.unet, PeftModel):
                    params_to_optimize.extend(filter(lambda p: p.requires_grad, self.unet.parameters()))
                    models_to_prep.append(self.unet)
                    self.logger.info("Added UNet LoRA params.")
                optimizer = None
                if params_to_optimize:
                    try:
                        optimizer = AdamW8bit(params_to_optimize, lr=learning_rate)
                        self.logger.info("Using AdamW8bit.")
                    except Exception as e:
                        self.logger.warning(f"AdamW8bit failed: {e}, using AdamW.")
                        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
                else:
                    self.logger.error("No trainable parameters!")
                    return -1, None, None, -1

                t5_model_name = self.text_encoder.name_or_path if hasattr(self.text_encoder, 'name_or_path') else 'Unknown T5'
                hyperparameters = {
                    'model_name': self.model_name,
                    'text_encoder': t5_model_name,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'gradient_accumulation_steps': gradient_accumulation_steps,
                    'lora_r': 8,
                    'lora_alpha': 16,
                    'lora_dropout': 0.1,
                    'apply_lora_text_encoder': self._apply_lora_text_flag,
                    'apply_lora_unet': self._apply_lora_unet_flag
                }

                self.logger.info("Preparing components with Accelerator...")
                if models_to_prep and optimizer:
                    components_to_prepare = models_to_prep + [optimizer, train_dataloader, val_dataloader]
                    prepared_components = self.accelerator.prepare(*components_to_prepare)
                    component_iter = iter(prepared_components)
                    if self.projection_layer in models_to_prep:
                        self.projection_layer = next(component_iter)
                    if self.unet in models_to_prep:
                        self.unet = next(component_iter)
                    self.optimizer = next(component_iter)
                    train_dataloader = next(component_iter)
                    val_dataloader = next(component_iter)
                else:
                    self.logger.error("Cannot prepare components.")
                    return -1, None, None, -1

                if self.text_encoder:
                    self.text_encoder.to(self.accelerator.device)
                    self.text_encoder.eval()
                if self.vae:
                    self.vae.to(self.accelerator.device)
                    self.vae.eval()

                num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
                max_train_steps = epochs * num_update_steps_per_epoch
                self.logger.info("***** Running training *****")
                global_step = 0

                for epoch in range(epochs):
                    self.current_epoch = epoch + 1
                    self.logger.info(f"--- Starting Epoch {self.current_epoch}/{epochs} ---")
                    if self.unet:
                        self.unet.train()
                    if self.projection_layer:
                        self.projection_layer.train()

                    total_train_loss = 0.0
                    num_train_batches = 0
                    for step, batch in enumerate(train_dataloader):
                        try:
                            image_filenames = batch['image']
                            prompts = batch['prompt']
                            if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
                                self.logger.debug(f"Skipping train batch {step+1}: Invalid prompts")
                                continue
                            pixel_values_list = []
                            valid_prompts = []
                            for i, (img_filename, prompt) in enumerate(zip(image_filenames, prompts)):
                                try:
                                    image_path = os.path.join(image_folder, img_filename)
                                    image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
                                    image_np = np.array(image).astype(np.float32) / 255.0
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

                            with self.accelerator.accumulate(models_to_prep[0]):
                                loss = None
                                noisy_input = None
                                target_values = None
                                encoder_hidden_states = None
                                try:
                                    t5_max_length = self.tokenizer.model_max_length
                                    inputs = self.tokenizer(prompts, padding="max_length", max_length=t5_max_length, truncation=True, return_tensors="pt").to(self.accelerator.device)
                                    with torch.no_grad():
                                        t5_outputs = self.text_encoder(inputs.input_ids)
                                        t5_hidden_states = t5_outputs.last_hidden_state.to(dtype=self.projection_layer.weight.dtype)
                                    projected_embeddings = self.projection_layer(t5_hidden_states)
                                    if torch.isnan(projected_embeddings).any() or torch.isinf(projected_embeddings).any():
                                        self.logger.debug(f"Skipping train batch {step+1}: Invalid embeddings")
                                        continue
                                    encoder_hidden_states = projected_embeddings
                                except Exception as text_err:
                                    self.logger.error(f"Train Text/Projection failed in batch {step+1}: {text_err}")
                                    continue

                                if self.model_name in ["sdxl", "kandinsky"]:
                                    pixel_values_norm = pixel_values * 2.0 - 1.0
                                    with torch.no_grad():
                                        input_vae = pixel_values_norm.to(dtype=torch.float32)
                                        vae_output = self.vae.encode(input_vae)
                                    if isinstance(self.vae, AutoencoderKL):
                                        latents = vae_output.latent_dist.sample()
                                    elif isinstance(self.vae, VQModel):
                                        latents = vae_output.latents
                                    else:
                                        self.logger.debug(f"Skipping train batch {step+1}: Unknown VAE type")
                                        continue
                                    if latents is None or torch.isnan(latents).any() or torch.isinf(latents).any():
                                        self.logger.debug(f"Skipping train batch {step+1}: Invalid latents")
                                        continue
                                    scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)
                                    latents = (latents.float() * scaling_factor).to(dtype=self.dtype)
                                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                                        self.logger.debug(f"Skipping train batch {step+1}: Invalid scaled latents")
                                        continue
                                    noise = torch.randn_like(latents)
                                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.accelerator.device).long()
                                    noisy_input = self.scheduler.add_noise(latents, noise, timesteps)

                                if noisy_input is None or torch.isnan(noisy_input).any() or torch.isinf(noisy_input).any():
                                    self.logger.debug(f"Skipping train batch {step+1}: Invalid noisy input")
                                    continue
                                target_values = noise

                                if self.unet and encoder_hidden_states is not None and self.model_name != "karlo":
                                    unet_args = {
                                        "sample": noisy_input,
                                        "timestep": timesteps,
                                        "encoder_hidden_states": encoder_hidden_states,
                                        "added_cond_kwargs": {}
                                    }
                                    if self.model_name == "kandinsky":
                                        image_embed_dim = 1280
                                        batch_size_current = noisy_input.shape[0]
                                        zero_image_embeds = torch.zeros((batch_size_current, image_embed_dim), dtype=self.unet.dtype, device=self.accelerator.device)
                                        unet_args["added_cond_kwargs"]["image_embeds"] = zero_image_embeds

                                    model_pred = self.unet(**unet_args).sample
                                    if hasattr(model_pred, "sample"):
                                        model_pred = model_pred.sample
                                    if model_pred.shape[1] != target_values.shape[1]:
                                        if model_pred.shape[1] == target_values.shape[1] * 2:
                                            model_pred = model_pred[:, :target_values.shape[1], :, :]
                                        else:
                                            self.logger.debug(f"Skipping train batch {step+1}: Mismatched prediction shape")
                                            continue
                                    if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                                        self.logger.debug(f"Skipping train batch {step+1}: Invalid model prediction")
                                        continue
                                    loss = torch.nn.functional.mse_loss(model_pred.float(), target_values.float(), reduction="mean")

                                if loss is not None and not torch.isnan(loss):
                                    self.accelerator.backward(loss / gradient_accumulation_steps)
                                    total_train_loss += loss.item()
                                    num_train_batches += 1
                                    if self.accelerator.sync_gradients:
                                        if params_to_optimize:
                                            self.accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                                        if self.optimizer:
                                            self.optimizer.step()
                                            self.optimizer.zero_grad()
                                    if self.accelerator.is_main_process and global_step % 50 == 0:
                                        self.logger.info(f"Epoch {self.current_epoch}, Step {global_step}, Train Loss: {loss.item():.4f}")
                                else:
                                    self.logger.warning(f"Invalid loss ({loss}) in train step {step+1}.")
                                global_step += 1
                        except Exception as e:
                            self.logger.error(f"Training step {step+1} failed: {e}\n{traceback.format_exc()}")
                            continue

                    avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('nan')
                    self.logger.info(f"--- Epoch {self.current_epoch} Finished --- Avg Train Loss: {avg_train_loss:.4f}")

                    avg_val_loss = -1
                    if val_dataloader:
                        avg_val_loss = self.validate(val_dataloader, dataset_path)
                    else:
                        self.logger.info("Skipping validation (no validation set).")
                        avg_val_loss = avg_train_loss

                    if not np.isnan(avg_val_loss) and (self.best_val_loss == -1 or avg_val_loss < self.best_val_loss):
                        self.best_val_loss = avg_val_loss
                        self.best_epoch = self.current_epoch
                        self.logger.info(f"New best validation loss for fold {fold_idx + 1}: {avg_val_loss:.4f} (Epoch {self.best_epoch}). Storing state.")
                        fold_best_model_state = {}
                        if self.unet and isinstance(self.unet, PeftModel):
                            fold_best_model_state['unet'] = copy.deepcopy(self.accelerator.unwrap_model(self.unet).state_dict())
                        if self.projection_layer:
                            fold_best_model_state['projection'] = copy.deepcopy(self.accelerator.unwrap_model(self.projection_layer).state_dict())

                    gc.collect()
                    torch.cuda.empty_cache()

                valid_fold_loss = self.best_val_loss if not np.isnan(self.best_val_loss) and self.best_val_loss != -1 else -1
                fold_val_losses.append(valid_fold_loss)
                self.logger.info(f"Fold {fold_idx + 1} best validation loss: {valid_fold_loss:.4f}")

            except Exception as e:
                self.logger.error(f"Fold {fold_idx + 1} failed: {e}\n{traceback.format_exc()}")
                fold_val_losses.append(-1)

        valid_losses = [loss for loss in fold_val_losses if loss != -1]
        avg_val_loss = np.mean(valid_losses) if valid_losses else -1
        self.logger.info(f"Average validation loss across {len(valid_losses)} valid folds: {avg_val_loss:.4f}")

        if valid_losses and not np.isnan(avg_val_loss) and avg_val_loss != -1:
            global_best_avg_val_loss = avg_val_loss
            global_best_model_state = fold_best_model_state
            global_best_hyperparameters = hyperparameters
            global_best_epoch = self.best_epoch
            self.logger.info(f"Selected model state for saving: Avg Val Loss: {avg_val_loss:.4f} (Epoch {global_best_epoch})")
        else:
            self.logger.warning(f"No valid model state to save. Avg Val Loss: {avg_val_loss:.4f}, Valid folds: {len(valid_losses)}")

        return avg_val_loss, global_best_model_state, global_best_hyperparameters, global_best_epoch