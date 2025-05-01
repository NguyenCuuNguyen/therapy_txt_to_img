import os
import torch
import logging
import yaml
import traceback
import numpy as np
import json
from accelerate import Accelerator # Import Accelerator
from accelerate.logging import get_logger # Use accelerate logger
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    UnCLIPPipeline,
    DPMSolverMultistepScheduler,
    VQModel,
    AutoencoderKL,
    KandinskyV22PriorPipeline, # Needed for Kandinsky loading context
    UNet2DConditionModel, # Import specific UNet class
)
from peft import LoraConfig, get_peft_model
from transformers import (
    CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer, AutoTokenizer,
    CLIPImageProcessor # Keep if needed for other models
)
from torch.utils.data import DataLoader, Subset
from PIL import Image
import gc
from src.utils.dataset import load_dataset, CocoFinetuneDataset
from itertools import chain
from bitsandbytes.optim import AdamW8bit

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
    def __init__(self, model_name, output_dir, logger_instance=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.logger = logger_instance or logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 # Target precision for UNet LoRA / Base UNet
        # self.model = None # Not used
        self.tokenizer = None
        self.tokenizer_2 = None
        self.text_encoder = None # This will now be T5 for Kandinsky/SDXL
        self.text_encoder_2 = None # Not used if replacing with single T5
        self.unet = None
        self.scheduler = None
        self.vae = None
        self.projection_layer = None # !--- New attribute ---!
        self.image_size = 1024
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.current_epoch = 0

    def load_model(self):
        """Load model components, replacing CLIP with T5 for specified models."""
        self.logger.info(f"Loading model: {self.model_name}")
        # Define the T5 model we want to use
        t5_model_name = "google/flan-t5-base" #"google/flan-t5-xl" # 3B params, ~2048 hidden size
        # t5_model_name = "google/flan-t5-large" # Smaller option, ~1024 hidden size, also google/flan-t5-base. both bave 512 tokens
        self.logger.info(f"Using Text Encoder: {t5_model_name}")

        try:
            # --- Load T5 Encoder and Tokenizer (Common for modified models) ---
            if self.model_name in ["sdxl", "kandinsky"]:
                 self.logger.info(f"Loading Tokenizer for {t5_model_name}...")
                 self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                 self.logger.info(f"Loading Text Encoder {t5_model_name} (this may take time)...")
                 # Load T5 in bfloat16 if possible for stability, otherwise float32. Freeze it.
                 try:
                     t5_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
                     self.logger.info(f"Loading T5 in dtype: {t5_dtype}")
                     self.text_encoder = T5EncoderModel.from_pretrained(t5_model_name, torch_dtype=t5_dtype)
                 except Exception as e:
                      self.logger.warning(f"Failed loading T5 in {t5_dtype}, trying float32: {e}")
                      self.text_encoder = T5EncoderModel.from_pretrained(t5_model_name, torch_dtype=torch.float32)

                 # Freeze the T5 encoder
                 for param in self.text_encoder.parameters():
                     param.requires_grad = False
                 self.text_encoder.eval() # Keep T5 in eval mode
                 self.text_encoder.to(self.device) # Move T5 to device
                 self.logger.info(f"Loaded and froze Text Encoder {t5_model_name} on {self.device}.")


            # --- Load Model-Specific Components (UNet, VAE, Scheduler) ---
            if self.model_name == "sdxl":
                self.image_size = 1024
                base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                self.logger.info(f"Loading SDXL UNet/VAE/Scheduler from {base_model_id}...")
                self.vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float32).to(self.device) # VAE fp32
                self.vae.eval()
                self.unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=self.dtype, variant="fp16").to(self.device) # UNet fp16
                self.scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")
                # SDXL originally used two text encoders. We only have one (T5).
                self.tokenizer_2 = None
                self.text_encoder_2 = None
                # !--- Define Projection Layer for SDXL ---!
                t5_hidden_size = self.text_encoder.config.d_model # e.g., 2048 for T5-XL
                unet_cross_attn_dim = self.unet.config.cross_attention_dim # e.g., 2048 for SDXL
                self.projection_layer = nn.Linear(t5_hidden_size, unet_cross_attn_dim).to(self.device).to(self.dtype) # Trainable layer
                self.logger.info(f"Created projection layer: Linear({t5_hidden_size}, {unet_cross_attn_dim})")

            elif self.model_name == "kandinsky":
                self.image_size = 512
                decoder_model_id = "kandinsky-community/kandinsky-2-2-decoder"
                # Prior pipeline/encoder not loaded here as we replace the decoder's text input
                self.logger.info(f"Loading Kandinsky UNet/VAE/Scheduler from {decoder_model_id}...")
                self.vae = VQModel.from_pretrained(decoder_model_id, subfolder="movq", torch_dtype=torch.float32).to(self.device) # VAE fp32
                self.vae.eval()
                self.unet = UNet2DConditionModel.from_pretrained(decoder_model_id, subfolder="unet", torch_dtype=self.dtype).to(self.device) # UNet fp16
                self.scheduler = DPMSolverMultistepScheduler.from_pretrained(decoder_model_id, subfolder="scheduler")
                # !--- Define Projection Layer for Kandinsky ---!
                t5_hidden_size = self.text_encoder.config.d_model # e.g., 2048 for T5-XL
                unet_cross_attn_dim = self.unet.config.cross_attention_dim # e.g., 1024 for Kandinsky Decoder
                self.projection_layer = nn.Linear(t5_hidden_size, unet_cross_attn_dim).to(self.device).to(self.dtype) # Trainable layer
                self.logger.info(f"Created projection layer: Linear({t5_hidden_size}, {unet_cross_attn_dim})")

            elif self.model_name == "karlo":
                 # Karlo doesn't fit this paradigm well, keep original simple load
                 self.image_size = 64
                 clip_id = "openai/clip-vit-base-patch32"
                 self.text_encoder = CLIPTextModel.from_pretrained(clip_id, torch_dtype=self.dtype).to(self.device)
                 self.tokenizer = CLIPTokenizer.from_pretrained(clip_id)
                 self.unet, self.scheduler, self.vae, self.projection_layer = None, None, None, None
                 self.logger.warning("Karlo loading unchanged, uses CLIP.")

            else:
                raise ValueError(f"Unknown model_name: {self.model_name}")

            # Freeze base UNet weights (only projection layer and LoRA adapters will be trained)
            if self.unet:
                for param in self.unet.parameters():
                    param.requires_grad = False
                self.unet.train() # Set UNet to train mode for LoRA application/training

            self.logger.info(f"Finished loading components for {self.model_name} on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}\n{traceback.format_exc()}")
            raise

        
    def modify_architecture(self, apply_lora_to_text_encoder=True, apply_lora_to_unet=True):
        """Applies LoRA to the UNet if specified. Skips Text Encoder LoRA."""
        self.logger.info(f"Configuring LoRA (Apply UNet: {apply_lora_to_unet})")
        # Store flags for saving later
        self._apply_lora_text_flag = False # Forcing False as we use T5 now
        self._apply_lora_unet_flag = apply_lora_to_unet

        # --- Apply LoRA to UNet ---
        if apply_lora_to_unet and self.unet:
            lora_r = 8; lora_alpha = 16; lora_dropout = 0.1; lora_bias = "none"
            unet_target_modules = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2.proj"]
            lora_config_unet = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=lora_bias, target_modules=unet_target_modules)
            try:
                self.unet = get_peft_model(self.unet, lora_config_unet)
                self.logger.info(f"Applied LoRA to UNet ({self.model_name})")
                # Ensure only LoRA params are trainable in UNet
                self.unet.print_trainable_parameters()
            except Exception as e:
                 self.logger.error(f"Failed to apply LoRA to UNet: {e}\n{traceback.format_exc()}")
                 # Decide if you want to continue without UNet LoRA or raise error
                 # raise e
        elif self.unet:
             self.logger.info(f"Skipped applying LoRA to UNet (Flag was {apply_lora_to_unet}).")

        # --- Text Encoder LoRA is skipped ---
        self.logger.info("Skipping LoRA application for T5 Text Encoder (training projection layer instead).")


    def _encode_prompt_sdxl(self, prompt_batch):
        """Encodes prompts for SDXL using both text encoders."""
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]
        prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt_batch,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)
            encoder_outputs = text_encoder(text_input_ids, output_hidden_states=True, return_dict=True)
            prompt_embeds = encoder_outputs.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)
            if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
                pooled_prompt_embeds = encoder_outputs.pooler_output
            else:
                pooled_prompt_embeds = encoder_outputs.last_hidden_state[:, 0, :]
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds_list[1]
        return prompt_embeds, pooled_prompt_embeds

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        """Helper to generate SDXL time IDs."""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Add dataset_path as an argument
    def validate(self, val_dataloader, dataset_path, batch_size=1):
        """Runs validation on the provided dataloader."""
        self.logger.info(f"--- Running Validation for {self.model_name} ---")

        # Check if necessary components exist
        if self.model_name != "karlo" and not self.unet:
            self.logger.warning("UNet not available for validation.")
            return float('inf') # Return infinite loss if UNet needed but missing
        if not self.text_encoder:
             self.logger.warning("Text Encoder not available for validation.")
             return float('inf')
        if self.model_name in ["sdxl", "kandinsky"] and not self.vae:
             self.logger.warning("VAE not available for validation for latent model.")
             return float('inf')
        if not self.scheduler:
             self.logger.warning("Scheduler not available for validation.")
             return float('inf')


        # Set models to eval mode
        if self.unet: self.unet.eval()
        if self.text_encoder: self.text_encoder.eval()
        if hasattr(self, 'text_encoder_2') and self.text_encoder_2: self.text_encoder_2.eval() # For SDXL
        if self.vae: self.vae.eval()

        total_val_loss = 0.0
        num_val_batches = 0
        image_folder = os.path.join(os.path.dirname(dataset_path), "images") # Define image folder path once

        with torch.no_grad(): # No gradients needed for validation
            for step, batch in enumerate(val_dataloader):
                try:
                    # --- Load and Preprocess Images (from batch) ---
                    image_filenames = batch['image']
                    prompts = batch['prompt']
                    if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
                        self.logger.warning(f"Val Step {step+1}: Skipping batch due to invalid prompts.")
                        continue

                    pixel_values_list = []
                    valid_prompts = []
                    for i, (img_filename, prompt) in enumerate(zip(image_filenames, prompts)):
                        try:
                            image_path = os.path.join(image_folder, img_filename)
                            image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
                            image_np = np.array(image).astype(np.float32) / 255.0
                            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                            pixel_values_list.append(image_tensor); valid_prompts.append(prompt)
                        except FileNotFoundError:
                             self.logger.warning(f"Val Skipping image {img_filename}: File not found.")
                        except Exception as img_err:
                             self.logger.warning(f"Val Skipping image {img_filename}: Error loading/processing - {img_err}")
                    if not pixel_values_list:
                        self.logger.warning(f"Val Step {step+1}: Skipping batch as no valid images could be processed.")
                        continue
                    # Move to device, keep as FP32 for VAE input if applicable
                    pixel_values = torch.stack(pixel_values_list).to(self.device, dtype=torch.float32)
                    prompts = valid_prompts
                    # --- End Image Loading ---

                    # --- Prepare Inputs (Mirror training logic) ---
                    added_cond_kwargs = {}; target_values = None; encoder_hidden_states = None; noisy_input = None; latents = None

                    # --- VAE Encoding (if needed) ---
                    if self.model_name in ["sdxl", "kandinsky"]:
                        pixel_values_norm = pixel_values * 2.0 - 1.0
                        if torch.isnan(pixel_values_norm).any(): self.logger.warning(f"Val Step {step+1}: NaN in pixel_values_norm."); continue

                        try:
                            input_vae = pixel_values_norm.to(dtype=torch.float32) # Ensure fp32 for fp32 VAE
                            vae_output = self.vae.encode(input_vae)
                            if isinstance(self.vae, AutoencoderKL):
                                if not hasattr(vae_output, 'latent_dist'): self.logger.error("Val SDXL VAE missing 'latent_dist'."); continue
                                latents = vae_output.latent_dist.sample()
                            elif isinstance(self.vae, VQModel):
                                if hasattr(vae_output, 'latents'): latents = vae_output.latents
                                else: self.logger.error("Val VQEncoderOutput missing '.latents'."); continue
                            else: self.logger.error(f"Val Unhandled VAE type: {type(self.vae)}."); continue
                        except Exception as e: self.logger.error(f"Val VAE encode fail Step {step+1}: {e}"); continue

                        if latents is None or torch.isnan(latents).any() or torch.isinf(latents).any(): self.logger.warning(f"Val Step {step+1}: NaN/Inf after VAE encode."); continue

                        scaling_factor = getattr(self.vae.config, 'scaling_factor', 1.0 if isinstance(self.vae, VQModel) else 0.18215)
                        latents = (latents.float() * scaling_factor) # Scale in fp32
                        if torch.isnan(latents).any() or torch.isinf(latents).any(): self.logger.warning(f"Val Step {step+1}: NaN/Inf after scaling latents."); continue

                        latents = latents.to(dtype=self.dtype) # Cast to target dtype

                    # --- Text Encoding ---
                    try:
                        if self.model_name == "sdxl":
                            # ... (SDXL text encoding logic - needs _encode_prompt_sdxl helper) ...
                            prompt_embeds, pooled_embeds = self._encode_prompt_sdxl(prompts)
                            if torch.isnan(prompt_embeds).any() or torch.isinf(prompt_embeds).any() or \
                               torch.isnan(pooled_embeds).any() or torch.isinf(pooled_embeds).any():
                                self.logger.warning(f"Val Step {step+1}: NaN/Inf in SDXL embeddings."); continue
                            encoder_hidden_states = prompt_embeds
                            add_time_ids = self._get_add_time_ids((self.image_size, self.image_size), (0,0), (self.image_size, self.image_size), dtype=prompt_embeds.dtype).repeat(len(prompts), 1).to(self.device)
                            added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}
                        else: # Kandinsky, DeepFloyd, Karlo (single text encoder)
                            if self.model_name == "deepfloyd_if": processed_prompts = [summarize_prompt(p) for p in prompts]
                            else: processed_prompts = prompts
                            inputs = self.tokenizer(processed_prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(self.device)
                            # Use the text encoder (which might be fp16 or fp32 depending on loading)
                            temp_hidden_states = self.text_encoder(inputs.input_ids)[0]
                            if torch.isnan(temp_hidden_states).any() or torch.isinf(temp_hidden_states).any():
                                self.logger.warning(f"Val Step {step+1}: NaN/Inf in {self.model_name} embeddings."); continue
                            encoder_hidden_states = temp_hidden_states.to(dtype=self.dtype) # Ensure target dtype for UNet input later
                    except Exception as text_err: self.logger.error(f"Val Text encode fail Step {step+1}: {text_err}"); continue

                    # --- Prepare Noise and Target for UNet ---
                    if self.model_name != "karlo": # Karlo doesn't need noise/target for validation loss
                        if self.model_name in ["sdxl", "kandinsky"]: # Latent space
                            if latents is None: self.logger.error("Val Latents are None before noise step."); continue
                            noise = torch.randn_like(latents)
                            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()
                            noisy_input = self.scheduler.add_noise(latents, noise, timesteps)
                        elif self.model_name == "deepfloyd_if": # Pixel space
                            pixel_values_target_dtype = pixel_values.to(dtype=self.target_dtype)
                            noise = torch.randn_like(pixel_values_target_dtype)
                            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (pixel_values_target_dtype.shape[0],), device=self.device).long()
                            noisy_input = self.scheduler.add_noise(pixel_values_target_dtype, noise, timesteps)
                        if noisy_input is None or torch.isnan(noisy_input).any() or torch.isinf(noisy_input).any(): self.logger.error(f"Val Step {step+1}: NaN/Inf in noisy_input."); continue
                        target_values = noise

                    # --- Forward Pass ---
                    if self.model_name != "karlo":
                        if noisy_input is None or encoder_hidden_states is None or self.unet is None: continue
                        unet_dtype = self.unet.dtype if hasattr(self.unet, 'dtype') else self.dtype
                        unet_args = { "sample": noisy_input.to(dtype=unet_dtype),
                                      "timestep": timesteps,
                                      "encoder_hidden_states": encoder_hidden_states.to(dtype=unet_dtype),
                                      "added_cond_kwargs": {} }
                        model_pred = self.unet(**unet_args).sample
                        if hasattr(model_pred, "sample"): model_pred = model_pred.sample # Handle nested output
                        # ... (Shape slicing logic if needed) ...
                        if model_pred.shape[1] != target_values.shape[1]:
                            if model_pred.shape[1] == target_values.shape[1] * 2: model_pred = model_pred[:, :target_values.shape[1], :, :]
                            else: continue
                        if torch.isnan(model_pred).any() or torch.isinf(model_pred).any(): continue
                        loss = torch.nn.functional.mse_loss(model_pred.float(), target_values.float(), reduction="mean")


                        unet_output = self.unet(**unet_args)
                        model_pred = unet_output.sample if hasattr(unet_output, "sample") else unet_output
                        if model_pred.shape[1] != target_values.shape[1]:
                            if model_pred.shape[1] == target_values.shape[1] * 2: model_pred = model_pred[:, :target_values.shape[1], :, :]
                            else: self.logger.error(f"Val Step {step+1}: Shape mismatch after slice."); continue
                        if torch.isnan(model_pred).any() or torch.isinf(model_pred).any(): self.logger.error(f"Val Step {step+1}: NaN/Inf in model_pred."); continue

                        # --- Calculate Validation Loss ---
                        val_loss = torch.nn.functional.mse_loss(model_pred.float(), target_values.float(), reduction="mean")
                        if not torch.isnan(val_loss):
                            total_val_loss += val_loss.item()
                            num_val_batches += 1
                        else:
                             self.logger.warning(f"Val Step {step+1}: Calculated loss is NaN.")

                except Exception as val_step_err:
                    self.logger.error(f"Validation step {step+1} failed unexpectedly: {val_step_err}\n{traceback.format_exc()}")
                    continue # Continue to next validation batch

        # --- End Validation Loop ---

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        epoch_num = getattr(self, 'current_epoch', 'N/A') # Get epoch if available
        self.logger.info(f"Epoch {epoch_num}, Average Validation Loss: {avg_val_loss:.4f} ({num_val_batches} batches)")

        # Set models back to train mode
        if self.unet: self.unet.train()
        if self.text_encoder: self.text_encoder.train()
        if hasattr(self, 'text_encoder_2') and self.text_encoder_2: self.text_encoder_2.train()

        return avg_val_loss


    def save_lora_weights(self, save_type="epoch", epoch=None, val_loss=None, hyperparameters=None):
        """Saves the trained LoRA weights AND the projection layer."""
        if epoch is None: epoch = self.current_epoch
        save_label = f"{save_type}_epoch_{epoch}"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None and not np.isnan(val_loss) and not np.isinf(val_loss) else "N/A"
        self.logger.info(f"Saving model state for {self.model_name} as '{save_label}' (Val Loss: {val_loss_str})")

        output_subdir = os.path.join(self.output_dir, save_label)
        os.makedirs(output_subdir, exist_ok=True)
        save_paths = {}

        # --- Save UNet LoRA Adapters (if they exist) ---
        if self.unet and isinstance(self.unet, PeftModel): # Check if it's a PEFT model
            try:
                unet_path = os.path.join(output_subdir, "unet_lora")
                self.unet.save_pretrained(unet_path) # Saves only adapters
                save_paths["UNet_LoRA"] = unet_path
                self.logger.info(f"Saved UNet LoRA weights to {unet_path}")
            except Exception as e:
                self.logger.error(f"Failed to save UNet LoRA weights: {e}")
        elif self.unet:
             self.logger.info("UNet was not adapted with LoRA, skipping UNet LoRA save.")

        # --- Save Projection Layer State Dict ---
        if self.projection_layer:
            try:
                proj_path = os.path.join(output_subdir, "projection_layer.pth")
                torch.save(self.projection_layer.state_dict(), proj_path)
                save_paths["ProjectionLayer"] = proj_path
                self.logger.info(f"Saved Projection Layer state_dict to {proj_path}")
            except Exception as e:
                 self.logger.error(f"Failed to save Projection Layer state_dict: {e}")

        # --- Save Hyperparameters ---
        if hyperparameters:
            # Add info about which LoRA parts were applied
            hyperparameters['_apply_lora_text_encoder'] = self._apply_lora_text_flag
            hyperparameters['_apply_lora_unet'] = self._apply_lora_unet_flag
            hyperparam_path = os.path.join(output_subdir, f"training_args_{save_label}.json")
            save_data = { 'model_name': self.model_name, 'save_type': save_type, 'epoch': epoch,
                          'best_epoch': self.best_epoch if save_type == 'best' else None,
                          'validation_loss': val_loss if val_loss is not None else None,
                          'best_validation_loss': self.best_val_loss if self.best_val_loss != float('inf') else None,
                          'hyperparameters': hyperparameters }
            try:
                with open(hyperparam_path, 'w') as f: json.dump(save_data, f, indent=4, default=str)
                self.logger.info(f"Saved training args and metrics to {hyperparam_path}")
            except Exception as e: self.logger.error(f"Failed to save hyperparameters: {e}")

        if not save_paths:
             self.logger.warning(f"No trainable weights (LoRA/Projection) were saved for {self.model_name} ({save_label}).")


    def fine_tune(self, dataset_path, epochs=1, batch_size=1, learning_rate=1e-5, val_split=0.2):
        """Fine-tune the model with LoRA, including validation and checkpointing."""
        """Fine-tune the UNet LoRA adapters and the T5 projection layer."""
        self.logger.info(f"Starting T5-UNet fine-tuning for {self.model_name}...")
        self.logger.info(f"Dataset: {dataset_path}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}, Val Split: {val_split}")
        self.best_val_loss = float('inf'); self.best_epoch = -1

        # --- Model Component Checks ---
        if self.model_name != "karlo" and not self.unet: raise ValueError("UNet not loaded.")
        if not self.text_encoder: raise ValueError("Text Encoder (T5) not loaded.")
        if not self.projection_layer and self.model_name != "karlo": raise ValueError("Projection Layer not created.")
        if not self.tokenizer: raise ValueError("Tokenizer not loaded.")
        if not self.scheduler: raise ValueError("Scheduler not loaded.")
        if self.model_name in ["sdxl", "kandinsky"] and not self.vae: raise ValueError("VAE not loaded.")

        # Reset best validation loss for this run
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        
        try:
            data_dir = os.path.dirname(dataset_path)
            image_folder = os.path.join(data_dir, "images")
            self.logger.info(f"Expecting images in: {image_folder}")
            dataset = load_dataset(dataset_path)
            if not dataset:
                raise ValueError("load_dataset returned None or empty dataset.")
            self.logger.info(f"Loaded dataset with {len(dataset)} entries.")
            

            if val_split > 0 and val_split < 1:
                train_size = int((1.0 - val_split) * len(dataset))
                val_size = len(dataset) - train_size
                if train_size == 0 or val_size == 0:
                    self.logger.warning(f"Dataset split resulted in 0 samples for train ({train_size}) or validation ({val_size}). Training on full dataset.")
                    train_dataset = dataset
                    val_dataset = None
                else:
                    # Use random_split for better shuffling
                    train_dataset, val_dataset = torch.utils.data.random_split(
                        dataset, [train_size, val_size],
                        generator=torch.Generator().manual_seed(42) # for reproducible splits
                    )
            else:
                self.logger.warning("val_split is not between 0 and 1. Training on full dataset.")
                train_dataset = dataset
                val_dataset = None

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            self.logger.info(f"Training dataset size: {len(train_dataset)}")
            if val_dataset:
                self.logger.info(f"Validation dataset size: {len(val_dataset)}")
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Dataloader for validation
            else:
                val_dataloader = None

            
            params_to_optimize = []
            trainable_param_count = 0
            if self.projection_layer:
                params_to_optimize.extend(self.projection_layer.parameters())
                self.logger.info("Added Projection Layer parameters to optimizer.")
            # Add UNet LoRA parameters (if LoRA was applied)
            if self.unet and isinstance(self.unet, PeftModel):
                params_to_optimize.extend(filter(lambda p: p.requires_grad, self.unet.parameters()))
                self.logger.info("Added UNet LoRA parameters to optimizer.")

            optimizer = None
            if params_to_optimize:
                trainable_param_count = sum(p.numel() for p in params_to_optimize)
                self.logger.info(f"Total trainable parameters: {trainable_param_count}")
                try:
                    optimizer = AdamW8bit(params_to_optimize, lr=learning_rate)
                    self.logger.info(f"Using AdamW8bit optimizer with LR: {learning_rate}")
                except Exception as e:
                    self.logger.warning(f"Failed AdamW8bit init: {e}, using standard AdamW.")
                    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
            else:
                self.logger.error("No trainable parameters found! Cannot train."); return
            
            self.logger.info(f"Total trainable parameters: {trainable_param_count}")
            optimizer = AdamW8bit(params_to_optimize, lr=learning_rate)
            
            # Store hyperparams for saving
            hyperparameters = { 'model_name': self.model_name, 'text_encoder': t5_model_name, # Log T5 used
                                'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
                                'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': 0.1, # Assuming fixed LoRA params
                                'apply_lora_text_encoder': False, # Explicitly False
                                'apply_lora_unet': getattr(self, '_apply_lora_unet_flag', None) }

            
            global_step = 0
            for epoch in range(epochs):
                self.current_epoch = epoch + 1 # Update current epoch (1-based)
                self.logger.info(f"--- Starting Epoch {self.current_epoch}/{epochs} ---")
                # Set trainable models to train mode
                if self.unet: self.unet.train() # PEFT model handles base freeze
                if self.projection_layer: self.projection_layer.train()
                # Keep frozen models in eval
                if self.text_encoder: self.text_encoder.eval()
                if self.vae: self.vae.eval()

                total_train_loss = 0.0
                num_train_batches = 0
                for step, batch in enumerate(train_dataloader):
                    try:
                        image_filenames = batch['image']
                        prompts = batch['prompt']
                        if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
                            self.logger.warning(f"Skipping batch due to invalid prompts: {prompts}")
                            continue
                        pixel_values_list = []
                        valid_prompts = []
                        valid_indices = []
                        for i, (img_filename, prompt) in enumerate(zip(image_filenames, prompts)):
                            try:
                                image_path = os.path.join(image_folder, img_filename)
                                image = Image.open(image_path).convert('RGB')
                                image = image.resize((self.image_size, self.image_size))
                                image_np = np.array(image).astype(np.float32) / 255.0
                                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                                pixel_values_list.append(image_tensor)
                                valid_prompts.append(prompt)
                                valid_indices.append(i)
                            except Exception:
                                self.logger.warning(f"Error loading image {img_filename}")
                                continue
                        if not pixel_values_list:
                            self.logger.warning("Skipping batch as no valid images processed.")
                            continue
                        pixel_values = torch.stack(pixel_values_list).to(self.device, dtype=torch.float32)
                        prompts = valid_prompts
                        optimizer.zero_grad()
                        added_cond_kwargs = {}
                        target_values = None
                        loss = None
                        encoder_hidden_states = None
                        noisy_input = None
                        latents = None
                        # --- Text Encoding with T5 ---
                        try:
                            # Tokenize with T5 tokenizer, handle longer sequences
                            t5_max_length = self.tokenizer.model_max_length # Get T5's max length
                            inputs = self.tokenizer(
                                prompts, padding="max_length", max_length=t5_max_length,
                                truncation=True, return_tensors="pt"
                            ).to(self.device)
                            # Get T5 embeddings (ensure T5 is in eval mode and correct dtype)
                            with torch.no_grad():
                                t5_outputs = self.text_encoder(inputs.input_ids)
                                t5_hidden_states = t5_outputs.last_hidden_state.to(dtype=self.projection_layer.weight.dtype) # Match proj layer input dtype
                            # Project T5 embeddings
                            projected_embeddings = self.projection_layer(t5_hidden_states)
                            if torch.isnan(projected_embeddings).any() or torch.isinf(projected_embeddings).any():
                                self.logger.warning(f"NaN/Inf in projected embeddings Step {step+1}."); continue
                            encoder_hidden_states = projected_embeddings # Use projected as input to UNet
                        except Exception as text_err: self.logger.error(f"Text/Projection failed: {text_err}"); continue


                        if self.model_name in ["sdxl", "kandinsky"]:
                            pixel_values_norm = pixel_values * 2.0 - 1.0
                            if torch.isnan(pixel_values_norm).any():
                                continue
                            with torch.no_grad():
                                vae_output = self.vae.encode(pixel_values_norm.to(dtype=torch.float32))
                                if isinstance(self.vae, AutoencoderKL):
                                    latent_dist = vae_output.latent_dist
                                    latents = latent_dist.sample()
                                else:
                                    latents = vae_output.latents
                            scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)
                            latents = latents * scaling_factor
                            latents = latents.to(dtype=self.dtype)
                            noise = torch.randn_like(latents)
                            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()
                            noisy_input = self.scheduler.add_noise(latents, noise, timesteps)
                            target_values = noise
                            if self.model_name == "sdxl":
                                prompt_embeds, pooled_embeds = self._encode_prompt_sdxl(prompts)
                                if torch.isnan(prompt_embeds).any() or torch.isinf(prompt_embeds).any() or torch.isnan(pooled_embeds).any() or torch.isinf(pooled_embeds).any():
                                    continue
                                encoder_hidden_states = prompt_embeds
                                add_time_ids = self._get_add_time_ids(
                                    (self.image_size, self.image_size), (0,0), (self.image_size, self.image_size), dtype=prompt_embeds.dtype
                                )
                                add_time_ids = add_time_ids.repeat(len(prompts), 1).to(self.device)
                                added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}
                            else:
                                inputs = self.tokenizer(
                                    prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
                                ).to(self.device)
                                temp_hidden_states = self.text_encoder(inputs.input_ids)[0].to(dtype=self.dtype)
                                if torch.isnan(temp_hidden_states).any() or torch.isinf(temp_hidden_states).any():
                                    continue
                                encoder_hidden_states = temp_hidden_states
                                zero_image_embeds = torch.zeros((noisy_input.shape[0], 1280), dtype=self.dtype, device=self.device)
                                added_cond_kwargs["image_embeds"] = zero_image_embeds
                        elif self.model_name == "karlo":
                            inputs = self.tokenizer(
                                prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
                            ).to(self.device)
                            encoder_hidden_states = self.text_encoder(inputs.input_ids)[0]
                            if torch.isnan(encoder_hidden_states).any() or torch.isinf(encoder_hidden_states).any():
                                continue
                            loss = torch.nn.functional.mse_loss(encoder_hidden_states.float(), torch.zeros_like(encoder_hidden_states).float())
                            noisy_input = None
                            target_values = None

                        # --- Forward Pass ---
                        if self.model_name != "karlo":
                            if self.unet is None or noisy_input is None or encoder_hidden_states is None:
                                continue
                            unet_args = {
                                "sample": noisy_input.to(dtype=self.dtype),
                                "timestep": timesteps,
                                "encoder_hidden_states": encoder_hidden_states.to(dtype=self.dtype),
                                "added_cond_kwargs": added_cond_kwargs
                            }
                            model_pred = self.unet(**unet_args).sample
                            if hasattr(model_pred, "sample"):
                                model_pred = model_pred.sample
                            if model_pred.shape[1] == target_values.shape[1] * 2:
                                model_pred = model_pred[:, :target_values.shape[1], :, :]
                            if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                                continue
                            loss = torch.nn.functional.mse_loss(model_pred.float(), target_values.float(), reduction="mean")
                        
                        # --- Backward & Step ---
                        if loss is not None and not torch.isnan(loss):
                            loss.backward()
                            if params_to_optimize: torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                            optimizer.step()
                            total_train_loss += loss.item(); num_train_batches += 1
                            if global_step % 50 == 0: self.logger.info(f"Epoch {self.current_epoch}, Step {global_step}, Train Loss: {loss.item():.4f}")
                        else: self.logger.warning(f"Invalid loss ({loss}) Step {step+1}.")
                        global_step += 1

                    except Exception as e:
                        self.logger.error(f"Training step failed: {e}\n{traceback.format_exc()}")
                        continue
                # --- End of Epoch ---
                avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('nan')
                self.logger.info(f"Epoch {self.current_epoch}, Average Train Loss: {avg_train_loss:.4f}")

                # --- Validation Step ---
                if val_dataloader: # Check if validation is enabled
                    # !--- Pass dataset_path to validate ---!
                    avg_val_loss = self.validate(val_dataloader, dataset_path, batch_size=batch_size)
                    # Checkpoint saving based on validation loss
                    if avg_val_loss < self.best_val_loss:
                        self.logger.info(f"New best validation loss: {avg_val_loss:.4f} (Previous: {self.best_val_loss:.4f}). Saving checkpoint.")
                        self.best_val_loss = avg_val_loss
                        self.best_epoch = self.current_epoch
                        # Pass hyperparams to save function if needed
                        self.save_lora_weights(save_type="best", epoch=self.current_epoch, val_loss=avg_val_loss, hyperparameters=hyperparameters)
                else:
                    self.logger.info("Skipping validation.")
                    # Optionally save model every epoch if no validation
                    # self.save_lora_weights(epoch=self.current_epoch, hyperparameters=hyperparameters)

                gc.collect()
                torch.cuda.empty_cache()

            # --- End Training Loop ---
            self.logger.info(f"Finished training {epochs} epochs.")
            if self.best_epoch != -1:
                 self.logger.info(f"Best validation loss recorded: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
            else:
                 self.logger.info("No validation performed or no improvement detected.")

            # --- Save Last Epoch Weights ---
            self.logger.info(f"Saving final weights from last epoch ({self.current_epoch})...")
            last_val_loss = avg_val_loss if val_dataloader else None # Use last calculated val loss if available
            self.save_lora_weights(save_type="last", epoch=self.current_epoch, val_loss=last_val_loss, hyperparameters=hyperparameters)

        except Exception as e:
            self.logger.error(f"Fine-tuning process failed: {e}\n{traceback.format_exc()}")
            raise
        finally:
            self.logger.info(f"Cleaning up GPU memory after fine-tuning run.")
            # del optimizer, train_dataloader, val_dataloader, dataset # Let GC handle
            gc.collect(); torch.cuda.empty_cache()
