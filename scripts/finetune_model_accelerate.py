import os
import torch
import logging
import yaml
import traceback
import numpy as np
# Use StableDiffusionXLPipeline for more direct access if needed, AutoPipeline is fine too
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    UnCLIPPipeline, # For Karlo
    DPMSolverMultistepScheduler, # Example scheduler
    VQModel, # For Kandinsky VAE,
    AutoencoderKL # For SDXL VAE
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.logging import get_logger # Use accelerate logger
from accelerate.utils import set_seed # For reproducibility
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer, CLIPImageProcessor
from torch.utils.data import DataLoader
from PIL import Image
import gc
from src.utils.dataset import load_dataset, CocoFinetuneDataset
from itertools import chain # To combine parameters for optimizer
from bitsandbytes.optim import AdamW8bit

# Configure logging
logging.basicConfig(
    filename='/home/iris/Documents/deep_learning/src/logs/finetune.log',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s',
    filemode='w',
)
logger = logging.getLogger(__name__, log_level="INFO")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def summarize_prompt(prompt, max_tokens=77):
    """Summarize string prompt for DeepFloyd IF only if it exceeds max_tokens."""
    from transformers import pipeline
    
    try:
        if not prompt or not isinstance(prompt, str):
            logger.warning(f"Invalid prompt: {prompt}, returning empty string")
            return ""
        
        # Count tokens (approximate by splitting on whitespace)
        token_count = len(prompt.split())
        if token_count <= max_tokens:
            # logger.debug(f"Prompt has {token_count} tokens, no summarization needed: {prompt}")
            return prompt
        
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
        # Adjust max_length to be half the input length or a minimum of 10
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
    def __init__(self, model_name, output_dir, accelerator, logger_instance=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.accelerator = accelerator # Store accelerator instance
        self.logger = logger_instance or logging.getLogger(__name__)
        # self.device = accelerator.device # Device handled by accelerator
        # Precision handled by accelerator (e.g., 'fp16'), use float32 for specific loads like VAE
        self.target_dtype = torch.float16 # Target precision for most components

        self.tokenizer = None
        self.tokenizer_2 = None # For SDXL
        self.text_encoder = None
        self.text_encoder_2 = None # For SDXL
        self.unet = None
        self.scheduler = None
        self.vae = None # VAE needed for latent diffusion models (SDXL, Kandinsky)
        self.optimizer = None # Will be prepared later
        self.image_size = 1024 # Default, adjust per model


    def load_model(self):
        """Load model components, specifying initial dtypes where needed."""
        self.logger.info(f"Loading model: {self.model_name}")

        # Use accelerator.main_process_first context manager for downloads/preparation
        # to prevent multiple processes downloading simultaneously.
        with self.accelerator.main_process_first():
            try:
                # --- SDXL ---
                if self.model_name == "sdxl":
                    self.image_size = 1024
                    vae_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                    pipeline_model_id = "stabilityai/stable-diffusion-xl-base-1.0"

                    # Load VAE in FP32 (more stable), accelerator won't manage its precision unless prepared
                    self.logger.info(f"Loading VAE in FP32 from {vae_model_id}/vae")
                    self.vae = AutoencoderKL.from_pretrained(
                        vae_model_id, subfolder="vae", torch_dtype=torch.float32
                    )
                    # We will move VAE to device later, outside prepare if not training it

                    # Load other components with target dtype hint
                    self.logger.info(f"Loading SDXL UNet/TextEncoders ({pipeline_model_id}) with target dtype {self.target_dtype}...")
                    self.unet = UNet2DConditionModel.from_pretrained(
                        pipeline_model_id, subfolder="unet", torch_dtype=self.target_dtype, variant="fp16"
                    )
                    self.text_encoder = CLIPTextModel.from_pretrained(
                        pipeline_model_id, subfolder="text_encoder", torch_dtype=self.target_dtype, variant="fp16"
                    )
                    self.text_encoder_2 = CLIPTextModel.from_pretrained(
                         pipeline_model_id, subfolder="text_encoder_2", torch_dtype=self.target_dtype, variant="fp16"
                    )
                    self.tokenizer = CLIPTokenizer.from_pretrained(pipeline_model_id, subfolder="tokenizer")
                    self.tokenizer_2 = CLIPTokenizer.from_pretrained(pipeline_model_id, subfolder="tokenizer_2")
                    self.scheduler = DPMSolverMultistepScheduler.from_pretrained(pipeline_model_id, subfolder="scheduler")
                    self.logger.info("Loaded SDXL components.")

                # --- Kandinsky ---
                elif self.model_name == "kandinsky":
                    self.image_size = 512
                    decoder_model_id = "kandinsky-community/kandinsky-2-2-decoder"
                    prior_model_id = "kandinsky-community/kandinsky-2-2-prior"

                    # Load VQ VAE in FP32
                    self.logger.info(f"Loading Kandinsky VAE (MoVQ) in FP32 from {decoder_model_id}/movq")
                    self.vae = VQModel.from_pretrained(
                        decoder_model_id, subfolder="movq", torch_dtype=torch.float32
                    )

                    # Load UNet/Scheduler in target dtype
                    self.logger.info(f"Loading Kandinsky UNet/Scheduler ({decoder_model_id}) target dtype {self.target_dtype}...")
                    self.unet = UNet2DConditionModel.from_pretrained(
                        decoder_model_id, subfolder="unet", torch_dtype=self.target_dtype
                    )
                    self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
                        decoder_model_id, subfolder="scheduler"
                    )

                    # Load Text Encoder/Tokenizer in target dtype
                    self.logger.info(f"Loading Kandinsky text encoder/tokenizer from {prior_model_id}...")
                    self.text_encoder = CLIPTextModel.from_pretrained(
                        prior_model_id, subfolder="text_encoder", torch_dtype=self.dtype,
                    )
                    self.tokenizer = CLIPTokenizer.from_pretrained(
                         prior_model_id, subfolder="tokenizer"
                    )
                    self.logger.info("Loaded Kandinsky components.")


                # --- Karlo ---
                elif self.model_name == "karlo":
                    self.image_size = 64
                    clip_id = "openai/clip-vit-large-patch14"
                    self.text_encoder = CLIPTextModel.from_pretrained(clip_id, torch_dtype=self.dtype)
                    self.tokenizer = CLIPTokenizer.from_pretrained(clip_id)
                    self.unet, self.scheduler, self.vae = None, None, None
                    self.logger.warning("Karlo loading focuses on Text Encoder...")

                # --- DeepFloyd/IF ---
                elif self.model_name == "deepfloyd_if":
                     self.image_size = 64
                     if_model_id = "DeepFloyd/IF-I-XL-v1.0"
                     self.logger.info(f"Loading DeepFloyd/IF components ({if_model_id})...")

                     # Load UNet in target dtype (fp16)
                     self.unet = UNet2DConditionModel.from_pretrained(
                         if_model_id, subfolder="unet", torch_dtype=self.target_dtype
                     )
                     self.logger.info(f"Loaded DeepFloyd/IF UNet ({self.unet.dtype}).")

                     # Load Text Encoder (T5) - try fp16 first, relying on accelerate offload
                     # If NaN persists, change this to torch.float32
                     text_encoder_dtype = self.target_dtype # or torch.float32
                     self.logger.info(f"Loading DeepFloyd/IF Text Encoder ({text_encoder_dtype}).")
                     self.text_encoder = T5EncoderModel.from_pretrained(
                         if_model_id, subfolder="text_encoder", torch_dtype=text_encoder_dtype
                     )
                     self.logger.info(f"Loaded DeepFloyd/IF Text Encoder ({self.text_encoder.dtype}).")

                     self.tokenizer = T5Tokenizer.from_pretrained(if_model_id, subfolder="tokenizer")
                     self.scheduler = DPMSolverMultistepScheduler.from_pretrained(if_model_id, subfolder="scheduler")
                     self.vae = None
                     self.logger.info("Loaded DeepFloyd/IF Stage I components.")

                else:
                    raise ValueError(f"Unknown model_name: {self.model_name}")

                # Enable gradient checkpointing BEFORE prepare
                self.logger.info("Attempting to enable gradient checkpointing...")
                try:
                    if self.unet and hasattr(self.unet, "enable_gradient_checkpointing"):
                        self.unet.enable_gradient_checkpointing()
                        self.logger.info("Enabled gradient checkpointing for UNet.")
                    if self.text_encoder and hasattr(self.text_encoder, "gradient_checkpointing_enable"):
                        self.text_encoder.gradient_checkpointing_enable()
                        self.logger.info("Enabled gradient checkpointing for Text Encoder 1.")
                    if self.text_encoder_2 and hasattr(self.text_encoder_2, "gradient_checkpointing_enable"):
                        self.text_encoder_2.gradient_checkpointing_enable()
                        self.logger.info("Enabled gradient checkpointing for Text Encoder 2.")
                except Exception as e:
                    self.logger.warning(f"Could not enable gradient checkpointing: {e}")

                # Try xformers BEFORE prepare
                self.logger.info("Attempting to enable xformers...")
                try:
                    if self.unet and hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
                        self.unet.enable_xformers_memory_efficient_attention()
                        self.logger.info("xFormers enabled for UNet.")
                except Exception as e:
                    self.logger.warning(f"Could not enable xFormers for UNet: {e}.")

            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {e}\n{traceback.format_exc()}")
                raise

        # Move VAE to device manually if it exists (not part of training, so not prepared with optimizer)
        if self.vae is not None:
             self.vae.to(self.accelerator.device)
             self.vae.eval()

        self.logger.info(f"Finished loading {self.model_name} components.")
        
        
    def modify_architecture(self, apply_lora_to_text_encoder=True, apply_lora_to_unet=True):
        """Add LoRA layers to relevant components, skipping text encoder for DeepFloyd/IF."""
        self.logger.info(f"Configuring LoRA for {self.model_name} (Global Flags: Apply TextEncoder: {apply_lora_to_text_encoder}, Apply UNet: {apply_lora_to_unet})")

        # --- Determine if LoRA should *actually* be applied to the text encoder for THIS specific model ---
        # It should be applied only if the global flag is True AND the model is NOT deepfloyd_if
        apply_text_lora_here = apply_lora_to_text_encoder and (self.model_name != "deepfloyd_if")

        # Log clearly if skipping text LoRA for DeepFloyd specifically
        if apply_lora_to_text_encoder and not apply_text_lora_here: # Check if the flag was True but we are skipping anyway
             self.logger.info(f"Skipping Text Encoder LoRA specifically for DeepFloyd/IF model override.")
        # ---------------------------------------------------------------------------------------------

        # Base LoRA parameters
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.1
        lora_bias = "none"

        try:
            # --- Apply LoRA to Text Encoders ---
            # Use the calculated flag specific to this model run
            if apply_text_lora_here and self.text_encoder:
                # Determine config based on model type (excluding deepfloyd_if now)
                if self.model_name in ["sdxl", "kandinsky", "karlo"]:
                    lora_config_text = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=lora_bias,
                        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"] # CLIP targets
                    )
                    self.text_encoder = get_peft_model(self.text_encoder, lora_config_text)
                    self.logger.info(f"Applied LoRA to Text Encoder 1 ({self.model_name}, type: CLIP)")

                    # Apply to Text Encoder 2 for SDXL (also conditional on apply_text_lora_here)
                    if self.model_name == "sdxl" and self.text_encoder_2:
                        # Create another instance for the second encoder
                        lora_config_clip_2 = LoraConfig(
                             r=lora_r,
                             lora_alpha=lora_alpha,
                             lora_dropout=lora_dropout,
                             bias=lora_bias,
                             target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
                        )
                        self.text_encoder_2 = get_peft_model(self.text_encoder_2, lora_config_clip_2)
                        self.logger.info(f"Applied LoRA to Text Encoder 2 ({self.model_name}, type: CLIP)")

                # Note: No need for a deepfloyd_if elif block here, as apply_text_lora_here is False for it

            elif self.text_encoder: # Log if text encoder exists but LoRA wasn't applied
                 self.logger.info(f"Skipped applying LoRA to Text Encoder 1 for {self.model_name} (Flag was {apply_lora_to_text_encoder} / Is DeepFloyd: {self.model_name == 'deepfloyd_if'}).")


            # --- Apply LoRA to UNet --- (This logic remains unchanged)
            if apply_lora_to_unet and self.unet:
                unet_target_modules = [
                    "to_q", "to_k", "to_v", "to_out.0",
                    "proj_in", "proj_out",
                    "ff.net.0.proj", "ff.net.2.proj"
                ]
                lora_config_unet = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    target_modules=unet_target_modules
                )
                self.unet = get_peft_model(self.unet, lora_config_unet)
                self.logger.info(f"Applied LoRA to UNet ({self.model_name})")
            elif self.model_name == "karlo":
                 self.logger.info("Skipping UNet LoRA application for Karlo (No applicable UNet).")
            elif not apply_lora_to_unet and self.unet:
                 self.logger.info(f"Skipped applying LoRA to UNet for {self.model_name} (Flag was False).")


        except Exception as e:
            self.logger.error(f"Failed to apply LoRA to {self.model_name}: {e}\n{traceback.format_exc()}")
            raise

    """Keep LoRA for all models"""   
    # def modify_architecture(self, apply_lora_to_text_encoder=True, apply_lora_to_unet=True):
    #     """Add LoRA layers to relevant components."""
    #     self.logger.info(f"Configuring LoRA for {self.model_name} (Apply TextEncoder: {apply_lora_to_text_encoder}, Apply UNet: {apply_lora_to_unet})")

    #     # Base LoRA parameters
    #     lora_r = 8
    #     lora_alpha = 16
    #     lora_dropout = 0.1
    #     lora_bias = "none"

    #     try:
    #         # --- Apply LoRA to Text Encoders ---
    #         if apply_lora_to_text_encoder and self.text_encoder:
    #             if self.model_name in ["sdxl", "kandinsky", "karlo"]:
    #                 # Create LoRA config for CLIP Text Encoders
    #                 lora_config_clip = LoraConfig(
    #                     r=lora_r,
    #                     lora_alpha=lora_alpha,
    #                     lora_dropout=lora_dropout,
    #                     bias=lora_bias,
    #                     target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    #                 )
    #                 self.text_encoder = get_peft_model(self.text_encoder, lora_config_clip)
    #                 self.logger.info(f"Applied LoRA to Text Encoder 1 ({self.model_name}, type: CLIP)")
    #             elif self.model_name == "deepfloyd_if":
    #                 # Create LoRA config for T5 Text Encoder
    #                 lora_config_t5 = LoraConfig(
    #                     r=lora_r,
    #                     lora_alpha=lora_alpha,
    #                     lora_dropout=lora_dropout,
    #                     bias=lora_bias,
    #                     # T5 specific layers - check model architecture if needed
    #                     target_modules=["q", "k", "v", "o", "wi", "wo"] # Example targets for T5 attention/FFN
    #                 )
    #                 self.text_encoder = get_peft_model(self.text_encoder, lora_config_t5)
    #                 self.logger.info(f"Applied LoRA to Text Encoder ({self.model_name}, type: T5)")

    #             # Apply to Text Encoder 2 for SDXL
    #             if self.model_name == "sdxl" and self.text_encoder_2:
    #                 # Create *another* instance for the second encoder
    #                 lora_config_clip_2 = LoraConfig(
    #                      r=lora_r,
    #                      lora_alpha=lora_alpha,
    #                      lora_dropout=lora_dropout,
    #                      bias=lora_bias,
    #                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    #                 )
    #                 self.text_encoder_2 = get_peft_model(self.text_encoder_2, lora_config_clip_2)
    #                 self.logger.info(f"Applied LoRA to Text Encoder 2 ({self.model_name}, type: CLIP)")

    #         # --- Apply LoRA to UNet ---
    #         if apply_lora_to_unet and self.unet:
    #             # Create LoRA config for UNet
    #             unet_target_modules = [
    #                 "to_q", "to_k", "to_v", "to_out.0", # Basic attention projections
    #                 "proj_in", "proj_out",             # ResNet block projections
    #                 "ff.net.0.proj", "ff.net.2.proj"   # Feed-forward layers
    #                 # Add more specific attention blocks if needed
    #             ]
    #             lora_config_unet = LoraConfig(
    #                 r=lora_r,
    #                 lora_alpha=lora_alpha,
    #                 lora_dropout=lora_dropout,
    #                 bias=lora_bias,
    #                 target_modules=unet_target_modules
    #             )
    #             self.unet = get_peft_model(self.unet, lora_config_unet)
    #             self.logger.info(f"Applied LoRA to UNet ({self.model_name})")
    #         elif self.model_name == "karlo":
    #              self.logger.info("Skipping UNet LoRA application for Karlo (No applicable UNet).")


    #     except Exception as e:
    #         self.logger.error(f"Failed to apply LoRA to {self.model_name}: {e}\n{traceback.format_exc()}")
    #         raise

    # --- SDXL Specific Helper ---
    def _encode_prompt_sdxl(self, prompt_batch):
        """Encodes prompts for SDXL using both text encoders."""
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]
        prompt_embeds_list = []
        pooled_prompt_embeds_list = [] # Store pooled separately initially

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt_batch,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(self.device)

            # Get encoder outputs, requesting hidden states
            encoder_outputs = text_encoder(text_input_ids, output_hidden_states=True, return_dict=True)

            # Final hidden state (sequence embeddings)
            prompt_embeds = encoder_outputs.hidden_states[-2] # Penultimate recommended for SDXL
            prompt_embeds_list.append(prompt_embeds)

            # Pooled output
            # Handling differs slightly: text_encoder (CLIP ViT-L) has pooled output directly
            # text_encoder_2 (OpenCLIP ViT-G/14) requires taking the CLS token's output from last hidden state
            if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
                 pooled_prompt_embeds = encoder_outputs.pooler_output
            else:
                 # Use CLS token's embedding from the *last* hidden state for OpenCLIP-style models
                 pooled_prompt_embeds = encoder_outputs.last_hidden_state[:, 0, :] # [batch_size, hidden_dim]

            pooled_prompt_embeds_list.append(pooled_prompt_embeds)


        # Combine embeddings
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1) # [batch, seq_len, combined_dim]

        # Combine pooled embeddings (simple concatenation or averaging might work, check SDXL impl.)
        # Standard SDXL passes pooled output from text_encoder_2
        pooled_prompt_embeds = pooled_prompt_embeds_list[1] # Use pooled from the *second* encoder

        return prompt_embeds, pooled_prompt_embeds

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        """Helper to generate SDXL time IDs."""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # --- Fine-tuning Loop ---
    def fine_tune(self, dataset_path, epochs=1, batch_size=1, learning_rate=1e-5, gradient_accumulation_steps=1):
        """Fine-tune the model with LoRA using Accelerate."""
        self.logger.info(f"Starting fine-tuning with Accelerate...")
        self.logger.info(f"Dataset: {dataset_path}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}, Grad Accum: {gradient_accumulation_steps}")

        # Model loading checks
        if self.model_name != "karlo" and not self.unet: raise ValueError("UNet not loaded.")
        if not self.text_encoder: raise ValueError("Text Encoder not loaded.")

        try:
            # --- Dataset and Dataloader ---
            dataset = load_dataset(dataset_path)
            if not dataset: raise ValueError("load_dataset returned None.")
            self.logger.info(f"Loaded dataset with {len(dataset)} entries.")
            # Dataloader prepared later

            # --- Optimizer Setup ---
            params_to_optimize = []
            trainable_param_count = 0
            if self.unet and any(p.requires_grad for p in self.unet.parameters()):
                unet_params = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
                params_to_optimize.extend(unet_params); trainable_param_count += sum(p.numel() for p in unet_params)
                self.logger.info(f"Found {len(unet_params)} trainable params in UNet.")
            if self.text_encoder and any(p.requires_grad for p in self.text_encoder.parameters()):
                te1_params = list(filter(lambda p: p.requires_grad, self.text_encoder.parameters()))
                params_to_optimize.extend(te1_params); trainable_param_count += sum(p.numel() for p in te1_params)
                self.logger.info(f"Found {len(te1_params)} trainable params in Text Encoder 1.")
            # Add text_encoder_2 if needed

            if not params_to_optimize:
                self.logger.warning("Optimizer: No trainable parameters found.")
                optimizer = None
            else:
                 self.logger.info(f"Total trainable LoRA parameters: {trainable_param_count}")
                 try:
                     optimizer = AdamW8bit(params_to_optimize, lr=learning_rate)
                     self.logger.info(f"Using AdamW8bit optimizer with LR: {learning_rate}")
                 except ImportError:
                     self.logger.warning("bitsandbytes not installed, falling back to standard AdamW.")
                     optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
                 except Exception as e:
                      self.logger.warning(f"Failed to init AdamW8bit: {e}, using standard AdamW.")
                      optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

            # --- Prepare with Accelerator ---
            self.logger.info("Preparing components with Accelerator...")
            models_to_prep = []
            if self.unet and any(p.requires_grad for p in self.unet.parameters()): models_to_prep.append(self.unet)
            if self.text_encoder and any(p.requires_grad for p in self.text_encoder.parameters()): models_to_prep.append(self.text_encoder)
            # Add text_encoder_2 if needed

            # Create dataloader before prepare
            prepared_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            if models_to_prep and optimizer:
                prepared_components = self.accelerator.prepare(
                    *models_to_prep, optimizer, prepared_dataloader
                )
                # Distribute prepared components back
                component_iter = iter(prepared_components)
                if self.unet and any(p.requires_grad for p in self.unet.parameters()): self.unet = next(component_iter)
                if self.text_encoder and any(p.requires_grad for p in self.text_encoder.parameters()): self.text_encoder = next(component_iter)
                # Add text_encoder_2 if needed
                self.optimizer = next(component_iter)
                prepared_dataloader = next(component_iter) # Use the prepared dataloader
            elif models_to_prep:
                 self.logger.warning("No optimizer to prepare, preparing models only.")
                 prepared_models = self.accelerator.prepare(*models_to_prep)
                 if len(models_to_prep) == 1: prepared_models = (prepared_models,)
                 component_iter = iter(prepared_models)
                 if self.unet and any(p.requires_grad for p in self.unet.parameters()): self.unet = next(component_iter)
                 if self.text_encoder and any(p.requires_grad for p in self.text_encoder.parameters()): self.text_encoder = next(component_iter)
                 prepared_dataloader = self.accelerator.prepare(prepared_dataloader)
            else:
                 self.logger.warning("No models with trainable parameters found to prepare.")
                 prepared_dataloader = self.accelerator.prepare(prepared_dataloader)

            # Ensure VAE is on correct device (it wasn't prepared)
            if self.vae: self.vae.to(self.accelerator.device); self.vae.eval()

            num_update_steps_per_epoch = len(prepared_dataloader) // gradient_accumulation_steps
            num_train_epochs = epochs
            max_train_steps = num_train_epochs * num_update_steps_per_epoch

            self.logger.info("***** Running training *****")
            # ... (Log training details) ...

            global_step = 0
            # --- Training Loop ---
            for epoch in range(num_train_epochs):
                # Set models with trainable params to train mode
                if self.unet and any(p.requires_grad for p in self.unet.parameters()): self.unet.train()
                if self.text_encoder and any(p.requires_grad for p in self.text_encoder.parameters()): self.text_encoder.train()
                # Add text_encoder_2 if needed

                for step, batch in enumerate(prepared_dataloader):
                    # --- Manual Image Loading ---
                    try:
                        image_filenames = batch['image']; prompts = batch['prompt']
                        if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts): continue
                        pixel_values_list = []; valid_prompts = []
                        image_folder = os.path.join(os.path.dirname(dataset_path), "images")
                        for i, (img_filename, prompt) in enumerate(zip(image_filenames, prompts)):
                            try:
                                image_path = os.path.join(image_folder, img_filename)
                                image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
                                image_np = np.array(image).astype(np.float32) / 255.0
                                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                                pixel_values_list.append(image_tensor); valid_prompts.append(prompt)
                            except Exception as img_err: self.logger.warning(f"Skipping image {img_filename}: {img_err}")
                        if not pixel_values_list: continue
                        pixel_values = torch.stack(pixel_values_list).to(self.accelerator.device, dtype=torch.float32)
                        prompts = valid_prompts
                    except Exception as batch_proc_err: self.logger.error(f"Batch proc error: {batch_proc_err}"); continue

                    # --- Accumulate Gradients ---
                    # Pass first prepared model (or None if none were prepared)
                    first_prep_model = models_to_prep[0] if models_to_prep else None
                    with self.accelerator.accumulate(first_prep_model):
                        # --- Initialize ---
                        added_cond_kwargs = {}; target_values = None; loss = None
                        encoder_hidden_states = None; noisy_input = None; latents = None

                        # --- VAE Encoding ---
                        if self.model_name in ["sdxl", "kandinsky"]:
                            # ... (VAE logic as before) ...
                            pass # Placeholder

                        # --- Text Encoding ---
                        try:
                            if self.model_name == "sdxl":
                                # ... (SDXL text encoding) ...
                                pass # Placeholder
                            else: # Kandinsky, DeepFloyd, Karlo
                                if self.model_name == "deepfloyd_if": processed_prompts = [summarize_prompt(p) for p in prompts]
                                else: processed_prompts = prompts
                                inputs = self.tokenizer(processed_prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(self.accelerator.device)
                                # Accelerator handles precision for prepared models
                                temp_hidden_states = self.text_encoder(inputs.input_ids)[0]
                                if torch.isnan(temp_hidden_states).any() or torch.isinf(temp_hidden_states).any():
                                    self.logger.warning(f"NaN/Inf in {self.model_name} embeddings Step {step+1}."); continue
                                encoder_hidden_states = temp_hidden_states
                        except Exception as text_err: self.logger.error(f"Text encode fail: {text_err}"); continue

                        # --- Prepare Noise/Target ---
                        if self.model_name != "karlo":
                            if self.model_name in ["sdxl", "kandinsky"]: # Latent space
                                if latents is None: self.logger.error("Latents are None before noise step."); continue
                                noise = torch.randn_like(latents) # Use latents shape/dtype from VAE processing
                                noisy_input = self.scheduler.add_noise(latents, noise, timesteps) # timesteps need generation
                            elif self.model_name == "deepfloyd_if": # Pixel space
                                # pixel_values should be fp16 here for noise generation if target_dtype is fp16
                                pixel_values_target_dtype = pixel_values.to(dtype=self.target_dtype)
                                noise = torch.randn_like(pixel_values_target_dtype)
                                timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (pixel_values_target_dtype.shape[0],), device=self.accelerator.device).long()
                                noisy_input = self.scheduler.add_noise(pixel_values_target_dtype, noise, timesteps)
                            if noisy_input is None or torch.isnan(noisy_input).any() or torch.isinf(noisy_input).any(): self.logger.error(f"NaN/Inf in noisy_input Step {step+1}."); continue
                            target_values = noise # Target is the noise added

                        # --- Forward Pass ---
                        if self.model_name == "karlo":
                            if encoder_hidden_states is None: continue
                            loss = torch.nn.functional.mse_loss(encoder_hidden_states.float(), torch.zeros_like(encoder_hidden_states).float())
                        else: # UNet models
                            if noisy_input is None or encoder_hidden_states is None or self.unet is None: continue
                            unet_args = { "sample": noisy_input, "timestep": timesteps, "encoder_hidden_states": encoder_hidden_states, "added_cond_kwargs": {} }
                            current_added_conds = {}
                            if self.model_name == "sdxl": current_added_conds = added_cond_kwargs
                            elif self.model_name == "kandinsky":
                                image_embed_dim = 1280; batch_size_current = noisy_input.shape[0]
                                zero_image_embeds = torch.zeros( (batch_size_current, image_embed_dim), dtype=self.unet.dtype, device=self.accelerator.device )
                                current_added_conds["image_embeds"] = zero_image_embeds
                            unet_args["added_cond_kwargs"] = current_added_conds

                            unet_output = self.unet(**unet_args) # Precision handled by accelerate
                            model_pred = unet_output.sample if hasattr(unet_output, "sample") else unet_output
                            if model_pred.shape[1] != target_values.shape[1]:
                                if model_pred.shape[1] == target_values.shape[1] * 2: model_pred = model_pred[:, :target_values.shape[1], :, :]
                                else: self.logger.error("Shape mismatch after slice."); continue
                            if torch.isnan(model_pred).any() or torch.isinf(model_pred).any(): self.logger.error(f"NaN/Inf in model_pred Step {step+1}."); continue
                            loss = torch.nn.functional.mse_loss(model_pred.float(), target_values.float(), reduction="mean")

                        if loss is None or torch.isnan(loss) or torch.isinf(loss):
                            self.logger.warning(f"Invalid loss ({loss}) Step {step+1}."); continue

                        # --- Backward Pass ---
                        self.accelerator.backward(loss / gradient_accumulation_steps)

                        # --- Optimizer Step ---
                        if self.accelerator.sync_gradients:
                            if params_to_optimize: self.accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                            if self.optimizer: self.optimizer.step(); self.optimizer.zero_grad()

                    # --- End Accumulate Block ---

                    global_step += 1
                    # Logging
                    if self.accelerator.is_main_process and self.accelerator.sync_gradients:
                        if global_step % (50 * gradient_accumulation_steps) == 0:
                             gathered_loss = self.accelerator.gather(loss.detach()).mean()
                             self.logger.info(f"Epoch {epoch+1}, Opt Step {global_step // gradient_accumulation_steps}, Loss: {gathered_loss.item():.4f}")

                # End Epoch
                self.logger.info(f"--- Finished Epoch {epoch+1} ---")

            # --- Save Final Weights ---
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.save_lora_weights_accelerate()

        except Exception as e:
             self.logger.error(f"Fine-tuning failed: {e}\n{traceback.format_exc()}")
             raise
        finally:
             self.logger.info(f"Cleaning up...")
             # del optimizer, dataloader, dataset # Handled by accelerate? Check docs.
             gc.collect(); torch.cuda.empty_cache()



                    

    def save_lora_weights_accelerate(self):
        """Saves the trained LoRA weights unwrapping from Accelerator."""
        self.logger.info(f"Saving LoRA weights for {self.model_name} to {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        save_paths = {}

        # Helper to save unwrapped model
        def save_model(model_attr, name):
            model_instance = getattr(self, model_attr, None)
            # Check if it's a PEFT model by checking for save_pretrained on the base model
            if model_instance and hasattr(model_instance, 'save_pretrained'): # Simpler check for PEFT model
                try:
                    unwrapped_model = self.accelerator.unwrap_model(model_instance)
                    save_path = os.path.join(self.output_dir, f"{name}_lora")
                    # Ensure the unwrapped model still has save_pretrained (it should if it's PEFT)
                    if hasattr(unwrapped_model, 'save_pretrained'):
                        unwrapped_model.save_pretrained(save_path)
                        save_paths[name.capitalize()] = save_path
                        self.logger.info(f"Saved {name} LoRA weights to {save_path}")
                    else:
                        self.logger.warning(f"Unwrapped {name} model does not have save_pretrained method.")
                except Exception as e:
                    self.logger.error(f"Failed to save {name} LoRA: {e}")
            elif model_instance:
                 self.logger.warning(f"Could not save {name} LoRA (model found but not PEFT or missing save_pretrained).")
            else:
                 self.logger.warning(f"Could not save {name} LoRA (model attribute not found).")

        # Save components that might have LoRA
        save_model('unet', 'unet')
        save_model('text_encoder', 'text_encoder')
        if self.model_name == "sdxl":
            save_model('text_encoder_2', 'text_encoder_2')

        if not save_paths:
             self.logger.warning(f"No LoRA weights were saved for {self.model_name}.")



# --- Main Execution ---
def main():
    # --- Accelerator Setup ---
    gradient_accumulation_steps = 4 # Example: Accumulate over 4 steps
    # !--- Ensure Accelerator is instantiated correctly ---!
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16', # Or 'bf16' or 'no'
        log_with="tensorboard", # Optional
        project_dir=os.path.join("/home/iris/Documents/deep_learning/logs", "accelerate_logs") # Example path
    )
    # Setup logging properly only on main process after accelerator init
    logger.info(accelerator.state, main_process_only=False) # Log state on all processes
    if not accelerator.is_local_main_process:
        logger.setLevel(logging.WARNING) # Only show warnings and errors on other processes

    try:
        config_path = "/home/iris/Documents/deep_learning/config/config.yaml" # Make sure path is correct
        config = load_config(config_path) # Config might hold paths, params etc.

        # Ensure dataset path exists and is correct
        dataset_path = config.get("dataset_path", "/home/iris/Documents/deep_learning/data/finetune_dataset/coco/dataset.json")
        if not os.path.exists(dataset_path):
             logger.error(f"Dataset JSON file not found at: {dataset_path}")
             return # Exit if dataset manifest is missing

        # Ensure base output directory exists
        base_output_dir = config.get("base_output_dir", "/home/iris/Documents/deep_learning/experiments/custom_finetuned")
        os.makedirs(base_output_dir, exist_ok=True)

        # Define models and their specific output directories
        models_to_train = [
            # ("sdxl", os.path.join(base_output_dir, "sdxl")),
            # ("kandinsky", os.path.join(base_output_dir, "kandinsky")),
            # ("karlo", os.path.join(base_output_dir, "karlo")),
            ("deepfloyd_if", os.path.join(base_output_dir, "deepfloyd_if"))
        ]

        # Get training parameters from config or use defaults
        epochs = config.get("training_epochs", 1) # Default to 1 epoch if not in config
        batch_size = config.get("batch_size", 1)
        learning_rate = config.get("learning_rate", 1e-6)
        # Allow controlling which components get LoRA via config
        # Force LoRA OFF for Text Encoder ---!
        apply_lora_text = False # Set explicitly to False for this run
        apply_lora_unet = config.get("apply_lora_to_unet", True) # Keep UNet LoRA as configured
        logger.info(f"LoRA Config FORCED: Apply to TextEncoder = {apply_lora_text}, Apply to UNet = {apply_lora_unet}")
        # !----------------------------------------------!


        for model_name, output_dir in models_to_train:
            logger.info(f"========== Starting Pipeline for: {model_name} ==========")
            # Use accelerator to create dirs only on main process
            if accelerator.is_main_process:
                os.makedirs(current_output_dir, exist_ok=True)

            logger.info(f"========== Starting Pipeline for: {model_name} ==========")
            finetuner = None # Ensure defined for finally block
            try:
                finetuner = FinetuneModel(model_name, output_dir, accelerator, logger_instance=logger)
                # Load the base model architecture
                finetuner.load_model()

                # Apply LoRA modifications
                finetuner.modify_architecture(
                     apply_lora_to_text_encoder=apply_lora_text,
                     apply_lora_to_unet=apply_lora_unet
                )

                # Start the fine-tuning process
                finetuner.fine_tune(
                    dataset_path,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )

                logger.info(f"Successfully completed fine-tuning for {model_name}.")

            except Exception as e:
                # Log the error, but continue to the next model
                logger.error(f"!!!!!!!!!! Pipeline FAILED for {model_name}: {e} !!!!!!!!!\n{traceback.format_exc()}", exc_info=False) # exc_info=False as we add traceback manually

            finally:
                # Cleanup resources for the current model
                logger.info(f"--- Cleaning up resources for {model_name} ---")
                if finetuner:
                    # Explicitly delete components to help GC, especially large models on GPU
                    # Use try-except for del in case attribute doesn't exist
                    try: del finetuner.model
                    except AttributeError: pass
                    try: del finetuner.tokenizer
                    except AttributeError: pass
                    try: del finetuner.tokenizer_2
                    except AttributeError: pass
                    try: del finetuner.text_encoder
                    except AttributeError: pass
                    try: del finetuner.text_encoder_2
                    except AttributeError: pass
                    try: del finetuner.unet
                    except AttributeError: pass
                    try: del finetuner.scheduler
                    except AttributeError: pass
                    try: del finetuner.vae
                    except AttributeError: pass
                    del finetuner # Delete the instance itself
                gc.collect() # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() # Clear GPU cache
                logger.info(f"--- Finished cleanup for {model_name} ---")
                logger.info(f"========== Finished Pipeline for: {model_name} ==========\n")


    except Exception as main_err:
         logger.critical(f"Main execution failed: {main_err}\n{traceback.format_exc()}", exc_info=False)


if __name__ == "__main__":
    # The logging setup is now done globally at the start of the script
    logger.info("Script execution started.")
    main()
    logger.info("Script execution finished.")