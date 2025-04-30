import os
import torch
import logging
import yaml
import traceback
import numpy as np
import json
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    UnCLIPPipeline,
    DPMSolverMultistepScheduler,
    VQModel,
    AutoencoderKL
)
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPVisionModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer, CLIPImageProcessor
from torch.utils.data import DataLoader, Subset
from PIL import Image
import gc
from src.utils.dataset import load_dataset, CocoFinetuneDataset
from itertools import chain
from bitsandbytes.optim import AdamW8bit

# Configure logging
logging.basicConfig(
    filename='/home/iris/Documents/deep_learning/src/logs/new_finetune.log',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s',
    filemode='w',
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def summarize_prompt(prompt, max_tokens=77):
    """Summarize string prompt for DeepFloyd IF if it exceeds max_tokens."""
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

def format_topic_prompt(prompt_template, topic_description):
    """Format the prompt template with the topic description."""
    try:
        desc_str = "; ".join([f"{key}: {value}" for key, value in topic_description.items() if value])
        return prompt_template.format(txt_prompt=desc_str)
    except Exception as e:
        logger.error(f"Failed to format prompt: {e}")
        return prompt_template.format(txt_prompt="")

class FinetuneModel:
    def __init__(self, model_name, output_dir, logger_instance=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.logger = logger_instance or logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.model = None
        self.pipeline = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.text_encoder = None
        self.vision_encoder = None
        self.text_encoder_2 = None
        self.unet = None
        self.scheduler = None
        self.vae = None
        self.prior = None
        self.decoder = None
        self.image_processor = None
        self.text_embedding_projection = None
        self.image_embedding_projection = None
        self.image_size = 1024
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.current_epoch = 0

    def load_model(self):
        """Load the specified diffusion model and its components."""
        self.logger.info(f"Loading model: {self.model_name}")
        try:
            import transformers
            import diffusers
            self.logger.info(f"Transformers version: {transformers.__version__}, Diffusers version: {diffusers.__version__}")
            if self.model_name == "sdxl":
                self.image_size = 1024
                vae_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                pipeline_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
                self.logger.info(f"Loading VAE in FP32 from {vae_model_id}, subfolder 'vae'")
                vae = AutoencoderKL.from_pretrained(
                    vae_model_id,
                    subfolder="vae",
                    torch_dtype=torch.float32,
                    cache_dir="/tmp/hf_cache"
                ).to(self.device)
                vae.eval()
                self.logger.info("Loaded VAE in FP32.")
                self.logger.info(f"Loading SDXL pipeline ({pipeline_model_id}) with FP32 VAE...")
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    pipeline_model_id,
                    vae=vae,
                    torch_dtype=self.dtype,
                    variant="fp16",
                    use_safetensors=True,
                    cache_dir="/tmp/hf_cache"
                ).to(self.device)
                self.pipeline = pipe
                self.tokenizer = pipe.tokenizer
                self.tokenizer_2 = pipe.tokenizer_2
                self.text_encoder = pipe.text_encoder
                self.text_encoder_2 = pipe.text_encoder_2
                self.unet = pipe.unet
                self.scheduler = pipe.scheduler
                self.vae = vae
                self.text_encoder.train()
                self.text_encoder_2.train()
                self.unet.train()
                self.logger.info("Assigned SDXL components (UNet/TextEncoders in FP16, VAE in FP32).")
            elif self.model_name == "kandinsky":
                self.image_size = 512
                decoder_model_id = "kandinsky-community/kandinsky-2-2-decoder"
                prior_model_id = "kandinsky-community/kandinsky-2-2-prior"
                self.logger.info(f"Loading Kandinsky VAE (MoVQ) in FP32 from {decoder_model_id}, subfolder 'movq'")
                vae = VQModel.from_pretrained(
                    decoder_model_id,
                    subfolder="movq",
                    torch_dtype=torch.float32,
                    cache_dir="/tmp/hf_cache"
                ).to(self.device)
                vae.eval()
                self.vae = vae
                self.logger.info("Successfully loaded Kandinsky VQModel (VAE) in FP32.")
                self.logger.info(f"Loading Kandinsky decoder pipeline ({decoder_model_id})...")
                decoder_pipe = DiffusionPipeline.from_pretrained(
                    decoder_model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    cache_dir="/tmp/hf_cache"
                ).to(self.device)
                self.pipeline = decoder_pipe
                self.unet = decoder_pipe.unet
                self.scheduler = decoder_pipe.scheduler
                self.logger.info("Loaded Kandinsky decoder pipeline.")
                self.logger.info(f"Loading Kandinsky text encoder/tokenizer from {prior_model_id}...")
                self.text_encoder = CLIPTextModel.from_pretrained(
                    prior_model_id, subfolder="text_encoder", torch_dtype=self.dtype,
                    use_safetensors=False, cache_dir="/tmp/hf_cache"
                ).to(self.device)
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    prior_model_id, subfolder="tokenizer", cache_dir="/tmp/hf_cache"
                )
                self.logger.info("Loaded Kandinsky text encoder/tokenizer.")
                self.unet.train()
                self.text_encoder.train()
                self.logger.info("Assigned Kandinsky components (UNet/TextEncoder in FP16, VAE in FP32).")
            elif self.model_name == "karlo":
                self.image_size = 256
                karlo_model_id = "kakaobrain/karlo-v1-alpha"
                clip_id = "openai/clip-vit-large-patch14"
                self.logger.info(f"Loading Karlo pipeline ({karlo_model_id})...")
                pipe = UnCLIPPipeline.from_pretrained(
                    karlo_model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    cache_dir="/tmp/hf_cache"
                )
                self.pipeline = pipe
                self.logger.info(f"Loading CLIP text encoder from {clip_id}...")
                self.text_encoder = CLIPTextModel.from_pretrained(
                    clip_id, torch_dtype=self.dtype, use_safetensors=False, cache_dir="/tmp/hf_cache"
                ).to(self.device)
                self.logger.info(f"Text encoder hidden size: {self.text_encoder.config.hidden_size}")
                self.logger.info(f"Loading CLIP vision encoder from {clip_id}...")
                self.vision_encoder = CLIPVisionModel.from_pretrained(
                    clip_id, torch_dtype=self.dtype, use_safetensors=False, cache_dir="/tmp/hf_cache"
                ).to(self.device)
                self.logger.info(f"Vision encoder hidden size: {self.vision_encoder.config.hidden_size}")
                self.tokenizer = pipe.tokenizer
                self.prior = pipe.prior.to(self.device)
                self.decoder = pipe.decoder.to(self.device)
                self.image_processor = CLIPImageProcessor()
                # Fallback projection layers
                if self.text_encoder.config.hidden_size != 768:
                    self.logger.warning(f"Text encoder hidden size is {self.text_encoder.config.hidden_size}, adding projection layer to map to 768")
                    self.text_embedding_projection = torch.nn.Linear(self.text_encoder.config.hidden_size, 768).to(self.device, dtype=self.dtype)
                if self.vision_encoder.config.hidden_size != 768:
                    self.logger.warning(f"Vision encoder hidden size is {self.vision_encoder.config.hidden_size}, adding projection layer to map to 768")
                    self.image_embedding_projection = torch.nn.Linear(self.vision_encoder.config.hidden_size, 768).to(self.device, dtype=self.dtype)
                self.text_encoder.train()
                self.vision_encoder.train()
                self.prior.train()
                self.decoder.train()
                self.logger.info("Loaded Karlo components (Text Encoder, Vision Encoder, Prior, Decoder in FP16).")
            elif self.model_name == "deepfloyd_if":
                self.image_size = 64
                if_model_id = "DeepFloyd/IF-I-XL-v1.0"
                self.logger.info(f"Loading DeepFloyd IF pipeline ({if_model_id})...")
                pipe = DiffusionPipeline.from_pretrained(
                    if_model_id,
                    variant="fp16",
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    cache_dir="/tmp/hf_cache"
                ).to(self.device)
                self.pipeline = pipe
                self.text_encoder = pipe.text_encoder
                self.tokenizer = pipe.tokenizer
                self.unet = pipe.unet
                self.scheduler = pipe.scheduler
                self.text_encoder.to(self.device)
                self.unet.to(self.device)
                self.logger.info("Loaded DeepFloyd IF components.")
                try:
                    self.unet.enable_gradient_checkpointing()
                    if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
                        self.text_encoder.gradient_checkpointing_enable()
                    self.logger.info("Enabled gradient checkpointing for DeepFloyd IF.")
                except Exception as e:
                    self.logger.warning(f"Could not enable gradient checkpointing: {e}")
            else:
                raise ValueError(f"Unknown model_name: {self.model_name}")
            self.logger.info(f"Finished loading and setting up {self.model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}\n{traceback.format_exc()}")
            raise

    def modify_architecture(self, apply_lora_to_text_encoder=True, apply_lora_to_unet=True):
        """Add LoRA layers to relevant components."""
        self.logger.info(f"Configuring LoRA for {self.model_name} (Apply TextEncoder: {apply_lora_to_text_encoder}, Apply UNet/Prior/Decoder: {apply_lora_to_unet})")
        lora_r = 8
        lora_alpha = 16
        lora_dropout = 0.1
        lora_bias = "none"
        try:
            if apply_lora_to_text_encoder and self.text_encoder and self.model_name != "deepfloyd_if":
                lora_config_text = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
                )
                self.text_encoder = get_peft_model(self.text_encoder, lora_config_text)
                self.logger.info(f"Applied LoRA to Text Encoder 1 ({self.model_name}, type: CLIP)")
                if self.model_name == "sdxl" and self.text_encoder_2:
                    lora_config_clip_2 = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=lora_bias,
                        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
                    )
                    self.text_encoder_2 = get_peft_model(self.text_encoder_2, lora_config_clip_2)
                    self.logger.info(f"Applied LoRA to Text Encoder 2 ({self.model_name}, type: CLIP)")
                if self.model_name == "karlo" and self.vision_encoder:
                    self.vision_encoder = get_peft_model(self.vision_encoder, lora_config_text)
                    self.logger.info(f"Applied LoRA to Vision Encoder ({self.model_name}, type: CLIP)")
            elif self.text_encoder:
                self.logger.info(f"Skipped applying LoRA to Text Encoder 1 for {self.model_name}")

            if apply_lora_to_unet:
                lora_config_diffusion = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=lora_bias,
                    target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2.proj"]
                )
                if self.unet:
                    self.unet = get_peft_model(self.unet, lora_config_diffusion)
                    self.logger.info(f"Applied LoRA to UNet ({self.model_name})")
                if self.model_name == "karlo":
                    if self.prior:
                        self.prior = get_peft_model(self.prior, lora_config_diffusion)
                        self.logger.info(f"Applied LoRA to Karlo Prior")
                    if self.decoder:
                        self.decoder = get_peft_model(self.decoder, lora_config_diffusion)
                        self.logger.info(f"Applied LoRA to Karlo Decoder")
            elif self.unet:
                self.logger.info(f"Skipped applying LoRA to UNet for {self.model_name}")
            elif self.model_name == "karlo":
                self.logger.info("Skipping LoRA application for Karlo Prior/Decoder")
        except Exception as e:
            self.logger.error(f"Failed to apply LoRA to {self.model_name}: {e}\n{traceback.format_exc()}")
            raise

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
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype).to(self.device)
        return add_time_ids

    def validate(self, dataloader, dataset_path):
        """Compute validation loss on a validation dataset."""
        self.logger.info(f"Running validation for {self.model_name}")
        total_loss = 0.0
        num_batches = 0
        if self.unet:
            self.unet.eval()
        if self.text_encoder:
            self.text_encoder.eval()
        if self.vision_encoder:
            self.vision_encoder.eval()
        if self.text_encoder_2:
            self.text_encoder_2.eval()
        if self.prior:
            self.prior.eval()
        if self.decoder:
            self.decoder.eval()
        if self.text_embedding_projection:
            self.text_embedding_projection.eval()
        if self.image_embedding_projection:
            self.image_embedding_projection.eval()
        with torch.no_grad():
            for batch in dataloader:
                try:
                    image_filenames = batch['image']
                    prompts = batch['prompt']
                    if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
                        continue
                    pixel_values_list = []
                    valid_prompts = []
                    image_folder = os.path.join(os.path.dirname(dataset_path), "images")
                    for img_filename, prompt in zip(image_filenames, prompts):
                        try:
                            image_path = os.path.join(image_folder, img_filename)
                            image = Image.open(image_path).convert('RGB')
                            image = image.resize((self.image_size, self.image_size))
                            image_np = np.array(image).astype(np.float32) / 255.0
                            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                            pixel_values_list.append(image_tensor)
                            valid_prompts.append(prompt)
                        except Exception:
                            continue
                    if not pixel_values_list:
                        continue
                    pixel_values = torch.stack(pixel_values_list).to(self.device, dtype=torch.float32)
                    prompts = valid_prompts
                    if self.model_name in ["sdxl", "kandinsky"]:
                        pixel_values_norm = pixel_values * 2.0 - 1.0
                        vae_output = self.vae.encode(pixel_values_norm.to(dtype=torch.float32))
                        if isinstance(self.vae, AutoencoderKL):
                            latent_dist = vae_output.latent_dist
                            latents = latent_dist.sample()
                        else:
                            latents = vae_output.latents
                        scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)
                        latents = latents * scaling_factor
                        latents = latents.to(dtype=self.dtype)
                        noise = torch.randn_like(latents).to(self.device)
                        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()
                        noisy_input = self.scheduler.add_noise(latents, noise, timesteps)
                        target_values = noise
                        if self.model_name == "sdxl":
                            prompt_embeds, pooled_embeds = self._encode_prompt_sdxl(prompts)
                            add_time_ids = self._get_add_time_ids(
                                (self.image_size, self.image_size), (0,0), (self.image_size, self.image_size), dtype=prompt_embeds.dtype
                            )
                            add_time_ids = add_time_ids.repeat(len(prompts), 1).to(self.device)
                            added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}
                            model_pred = self.unet(
                                noisy_input, timestep=timesteps, encoder_hidden_states=prompt_embeds, added_cond_kwargs=added_cond_kwargs
                            ).sample
                        else:
                            inputs = self.tokenizer(
                                prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
                            ).to(self.device)
                            encoder_hidden_states = self.text_encoder(inputs.input_ids)[0].to(dtype=self.dtype)
                            zero_image_embeds = torch.zeros(
                                (noisy_input.shape[0], 1280), dtype=self.dtype, device=self.device
                            )
                            added_cond_kwargs = {"image_embeds": zero_image_embeds}
                            model_pred = self.unet(
                                noisy_input, timestep=timesteps, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                            ).sample
                        if model_pred.shape[1] == target_values.shape[1] * 2:
                            model_pred = model_pred[:, :target_values.shape[1], :, :]
                        loss = torch.nn.functional.mse_loss(model_pred.float(), target_values.float(), reduction="mean")
                    elif self.model_name == "karlo":
                        image_inputs = self.image_processor(pixel_values, return_tensors="pt").to(self.device, dtype=self.dtype)
                        image_embeddings = self.vision_encoder(image_inputs['pixel_values']).last_hidden_state.to(self.device)
                        inputs = self.tokenizer(
                            prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
                        ).to(self.device)
                        text_embeddings = self.text_encoder(inputs.input_ids)[0].to(self.device)
                        self.logger.debug(f"Embedding shapes before projection: text_embeddings={text_embeddings.shape}, image_embeddings={image_embeddings.shape}")
                        # Apply projections if needed
                        if self.text_embedding_projection is not None:
                            text_embeddings = self.text_embedding_projection(text_embeddings)
                        if self.image_embedding_projection is not None:
                            image_embeddings = self.image_embedding_projection(image_embeddings)
                        self.logger.debug(f"Embedding shapes after projection: text_embeddings={text_embeddings.shape}, image_embeddings={image_embeddings.shape}")
                        prior_noise = torch.randn_like(image_embeddings).to(self.device)
                        prior_timesteps = torch.randint(0, 1000, (image_embeddings.shape[0],), device=self.device).long()
                        prior_noisy_input = torch.randn_like(image_embeddings).to(self.device)
                        self.logger.debug(f"Prior input devices: prior_noisy_input={prior_noisy_input.device}, prior_timesteps={prior_timesteps.device}, text_embeddings={text_embeddings.device}, image_embeddings={image_embeddings.device}")
                        prior_pred = self.prior(
                            prior_noisy_input, timestep=prior_timesteps, encoder_hidden_states=text_embeddings, proj_embedding=image_embeddings
                        ).sample
                        prior_loss = torch.nn.functional.mse_loss(prior_pred.float(), prior_noise.float(), reduction="mean")
                        pixel_values_norm = pixel_values * 2.0 - 1.0
                        decoder_noise = torch.randn_like(pixel_values_norm).to(self.device)
                        decoder_timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=self.device).long()
                        decoder_noisy_input = torch.randn_like(pixel_values_norm).to(self.device)
                        decoder_pred = self.decoder(
                            decoder_noisy_input, timestep=decoder_timesteps, encoder_hidden_states=image_embeddings
                        ).sample
                        decoder_loss = torch.nn.functional.mse_loss(decoder_pred.float(), decoder_noise.float(), reduction="mean")
                        loss = prior_loss + decoder_loss
                    elif self.model_name == "deepfloyd_if":
                        pixel_values = pixel_values.to(dtype=self.dtype)
                        noise = torch.randn_like(pixel_values).to(self.device)
                        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (pixel_values.shape[0],), device=self.device).long()
                        noisy_input = self.scheduler.add_noise(pixel_values, noise, timesteps)
                        target_values = noise
                        processed_prompts = [summarize_prompt(p, max_tokens=self.tokenizer.model_max_length) for p in prompts]
                        inputs = self.tokenizer(
                            processed_prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
                        ).to(self.device)
                        encoder_hidden_states = self.text_encoder(inputs.input_ids)[0]
                        if torch.isnan(encoder_hidden_states).any() or torch.isinf(encoder_hidden_states).any():
                            continue
                        model_pred = self.unet(
                            noisy_input, timestep=timesteps, encoder_hidden_states=encoder_hidden_states
                        ).sample
                        loss = torch.nn.functional.mse_loss(model_pred.float(), target_values.float(), reduction="mean")
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        num_batches += 1
                except Exception as e:
                    self.logger.error(f"Validation step failed: {e}\n{traceback.format_exc()}")
                    continue
        if self.unet:
            self.unet.train()
        if self.text_encoder:
            self.text_encoder.train()
        if self.vision_encoder:
            self.vision_encoder.train()
        if self.text_encoder_2:
            self.text_encoder_2.train()
        if self.prior:
            self.prior.train()
        if self.decoder:
            self.decoder.train()
        if self.text_embedding_projection:
            self.text_embedding_projection.train()
        if self.image_embedding_projection:
            self.image_embedding_projection.train()
        if num_batches == 0:
            self.logger.warning("No valid validation batches processed.")
            return float('inf')
        avg_loss = total_loss / num_batches
        self.logger.info(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def save_lora_weights(self, epoch, val_loss, hyperparameters):
        """Saves the trained weights and hyperparameters."""
        self.logger.info(f"Saving weights for {self.model_name} at epoch {epoch} with val loss {val_loss:.4f}")
        os.makedirs(self.output_dir, exist_ok=True)
        save_paths = {}
        if self.unet and hasattr(self.unet, 'save_pretrained') and any(p.requires_grad for p in self.unet.parameters()):
            unet_path = os.path.join(self.output_dir, f"unet_lora_epoch_{epoch}")
            self.unet.save_pretrained(unet_path)
            save_paths["UNet"] = unet_path
        if self.text_encoder and hasattr(self.text_encoder, 'save_pretrained') and any(p.requires_grad for p in self.text_encoder.parameters()):
            te1_path = os.path.join(self.output_dir, f"text_encoder_lora_epoch_{epoch}")
            self.text_encoder.save_pretrained(te1_path)
            save_paths["TextEncoder1"] = te1_path
        if self.model_name == "sdxl" and self.text_encoder_2 and hasattr(self.text_encoder_2, 'save_pretrained') and any(p.requires_grad for p in self.text_encoder_2.parameters()):
            te2_path = os.path.join(self.output_dir, f"text_encoder_2_lora_epoch_{epoch}")
            self.text_encoder_2.save_pretrained(te2_path)
            save_paths["TextEncoder2"] = te2_path
        if self.model_name == "karlo":
            if self.vision_encoder and hasattr(self.vision_encoder, 'save_pretrained') and any(p.requires_grad for p in self.vision_encoder.parameters()):
                vision_path = os.path.join(self.output_dir, f"vision_encoder_lora_epoch_{epoch}")
                self.vision_encoder.save_pretrained(vision_path)
                save_paths["VisionEncoder"] = vision_path
            if self.prior and hasattr(self.prior, 'save_pretrained') and any(p.requires_grad for p in self.prior.parameters()):
                prior_path = os.path.join(self.output_dir, f"prior_lora_epoch_{epoch}")
                self.prior.save_pretrained(prior_path)
                save_paths["Prior"] = prior_path
            if self.decoder and hasattr(self.decoder, 'save_pretrained') and any(p.requires_grad for p in self.decoder.parameters()):
                decoder_path = os.path.join(self.output_dir, f"decoder_lora_epoch_{epoch}")
                self.decoder.save_pretrained(decoder_path)
                save_paths["Decoder"] = decoder_path
        if save_paths:
            self.logger.info(f"Saved weights to:")
            for name, path in save_paths.items():
                self.logger.info(f"- {name}: {path}")
        else:
            self.logger.warning(f"No weights saved for {self.model_name}.")
        hyperparam_path = os.path.join(self.output_dir, f"hyperparameters_epoch_{epoch}.json")
        with open(hyperparam_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'val_loss': val_loss,
                'hyperparameters': hyperparameters
            }, f, indent=4)
        self.logger.info(f"Saved hyperparameters to {hyperparam_path}")

    def generate_image(self, prompt_template, topic_description, theory="cbt"):
        """Generate an image based on the prompt template and topic description."""
        self.logger.info(f"Generating image for {self.model_name} with theory {theory}")
        try:
            prompt = format_topic_prompt(prompt_template, topic_description)
            if self.model_name == "deepfloyd_if":
                prompt = summarize_prompt(prompt, max_tokens=self.tokenizer.model_max_length)
            if self.unet:
                self.unet.eval()
            if self.text_encoder:
                self.text_encoder.eval()
            if self.vision_encoder:
                self.vision_encoder.eval()
            if self.text_encoder_2:
                self.text_encoder_2.eval()
            if self.prior:
                self.prior.eval()
            if self.decoder:
                self.decoder.eval()
            with torch.no_grad():
                if self.model_name in ["sdxl", "kandinsky", "deepfloyd_if"]:
                    image = self.pipeline(
                        prompt=prompt,
                        height=self.image_size,
                        width=self.image_size,
                        num_inference_steps=50,
                        guidance_scale=7.5
                    ).images[0]
                elif self.model_name == "karlo":
                    image = self.pipeline(
                        prompt=prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5
                    ).images[0]
                    image = image.resize((1024, 1024))
            if self.unet:
                self.unet.train()
            if self.text_encoder:
                self.text_encoder.train()
            if self.vision_encoder:
                self.vision_encoder.train()
            if self.text_encoder_2:
                self.text_encoder_2.train()
            if self.prior:
                self.prior.train()
            if self.decoder:
                self.decoder.train()
            return image
        except Exception as e:
            self.logger.error(f"Failed to generate image for {self.model_name}: {e}\n{traceback.format_exc()}")
            raise

    def fine_tune(self, dataset_path, epochs=1, batch_size=1, learning_rate=1e-5, val_split=0.2):
        """Fine-tune the model with LoRA, including validation and checkpointing."""
        self.logger.info(f"Starting fine-tuning for {self.model_name}...")
        self.logger.info(f"Dataset: {dataset_path}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}, Val Split: {val_split}")
        try:
            data_dir = os.path.dirname(dataset_path)
            image_folder = os.path.join(data_dir, "images")
            self.logger.info(f"Expecting images in: {image_folder}")
            dataset = load_dataset(dataset_path)
            if not dataset:
                raise ValueError("load_dataset returned None or empty dataset.")
            self.logger.info(f"Loaded dataset with {len(dataset)} entries.")
            
            train_size = int((1 - val_split) * len(dataset))
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, len(dataset)))
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            self.logger.info(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            params_to_optimize = []
            trainable_param_count = 0
            if self.unet:
                unet_params = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
                params_to_optimize.extend(unet_params)
                trainable_param_count += sum(p.numel() for p in unet_params)
            if self.text_encoder:
                te1_params = list(filter(lambda p: p.requires_grad, self.text_encoder.parameters()))
                params_to_optimize.extend(te1_params)
                trainable_param_count += sum(p.numel() for p in te1_params)
            if self.model_name == "sdxl" and self.text_encoder_2:
                te2_params = list(filter(lambda p: p.requires_grad, self.text_encoder_2.parameters()))
                params_to_optimize.extend(te2_params)
                trainable_param_count += sum(p.numel() for p in te2_params)
            if self.model_name == "karlo":
                if self.vision_encoder:
                    vision_params = list(filter(lambda p: p.requires_grad, self.vision_encoder.parameters()))
                    params_to_optimize.extend(vision_params)
                    trainable_param_count += sum(p.numel() for p in vision_params)
                if self.prior:
                    prior_params = list(filter(lambda p: p.requires_grad, self.prior.parameters()))
                    params_to_optimize.extend(prior_params)
                    trainable_param_count += sum(p.numel() for p in prior_params)
                if self.decoder:
                    decoder_params = list(filter(lambda p: p.requires_grad, self.decoder.parameters()))
                    params_to_optimize.extend(decoder_params)
                    trainable_param_count += sum(p.numel() for p in decoder_params)
                if self.text_embedding_projection:
                    text_proj_params = list(filter(lambda p: p.requires_grad, self.text_embedding_projection.parameters()))
                    params_to_optimize.extend(text_proj_params)
                    trainable_param_count += sum(p.numel() for p in text_proj_params)
                if self.image_embedding_projection:
                    image_proj_params = list(filter(lambda p: p.requires_grad, self.image_embedding_projection.parameters()))
                    params_to_optimize.extend(image_proj_params)
                    trainable_param_count += sum(p.numel() for p in image_proj_params)
            
            if not params_to_optimize:
                self.logger.warning("No trainable parameters found.")
                return
            
            self.logger.info(f"Total trainable parameters: {trainable_param_count}")
            optimizer = AdamW8bit(params_to_optimize, lr=learning_rate)
            
            hyperparameters = {
                'model_name': self.model_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'lora_r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.1
            }
            
            global_step = 0
            for epoch in range(epochs):
                self.logger.info(f"--- Starting Epoch {epoch+1}/{epochs} ---")
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
                            noise = torch.randn_like(latents).to(self.device)
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
                            model_pred = self.unet(
                                noisy_input, timestep=timesteps, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
                            ).sample
                            if model_pred.shape[1] == target_values.shape[1] * 2:
                                model_pred = model_pred[:, :target_values.shape[1], :, :]
                            if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                                continue
                            loss = torch.nn.functional.mse_loss(model_pred.float(), target_values.float(), reduction="mean")
                        elif self.model_name == "karlo":
                            image_inputs = self.image_processor(pixel_values, return_tensors="pt").to(self.device, dtype=self.dtype)
                            image_embeddings = self.vision_encoder(image_inputs['pixel_values']).last_hidden_state.to(self.device)
                            inputs = self.tokenizer(
                                prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
                            ).to(self.device)
                            text_embeddings = self.text_encoder(inputs.input_ids)[0].to(self.device)
                            self.logger.debug(f"Embedding shapes before projection: text_embeddings={text_embeddings.shape}, image_embeddings={image_embeddings.shape}")
                            # Apply projections if needed
                            if self.text_embedding_projection is not None:
                                text_embeddings = self.text_embedding_projection(text_embeddings)
                            if self.image_embedding_projection is not None:
                                image_embeddings = self.image_embedding_projection(image_embeddings)
                            self.logger.debug(f"Embedding shapes after projection: text_embeddings={text_embeddings.shape}, image_embeddings={image_embeddings.shape}")
                            prior_noise = torch.randn_like(image_embeddings).to(self.device)
                            prior_timesteps = torch.randint(0, 1000, (image_embeddings.shape[0],), device=self.device).long()
                            prior_noisy_input = torch.randn_like(image_embeddings).to(self.device)
                            self.logger.debug(f"Prior input devices: prior_noisy_input={prior_noisy_input.device}, prior_timesteps={prior_timesteps.device}, text_embeddings={text_embeddings.device}, image_embeddings={image_embeddings.device}")
                            prior_pred = self.prior(
                                prior_noisy_input, timestep=prior_timesteps, encoder_hidden_states=text_embeddings, proj_embedding=image_embeddings
                            ).sample
                            prior_loss = torch.nn.functional.mse_loss(prior_pred.float(), prior_noise.float(), reduction="mean")
                            pixel_values_norm = pixel_values * 2.0 - 1.0
                            decoder_noise = torch.randn_like(pixel_values_norm).to(self.device)
                            decoder_timesteps = torch.randint(0, 1000, (pixel_values.shape[0],), device=self.device).long()
                            decoder_noisy_input = torch.randn_like(pixel_values_norm).to(self.device)
                            decoder_pred = self.decoder(
                                decoder_noisy_input, timestep=decoder_timesteps, encoder_hidden_states=image_embeddings
                            ).sample
                            decoder_loss = torch.nn.functional.mse_loss(decoder_pred.float(), decoder_noise.float(), reduction="mean")
                            loss = prior_loss + decoder_loss
                        elif self.model_name == "deepfloyd_if":
                            pixel_values = pixel_values.to(dtype=self.dtype)
                            noise = torch.randn_like(pixel_values).to(self.device)
                            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (pixel_values.shape[0],), device=self.device).long()
                            noisy_input = self.scheduler.add_noise(pixel_values, noise, timesteps)
                            target_values = noise
                            processed_prompts = [summarize_prompt(p, max_tokens=self.tokenizer.model_max_length) for p in prompts]
                            inputs = self.tokenizer(
                                processed_prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
                            ).to(self.device)
                            encoder_hidden_states = self.text_encoder(inputs.input_ids)[0]
                            if torch.isnan(encoder_hidden_states).any() or torch.isinf(encoder_hidden_states).any():
                                continue
                            model_pred = self.unet(
                                noisy_input, timestep=timesteps, encoder_hidden_states=encoder_hidden_states
                            ).sample
                            loss = torch.nn.functional.mse_loss(model_pred.float(), target_values.float(), reduction="mean")
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
                        if loss is not None and not torch.isnan(loss):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                            optimizer.step()
                            total_train_loss += loss.item()
                            num_train_batches += 1
                            if global_step % 50 == 0:
                                self.logger.info(f"Epoch {epoch+1}, Step {global_step}, Train Loss: {loss.item():.4f}")
                        global_step += 1
                    except Exception as e:
                        self.logger.error(f"Training step failed: {e}\n{traceback.format_exc()}")
                        continue
                if num_train_batches > 0:
                    avg_train_loss = total_train_loss / num_train_batches
                    self.logger.info(f"Epoch {epoch+1}, Average Train Loss: {avg_train_loss:.4f}")
                val_loss = self.validate(val_dataloader, dataset_path)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.save_lora_weights(epoch, val_loss, hyperparameters)
                gc.collect()
                torch.cuda.empty_cache()
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch+1}")
        except Exception as e:
            self.logger.error(f"Fine-tuning failed for {self.model_name}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            self.logger.info(f"Cleaning up GPU memory after fine-tuning {self.model_name}.")
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    model = FinetuneModel(model_name="karlo", output_dir="/home/iris/Documents/deep_learning/experiments/custom_finetuned/karlo", logger_instance=logger)
    model.load_model()
    model.modify_architecture()
    model.fine_tune(
        dataset_path="/home/iris/Documents/deep_learning/data/finetune_dataset/coco/dataset.json",
        epochs=5,
        batch_size=1,
        learning_rate=1e-5,
        val_split=0.2
    )