import os
import torch
import logging
import yaml
import traceback
import numpy as np
import json
import math
import copy
import warnings
import types
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger

# Diffusers imports
from diffusers import (
    KandinskyV22PriorPipeline,
    KandinskyV22Pipeline,
    VQModel,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.models import PriorTransformer

# Transformers imports
from transformers import T5EncoderModel, T5Tokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor

# PEFT imports
from peft import LoraConfig, get_peft_model, PeftModel

# Dataloader and PIL
from torch.utils.data import DataLoader
from PIL import Image
import gc

# Local dataset import
try:
    from src.utils.dataset import load_dataset, CocoFinetuneDataset
except ImportError:
    print("Warning: Could not import dataset utilities from src.utils.dataset.")
    class CocoFinetuneDataset(torch.utils.data.Dataset): pass
    def load_dataset(path, splits): return []

# Optimizer
try:
    from bitsandbytes.optim import AdamW8bit
except ImportError:
    print("Warning: bitsandbytes not found. AdamW8bit optimizer unavailable.")
    AdamW8bit = torch.optim.AdamW

# --- Configuration ---
warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")

LOG_FILE = '/home/iris/Documents/deep_learning/src/logs/bigger_kandinsky_unet_prior_fixed.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s %(message)s',
    filemode='w',
)
logger = logging.getLogger(__name__)
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

# --- Custom Prior Pipeline ---
class KandinskyV22PriorPipelineWithT5(KandinskyV22PriorPipeline):
    def __init__(self, text_encoder, tokenizer, prior, scheduler, image_encoder, image_processor, logger_instance=None):
        super().__init__(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prior=prior,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor
        )
        self.logger = logger_instance or logging.getLogger(__name__)
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 512
        self.prior_embed_dim = getattr(prior.config, 'embedding_dim', 1280)
        self.projection_layer = nn.Linear(768, self.prior_embed_dim).to(
            device=self.prior.device, dtype=self.prior.dtype
        )
        nn.init.normal_(self.projection_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.projection_layer.bias)
        self.prior = prior
        if not hasattr(self.prior, 'logger'):
            self.prior.logger = self.logger

        def attention_hook(module, input, output):
            attn_scores = output[1] if isinstance(output, tuple) else output
            self.logger.debug(f"T5 block 1 attention scores stats - min: {attn_scores.min().item():.4f}, max: {attn_scores.max().item():.4f}, mean: {attn_scores.mean().item():.4f}")
            return output

        def dense_relu_dense_hook(module, input, output):
            output = output / 1000.0
            self.logger.debug(f"T5 block 1 DenseReluDense output stats - min: {output.min().item():.4f}, max: {output.max().item():.4f}, mean: {output.mean().item():.4f}")
            return output

        if hasattr(self.text_encoder, 'encoder') and len(self.text_encoder.encoder.block) > 1:
            self.text_encoder.encoder.block[1].register_forward_hook(attention_hook)
            self.text_encoder.encoder.block[1].layer[1].DenseReluDense.register_forward_hook(dense_relu_dense_hook)

        def custom_forward(
            self_prior,
            hidden_states,
            timestep,
            encoder_hidden_states=None,
            proj_embedding=None,
            attention_mask=None,
            **kwargs,
        ):
            if not hasattr(self_prior, 'logger'): self_prior.logger = logging.getLogger(__name__)
            prior_dtype = next(self_prior.parameters()).dtype
            hidden_states = hidden_states.to(prior_dtype)
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
            if proj_embedding is not None:
                proj_embedding = proj_embedding.to(prior_dtype)
                proj_embedding = torch.nan_to_num(proj_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(prior_dtype)
                encoder_hidden_states = torch.nan_to_num(encoder_hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
            if proj_embedding is not None:
                hidden_states = torch.cat([hidden_states, proj_embedding], dim=1)
            if encoder_hidden_states is not None:
                hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
            if hasattr(self_prior, 'proj_in'):
                hidden_states = self_prior.proj_in(hidden_states)
                hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                self_prior.logger.warning("PriorTransformer has no proj_in attribute!")
                raise AttributeError("Missing 'proj_in' attribute in PriorTransformer")
            if hasattr(self_prior, 'positional_embedding'):
                pos_embedding_module = self_prior.positional_embedding
                pos_embedding_weights = None
                if isinstance(pos_embedding_module, torch.nn.Embedding):
                    pos_embedding_weights = pos_embedding_module.weight
                elif isinstance(pos_embedding_module, torch.nn.Parameter):
                    pos_embedding_weights = pos_embedding_module
                elif isinstance(pos_embedding_module, torch.Tensor):
                    pos_embedding_weights = pos_embedding_module
                else:
                    self_prior.logger.warning(f"Unexpected type for self.positional_embedding: {type(pos_embedding_module)}")
                    pos_embedding_weights = pos_embedding_module
                if pos_embedding_weights is not None:
                    seq_length = hidden_states.shape[1]
                    positional_embedding_sliced = pos_embedding_weights[:, :seq_length, :]
                    positional_embedding_sliced = torch.nan_to_num(positional_embedding_sliced, nan=0.0, posinf=1.0, neginf=-1.0)
                    hidden_states = hidden_states + positional_embedding_sliced.to(hidden_states.dtype)
                    hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
                else:
                    self_prior.logger.error("Could not get positional_embedding weights!")
                    raise AttributeError("Missing positional_embedding weights")
            else:
                self_prior.logger.error("PriorTransformer object has no attribute 'positional_embedding'")
                raise AttributeError("Missing 'positional_embedding'")
            if attention_mask is not None:
                seq_len = hidden_states.shape[1]
                batch_size = hidden_states.shape[0]
                attention_mask = attention_mask[:, :seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.expand(-1, 1, seq_len, seq_len)
                attention_mask = (1 - attention_mask) * -10000.0
                attention_mask = torch.nan_to_num(attention_mask, nan=0.0, posinf=1.0, neginf=-1.0)
            if hasattr(self_prior, 'transformer_blocks'):
                for i, block in enumerate(self_prior.transformer_blocks):
                    hidden_states = block(hidden_states, attention_mask=attention_mask)
                    hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                self_prior.logger.error("PriorTransformer object has no attribute 'transformer_blocks'")
                raise AttributeError("Missing 'transformer_blocks'")
            if hasattr(self_prior, 'proj_to_clip_embeddings'):
                pooled_output = hidden_states[:, 0]
                predicted_image_embedding = self_prior.proj_to_clip_embeddings(pooled_output)
                predicted_image_embedding = torch.nan_to_num(predicted_image_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                self_prior.logger.error("PriorTransformer instance is missing 'proj_to_clip_embeddings' layer.")
                raise AttributeError("Missing 'proj_to_clip_embeddings' in PriorTransformer")
            return predicted_image_embedding

        self.prior.forward = types.MethodType(custom_forward, self.prior)
        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prior=self.prior,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor
        )

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        if isinstance(prompt, str): prompt = [prompt]
        if not all(isinstance(p, str) and p.strip() for p in prompt):
            self.logger.warning(f"Invalid prompts detected: {prompt}. Using default prompt.")
            prompt = ["a generic description"] * batch_size
        self.logger.debug(f"Tokenizer model_max_length: {self.tokenizer.model_max_length}")
        text_inputs = self.tokenizer(
            prompt, padding="max_length", max_length=512,
            truncation=True, return_tensors="pt",
        ).to(device)
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        self.logger.debug(f"Input IDs stats - min: {input_ids.min().item()}, max: {input_ids.max().item()}, mean: {input_ids.float().mean().item():.4f}")
        self.logger.debug(f"Input IDs sample: {input_ids[0, :10].tolist()}")
        self.logger.debug(f"Input IDs length: {input_ids.shape[1]}")
        self.logger.debug(f"Attention mask stats - min: {attention_mask.min().item()}, max: {attention_mask.max().item()}, mean: {attention_mask.float().mean().item():.4f}")
        with torch.no_grad():
            encoder = self.text_encoder.encoder
            hidden_states = self.text_encoder.shared(input_ids) / 1000.0
            self.logger.debug(f"T5 embedding output stats - min: {hidden_states.min().item():.4f}, max: {hidden_states.max().item():.4f}, mean: {hidden_states.mean().item():.4f}")
            for i, block in enumerate(encoder.block):
                hidden_states = hidden_states / (hidden_states.std(dim=-1, keepdim=True).clamp(min=1e-6) * 50.0)
                hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
                scale = 10000.0 if i in [4, 6, 8] else 1000.0
                hidden_states = hidden_states / scale
                hidden_states = torch.clamp(hidden_states, min=-500.0, max=500.0)
                self.logger.debug(f"T5 block {i} output stats - min: {hidden_states.min().item():.4f}, max: {hidden_states.max().item():.4f}, mean: {hidden_states.mean().item():.4f}")
            text_encoder_output = hidden_states
        last_hidden_state = text_encoder_output
        self.logger.debug(f"Last hidden state shape: {last_hidden_state.shape}")
        self.logger.debug(f"Last hidden state stats - min: {last_hidden_state.min().item():.4f}, max: {last_hidden_state.max().item():.4f}, mean: {last_hidden_state.mean().item():.4f}")
        prompt_embeds = last_hidden_state.mean(dim=1)
        target_dtype = self.projection_layer.weight.dtype
        prompt_embeds = prompt_embeds.to(target_dtype)
        prompt_embeds = torch.nan_to_num(prompt_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        self.logger.debug(f"Prompt embeds before norm stats - min: {prompt_embeds.min().item():.4f}, max: {prompt_embeds.max().item():.4f}, mean: {prompt_embeds.mean().item():.4f}")
        std = prompt_embeds.std(dim=1, keepdim=True)
        if (std > 1e-6).all():
            prompt_embeds = prompt_embeds / (std.clamp(min=1e-6) * 10.0)
        else:
            self.logger.warning("Zero or near-zero std in prompt_embeds. Using random embeddings.")
            prompt_embeds = torch.randn_like(prompt_embeds) * 0.05
        self.logger.debug(f"Prompt embeds after norm stats - min: {prompt_embeds.min().item():.4f}, max: {prompt_embeds.max().item():.4f}, mean: {prompt_embeds.mean().item():.4f}")
        prompt_embeds = self.projection_layer(prompt_embeds)
        prompt_embeds = torch.nan_to_num(prompt_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        text_sequence_length = 77
        prompt_embeds = prompt_embeds.unsqueeze(1).repeat(1, text_sequence_length, 1)
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or [""] * batch_size
            if isinstance(negative_prompt, str): negative_prompt = [negative_prompt] * batch_size
            negative_inputs = self.tokenizer(
                negative_prompt, padding="max_length", max_length=512,
                truncation=True, return_tensors="pt",
            ).to(device)
            negative_input_ids = negative_inputs.input_ids
            negative_attention_mask = negative_inputs.attention_mask
            self.logger.debug(f"Negative Input IDs length: {negative_input_ids.shape[1]}")
            with torch.no_grad():
                negative_output = self.text_encoder(negative_input_ids, attention_mask=negative_attention_mask)
            negative_embeds_pooled = negative_output.last_hidden_state.mean(dim=1)
            negative_embeds_pooled = torch.nan_to_num(negative_embeds_pooled, nan=0.0, posinf=1.0, neginf=-1.0)
            negative_embeds_pooled = negative_embeds_pooled.to(target_dtype)
            std_neg = negative_embeds_pooled.std(dim=1, keepdim=True)
            if (std_neg > 1e-6).all():
                negative_embeds_pooled = negative_embeds_pooled / (std_neg.clamp(min=1e-6) * 10.0)
            else:
                self.logger.warning("Zero or near-zero std in negative_embeds_pooled. Using random embeddings.")
                negative_embeds_pooled = torch.randn_like(negative_embeds_pooled) * 0.05
            negative_embeds_projected = self.projection_layer(negative_embeds_pooled)
            negative_embeds_projected = torch.nan_to_num(negative_embeds_projected, nan=0.0, posinf=1.0, neginf=-1.0)
            negative_embeds = negative_embeds_projected.unsqueeze(1).repeat(1, text_sequence_length, 1)
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
            text_mask = torch.cat([negative_attention_mask, attention_mask], dim=0)
        else:
            text_mask = attention_mask
        return prompt_embeds, None, text_mask

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        num_images_per_prompt=1,
        num_inference_steps=25,
        generator=None,
        guidance_scale=4.0,
        output_type="pt",
        **kwargs
    ):
        if isinstance(prompt, str): prompt = [prompt]
        if negative_prompt is not None and isinstance(negative_prompt, str): negative_prompt = [negative_prompt]
        batch_size = len(prompt)
        device = self.prior.device
        dtype = self.prior.dtype
        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt is not None
        prompt_embeds, _, text_mask = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt
        )
        prompt_embeds = torch.nan_to_num(prompt_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        effective_batch_size = prompt_embeds.shape[0]
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler_timesteps = self.scheduler.timesteps
        if generator is None: generator = torch.Generator(device=device)
        init_hidden_states = torch.randn(
            effective_batch_size, 1, self.prior_embed_dim,
            generator=generator, device=device, dtype=dtype
        )
        init_hidden_states = torch.nan_to_num(init_hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
        timestep_proj_dim = self.prior_embed_dim
        predicted_outputs = []
        for i, t in enumerate(scheduler_timesteps):
            t_input = t.unsqueeze(0) if t.ndim == 0 else t
            t_input = t_input.expand(effective_batch_size)
            timestep_emb = self.get_sinusoidal_embeddings(t_input, timestep_proj_dim)
            timestep_emb = timestep_emb.unsqueeze(1).to(dtype=dtype)
            timestep_emb = torch.nan_to_num(timestep_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            loop_hidden_states = init_hidden_states
            output = self.prior(
                hidden_states=loop_hidden_states,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                proj_embedding=timestep_emb,
                attention_mask=text_mask
            )
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
            predicted_outputs.append(output)
        if not predicted_outputs:
            raise ValueError("Inference loop did not produce any outputs.")
        predicted_image_embedding = predicted_outputs[-1]
        if predicted_image_embedding.shape[1] == 1:
            predicted_image_embedding = predicted_image_embedding.squeeze(1)
        predicted_image_embedding = torch.nan_to_num(predicted_image_embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        std = predicted_image_embedding.std(dim=1, keepdim=True)
        if (std > 1e-6).all():
            predicted_image_embedding = predicted_image_embedding / std.clamp(min=1e-6)
        self.logger.debug(f"Predicted image embedding stats - min: {predicted_image_embedding.min().item():.4f}, max: {predicted_image_embedding.max().item():.4f}, mean: {predicted_image_embedding.mean().item():.4f}")
        if do_classifier_free_guidance:
            image_embeds_uncond, image_embeds_text = predicted_image_embedding.chunk(2)
            image_embeds = image_embeds_uncond + guidance_scale * (image_embeds_text - image_embeds_uncond)
            negative_image_embeds = image_embeds_uncond
        else:
            image_embeds = predicted_image_embedding
            negative_image_embeds = None
        image_embeds = torch.clamp(image_embeds, -5.0, 5.0)
        self.logger.debug(f"Prior output stats - min: {image_embeds.min().item():.4f}, max: {image_embeds.max().item():.4f}, mean: {image_embeds.mean().item():.4f}")
        if output_type == "pt":
            output = image_embeds
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
        class PriorPipelineOutput:
            def __init__(self, image_embeds, negative_image_embeds=None):
                self.image_embeds = image_embeds
                self.negative_image_embeds = negative_image_embeds
        return PriorPipelineOutput(image_embeds=output, negative_image_embeds=negative_image_embeds)

    def get_sinusoidal_embeddings(self, timesteps, embedding_dim, max_period=10000):
        if timesteps.ndim == 0: timesteps = timesteps.unsqueeze(0)
        half = embedding_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
        return embedding

# --- FinetuneModel Class ---
class FinetuneModel:
    def __init__(self, model_name, output_dir, accelerator: Accelerator, logger_instance=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.logger = logger_instance or logging.getLogger(__name__)
        self.device = accelerator.device
        self.dtype = torch.float32
        self.tokenizer = None
        self.text_encoder = None
        self.unet = None
        self.scheduler = None
        self.vae = None
        self.prior_pipeline = None
        self.image_size = 512
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.current_epoch = 0
        self._apply_lora_text_flag = False
        self._apply_lora_unet_flag = False
        self.fold_val_losses = []

    def load_model(self):
        t5_model_name = "t5-base"
        prior_model_id = "kandinsky-community/kandinsky-2-2-prior"
        decoder_model_id = "kandinsky-community/kandinsky-2-2-decoder"
        try:
            cache_dir = "/home/iris/.cache/huggingface/hub/models--t5-base"
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
            self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, force_download=True)
            self.tokenizer.model_max_length = 512
            self.logger.debug(f"Tokenizer model_max_length set to: {self.tokenizer.model_max_length}")
            self.text_encoder = T5EncoderModel.from_pretrained(t5_model_name, force_download=True)
            self.text_encoder.to(self.device, dtype=self.dtype)
            self.text_encoder.gradient_checkpointing_enable()
            t5_weight_stats = {name: param.abs().mean().item() for name, param in self.text_encoder.named_parameters()}
            self.logger.debug(f"T5 encoder weight stats: {t5_weight_stats}")
            test_prompt = "a man standing in front of a grill"
            test_inputs = self.tokenizer([test_prompt], padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(self.device)
            self.logger.debug(f"T5 test input IDs: {test_inputs['input_ids'][0, :10].tolist()}")
            with torch.no_grad():
                test_output = self.text_encoder(**test_inputs)
            test_hidden_state = test_output.last_hidden_state
            self.logger.debug(f"T5 test encoder output - shape: {test_hidden_state.shape}, min: {test_hidden_state.min().item():.4f}, max: {test_hidden_state.max().item():.4f}, mean: {test_hidden_state.mean().item():.4f}")
            prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior").to(dtype=self.dtype)
            self.logger.warning("PriorTransformer does not support gradient checkpointing; skipping")
            prior_scheduler = DPMSolverMultistepScheduler.from_pretrained(prior_model_id, subfolder="scheduler")
            image_encoder = None
            image_processor = None
            self.prior_pipeline = KandinskyV22PriorPipelineWithT5(
                text_encoder=self.text_encoder, tokenizer=self.tokenizer, prior=prior,
                scheduler=prior_scheduler, image_encoder=image_encoder,
                image_processor=image_processor, logger_instance=self.logger
            )
            self.prior_pipeline.to(torch_device=self.device, torch_dtype=self.dtype)
            if hasattr(self.prior_pipeline, 'projection_layer'):
                self.prior_pipeline.projection_layer.to(device=self.device, dtype=self.dtype)
            self.vae = VQModel.from_pretrained(decoder_model_id, subfolder="movq").to(device=self.device, dtype=torch.float32)
            self.vae.eval()
            self.unet = UNet2DConditionModel.from_pretrained(decoder_model_id, subfolder="unet").to(device=self.device, dtype=self.dtype)
            if hasattr(self.unet, 'enable_gradient_checkpointing'):
                self.unet.enable_gradient_checkpointing()
            else:
                self.logger.warning("UNet does not support enable_gradient_checkpointing; skipping")
            self.scheduler = DPMSolverMultistepScheduler.from_pretrained(decoder_model_id, subfolder="scheduler")
        except Exception as e:
            self.logger.error(f"Failed to load model components: {e}\n{traceback.format_exc()}")
            raise

    def modify_architecture(self, apply_lora_to_unet=True, apply_lora_to_text_encoder=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        self.logger.debug(f"Applying LoRA to text encoder: {apply_lora_to_text_encoder}")
        self._apply_lora_text_flag = apply_lora_to_text_encoder
        self._apply_lora_unet_flag = apply_lora_to_unet
        lora_bias = "none"
        if apply_lora_to_unet and self.unet:
            unet_target_modules = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
            lora_config_unet = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=lora_bias, target_modules=unet_target_modules)
            try:
                self.unet = get_peft_model(self.unet, lora_config_unet)
                self.unet.print_trainable_parameters()
            except Exception as e:
                self.logger.error(f"Failed LoRA UNet: {e}")
        if apply_lora_to_text_encoder and hasattr(self.prior_pipeline, 'text_encoder') and hasattr(self.prior_pipeline, 'projection_layer'):
            text_target_modules = ["q", "k", "v", "o"]
            lora_config_text = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=lora_bias,
                target_modules=text_target_modules
            )
            try:
                self.prior_pipeline.text_encoder = get_peft_model(self.prior_pipeline.text_encoder, lora_config_text)
                self.prior_pipeline.text_encoder.print_trainable_parameters()
                for param in self.prior_pipeline.projection_layer.parameters():
                    param.requires_grad = True
                    param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
                proj_layer_trainable = any(p.requires_grad for p in self.prior_pipeline.projection_layer.parameters())
            except Exception as e:
                self.logger.error(f"Failed LoRA TextEncoder: {e}\n{traceback.format_exc()}")

    def validate(self, val_dataloader, dataset_path):
        if not all([self.unet, self.prior_pipeline, self.vae, self.scheduler]):
            self.logger.error("Missing required components for validation")
            return float('inf')
        self.unet.eval()
        self.prior_pipeline.text_encoder.eval()
        if hasattr(self.prior_pipeline, 'projection_layer'):
            self.prior_pipeline.projection_layer.eval()
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
                        self.logger.debug(f"Skipping val batch {step+1}: No valid images loaded")
                        continue
                    pixel_values = torch.stack(pixel_values_list).to(self.accelerator.device)
                    self.logger.debug(f"Val Step {step+1}: Image batch stats - min: {pixel_values.min().item():.4f}, max: {pixel_values.max().item():.4f}, mean: {pixel_values.mean().item():.4f}")
                    prompts = valid_prompts
                    prior_output = self.prior_pipeline(
                        prompt=prompts,
                        num_inference_steps=25,
                        generator=torch.Generator(device=self.accelerator.device).manual_seed(42+step)
                    )
                    image_embeds = prior_output.image_embeds.to(dtype=self.dtype)
                    if torch.isnan(image_embeds).any() or torch.isinf(image_embeds).any():
                        self.logger.warning(f"Val Step {step+1}: NaN/Inf in image_embeds")
                        continue
                    image_embeds = torch.nan_to_num(image_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
                    pixel_values_norm = pixel_values.to(dtype=torch.float32) * 2.0 - 1.0
                    vae_output = self.vae.encode(pixel_values_norm)
                    latents = vae_output.latents
                    scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)
                    latents = (latents * scaling_factor).to(dtype=self.dtype)
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        self.logger.warning(f"Val Step {step+1}: NaN/Inf in latents")
                        continue
                    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
                    noise = torch.randn_like(latents)
                    if torch.isnan(noise).any() or torch.isinf(noise).any():
                        self.logger.warning(f"Val Step {step+1}: NaN/Inf in noise")
                        continue
                    noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                    noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                    if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                        self.logger.warning(f"Val Step {step+1}: NaN/Inf in noisy_latents")
                        continue
                    noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0)
                    noisy_latents = noisy_latents / noisy_latents.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
                    image_embeds = image_embeds / image_embeds.std(dim=1, keepdim=True).clamp(min=1e-6)
                    added_cond_kwargs = {"image_embeds": image_embeds.to(dtype=self.dtype)}
                    model_pred = self.unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=None,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                    if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                        self.logger.warning(f"Val Step {step+1}: NaN/Inf in model_pred")
                        continue
                    model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1.0, neginf=-1.0)
                    model_pred = torch.clamp(model_pred, -1.0, 1.0)
                    if model_pred.shape[1] == 8 and noise.shape[1] == 4:
                        model_pred = model_pred[:, :4, :, :]
                    self.logger.debug(f"Val Step {step+1}: model_pred stats - min: {model_pred.min().item():.4f}, max: {model_pred.max().item():.4f}, mean: {model_pred.mean().item():.4f}")
                    self.logger.debug(f"Val Step {step+1}: noise stats - min: {noise.min().item():.4f}, max: {noise.max().item():.4f}, mean: {noise.mean().item():.4f}")
                    val_loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    self.logger.debug(f"Val Step {step+1}: Loss: {val_loss.item():.4f}")
                    if torch.isnan(val_loss) or torch.isinf(val_loss):
                        self.logger.warning(f"Val Step {step+1}: Calculated loss is NaN or Inf")
                        continue
                    gathered_loss = self.accelerator.gather(val_loss.repeat(len(batch['image'])))
                    total_val_loss += gathered_loss.mean().item()
                    num_val_batches += 1
                except Exception as val_step_err:
                    self.logger.error(f"Validation step {step+1} failed: {val_step_err}\n{traceback.format_exc()}")
                    continue
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        self.unet.train()
        self.prior_pipeline.text_encoder.train()
        if hasattr(self.prior_pipeline, 'projection_layer') and any(p.requires_grad for p in self.prior_pipeline.projection_layer.parameters()):
            self.prior_pipeline.projection_layer.train()
        self.logger.info(f"Validation completed - Avg validation loss: {avg_val_loss:.4f}, Batches: {num_val_batches}")
        return avg_val_loss

    def save_model_state(self, epoch=None, val_loss=None, hyperparameters=None, subdir=None):
        if not self.accelerator.is_main_process:
            return
        output_subdir = subdir if subdir is not None else self.output_dir
        try:
            os.makedirs(output_subdir, exist_ok=True)
            self.logger.debug(f"Created output directory: {output_subdir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory {output_subdir}: {e}")
            return
        save_label = "best"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None and not np.isnan(val_loss) else "N/A"
        save_paths = {}
        if self.unet and isinstance(self.unet, PeftModel):
            try:
                unet_path = os.path.join(output_subdir, f"{save_label}_unet_lora")
                self.accelerator.unwrap_model(self.unet).save_pretrained(unet_path)
                save_paths["UNet_LoRA"] = unet_path
                self.logger.info(f"Saved UNet LoRA to {unet_path}")
            except Exception as e:
                self.logger.error(f"Failed to save UNet LoRA to {unet_path}: {e}")
        if hasattr(self.prior_pipeline, 'text_encoder') and isinstance(self.prior_pipeline.text_encoder, PeftModel):
            try:
                text_encoder_path = os.path.join(output_subdir, f"{save_label}_text_encoder_lora")
                self.accelerator.unwrap_model(self.prior_pipeline.text_encoder).save_pretrained(text_encoder_path)
                save_paths["TextEncoder_LoRA"] = text_encoder_path
                self.logger.info(f"Saved TextEncoder LoRA to {text_encoder_path}")
            except Exception as e:
                self.logger.error(f"Failed to save TextEncoder LoRA to {text_encoder_path}: {e}")
        if hasattr(self.prior_pipeline, 'projection_layer'):
            is_proj_trainable = any(p.requires_grad for p in self.prior_pipeline.projection_layer.parameters())
            if is_proj_trainable:
                proj_layer_path = os.path.join(output_subdir, f"{save_label}_projection_layer.pth")
                try:
                    proj_state_dict = self.accelerator.get_state_dict(self.prior_pipeline.projection_layer)
                    torch.save(proj_state_dict, proj_layer_path)
                    save_paths["Projection_Layer"] = proj_layer_path
                    self.logger.info(f"Saved Projection Layer to {proj_layer_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save Projection Layer to {proj_layer_path}: {e}")
        if hyperparameters:
            hyperparameters['_apply_lora_text_encoder'] = self._apply_lora_text_flag
            hyperparameters['_apply_lora_unet'] = self._apply_lora_unet_flag
            hyperparam_path = os.path.join(output_subdir, f"{save_label}_hyperparameters.json")
            save_data = {'model_name': self.model_name, 'epoch': epoch, 'validation_loss': val_loss_str, 'hyperparameters': hyperparameters}
            try:
                with open(hyperparam_path, 'w') as f:
                    json.dump(save_data, f, indent=4)
                self.logger.info(f"Saved hyperparameters to {hyperparam_path}")
            except Exception as e:
                self.logger.error(f"Failed to save hyperparameters to {hyperparam_path}: {e}")
        if not save_paths:
            self.logger.warning(f"No trainable weights were saved for {self.model_name} ({save_label}).")
        else:
            self.logger.info(f"Saved model components: {save_paths}")

    def fine_tune(self, dataset_path, train_val_splits, epochs=1, batch_size=1, learning_rate=5e-8, gradient_accumulation_steps=8, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        self.logger.info(f"Requested learning rate: {learning_rate}")
        if learning_rate > 5e-8:
            self.logger.warning(f"High learning rate detected: {learning_rate}. Capping at 5e-8 for stability.")
            learning_rate = 5e-8
        self.fold_val_losses = []
        global_best_avg_val_loss = float('inf')
        global_best_hyperparameters = None
        global_best_epoch = -1
        for fold_idx, (train_dataset, val_dataset) in enumerate(train_val_splits):
            self.best_val_loss = float('inf')
            self.best_epoch = -1
            try:
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
                image_folder = os.path.join(os.path.dirname(dataset_path), "images")
                params_to_optimize = []
                if self._apply_lora_unet_flag and isinstance(self.unet, PeftModel):
                    params_to_optimize.extend(self.unet.parameters())
                if self._apply_lora_text_flag and hasattr(self.prior_pipeline, 'text_encoder') and isinstance(self.prior_pipeline.text_encoder, PeftModel):
                    params_to_optimize.extend(self.prior_pipeline.text_encoder.parameters())
                if hasattr(self.prior_pipeline, 'projection_layer'):
                    is_proj_trainable = any(p.requires_grad for p in self.prior_pipeline.projection_layer.parameters())
                    if is_proj_trainable:
                        text_encoder_param_ids = set(id(p) for p in self.prior_pipeline.text_encoder.parameters()) if isinstance(self.prior_pipeline.text_encoder, PeftModel) else set()
                        proj_params_to_add = [p for p in self.prior_pipeline.projection_layer.parameters() if id(p) not in text_encoder_param_ids]
                        if proj_params_to_add:
                            params_to_optimize.extend(proj_params_to_add)
                if not params_to_optimize:
                    self.logger.error("Optimizer params list is empty!")
                    continue
                params_to_optimize = list({id(p): p for p in params_to_optimize}.values())
                try:
                    optimizer = AdamW8bit(params_to_optimize, lr=learning_rate)
                except NameError:
                    self.logger.warning("bitsandbytes not found, using torch.optim.AdamW.")
                    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
                except Exception as e:
                    self.logger.warning(f"Optimizer setup failed: {e}, using AdamW.")
                    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)
                prepare_list = []
                models_prepared_names = []
                if self._apply_lora_unet_flag and isinstance(self.unet, PeftModel):
                    prepare_list.append(self.unet)
                    models_prepared_names.append("unet")
                prepare_list.extend([optimizer, train_dataloader, val_dataloader])
                prepared_components = self.accelerator.prepare(*prepare_list)
                component_iter = iter(prepared_components)
                if "unet" in models_prepared_names:
                    self.unet = next(component_iter)
                self.optimizer = next(component_iter)
                train_dataloader = next(component_iter)
                val_dataloader = next(component_iter)
                self.prior_pipeline.to(self.accelerator.device)
                self.vae.to(self.accelerator.device)
                self.vae.eval()
                self.accelerator.scaler._growth_factor = 1.1
                t5_model_name = getattr(self.prior_pipeline.text_encoder.config, "_name_or_path", "Unknown T5")
                hyperparameters = {
                    'model_name': self.model_name,
                    'text_encoder': t5_model_name,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'gradient_accumulation_steps': gradient_accumulation_steps,
                    'lora_r': lora_r,
                    'lora_alpha': lora_alpha,
                    'lora_dropout': lora_dropout,
                    'apply_lora_text_encoder': self._apply_lora_text_flag,
                    'apply_lora_unet': self._apply_lora_unet_flag
                }
                num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
                max_train_steps = epochs * num_update_steps_per_epoch
                global_step = 0
                for epoch in range(epochs):
                    self.current_epoch = epoch + 1
                    self.unet.train()
                    self.prior_pipeline.text_encoder.train()
                    if hasattr(self.prior_pipeline, 'projection_layer') and any(p.requires_grad for p in self.prior_pipeline.projection_layer.parameters()):
                        self.prior_pipeline.projection_layer.train()
                    train_loss_epoch = 0.0
                    num_train_batches_epoch = 0
                    num_skipped_steps = 0
                    for step, batch in enumerate(train_dataloader):
                        try:
                            image_filenames = batch.get('image')
                            prompts = batch.get('prompt')
                            batch_idx = step * batch_size
                            self.logger.debug(f"Fold {fold_idx+1}, Epoch {self.current_epoch}, Train Step {step+1} (Batch {batch_idx}): Prompts: {prompts}")
                            if not image_filenames or not prompts or \
                               not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
                                self.logger.warning(f"Fold {fold_idx+1}, Train Step {step+1}: Invalid prompts or image filenames")
                                continue
                            pixel_values_list = []
                            valid_prompts = []
                            for img_filename, prompt in zip(image_filenames, prompts):
                                try:
                                    image_path = os.path.join(image_folder, img_filename)
                                    image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
                                    image_np = np.array(image).astype(np.float32) / 255.0
                                    if np.any(np.isnan(image_np)) or np.any(np.isinf(image_np)):
                                        self.logger.warning(f"Fold {fold_idx+1}, Train Step {step+1}: NaN/Inf in image {img_filename}")
                                        continue
                                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
                                    pixel_values_list.append(image_tensor)
                                    valid_prompts.append(prompt)
                                except Exception as img_err:
                                    self.logger.debug(f"Fold {fold_idx+1}, Train Skip img {img_filename}: {img_err}")
                            if not pixel_values_list or not valid_prompts:
                                self.logger.debug(f"Fold {fold_idx+1}, Skipping train batch {step+1}: No valid images or prompts loaded.")
                                continue
                            pixel_values = torch.stack(pixel_values_list).to(self.accelerator.device)
                            self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Image batch stats - min: {pixel_values.min().item():.4f}, max: {pixel_values.max().item():.4f}, mean: {pixel_values.mean().item():.4f}")
                            prompts = valid_prompts
                            if step + 1 == 16:
                                self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Detailed debug mode activated")
                            if hasattr(self.prior_pipeline, 'text_encoder'):
                                self.prior_pipeline.text_encoder.eval()
                                self.logger.debug(f"Fold {fold_idx+1}, Set text_encoder to eval mode for prior pipeline call.")
                            with self.accelerator.accumulate(self.unet):
                                with torch.cuda.amp.autocast():
                                    prior_output = self.prior_pipeline(
                                        prompt=prompts,
                                        num_inference_steps=25,
                                        generator=torch.Generator(device=self.accelerator.device).manual_seed(42 + global_step)
                                    )
                                    image_embeds = prior_output.image_embeds.to(dtype=self.dtype)
                                    if torch.isnan(image_embeds).any() or torch.isinf(image_embeds).any():
                                        self.logger.warning(f"Fold {fold_idx+1}, Train Step {step+1}: NaN/Inf in image_embeds")
                                        continue
                                    image_embeds = torch.nan_to_num(image_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
                                    with torch.no_grad():
                                        pixel_values_f32 = pixel_values.to(dtype=torch.float32) * 2.0 - 1.0
                                        vae_output = self.vae.encode(pixel_values_f32)
                                        latents = vae_output.latents * getattr(self.vae.config, 'scaling_factor', 0.18215)
                                        latents = latents.to(dtype=self.dtype)
                                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                                        self.logger.warning(f"Fold {fold_idx+1}, Train Step {step+1}: NaN/Inf in latents")
                                        continue
                                    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
                                    noise = torch.randn_like(latents)
                                    if torch.isnan(noise).any() or torch.isinf(noise).any():
                                        self.logger.warning(f"Fold {fold_idx+1}, Train Step {step+1}: NaN/Inf in noise")
                                        continue
                                    noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=-1.0)
                                    timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                                    noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                                    if torch.isnan(noisy_latents).any() or torch.isinf(noisy_latents).any():
                                        self.logger.warning(f"Fold {fold_idx+1}, Train Step {step+1}: NaN/Inf in noisy_latents")
                                        continue
                                    noisy_latents = torch.nan_to_num(noisy_latents, nan=0.0, posinf=1.0, neginf=-1.0)
                                    noisy_latents = noisy_latents / noisy_latents.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
                                    image_embeds = image_embeds / image_embeds.std(dim=1, keepdim=True).clamp(min=1e-6)
                                    self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Noisy latents stats - min: {noisy_latents.min().item():.4f}, max: {noisy_latents.max().item():.4f}, mean: {noisy_latents.mean().item():.4f}")
                                    self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Image embeds stats - min: {image_embeds.min().item():.4f}, max: {image_embeds.max().item():.4f}, mean: {image_embeds.mean().item():.4f}")
                                    added_cond_kwargs = {"image_embeds": image_embeds.to(dtype=self.dtype)}
                                    model_pred = self.unet(
                                        sample=noisy_latents,
                                        timestep=timesteps,
                                        encoder_hidden_states=None,
                                        added_cond_kwargs=added_cond_kwargs
                                    ).sample
                                    if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
                                        self.logger.warning(f"Fold {fold_idx+1}, Train Step {step+1}: NaN/Inf in model_pred")
                                        continue
                                    model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1.0, neginf=-1.0)
                                    model_pred = torch.clamp(model_pred, -1.0, 1.0)
                                    if model_pred.shape[1] == 8 and noise.shape[1] == 4:
                                        model_pred = model_pred[:, :4, :, :]
                                    self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: model_pred stats - min: {model_pred.min().item():.4f}, max: {model_pred.max().item():.4f}, mean: {model_pred.mean().item():.4f}")
                                    self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: noise stats - min: {noise.min().item():.4f}, max: {noise.max().item():.4f}, mean: {noise.mean().item():.4f}")
                                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                                    self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Loss: {loss.item():.4f}")
                                    if torch.isnan(loss) or torch.isinf(loss):
                                        self.logger.warning(f"Fold {fold_idx+1}, Train Step {step+1}: Calculated loss is NaN or Inf")
                                        continue
                                self.accelerator.backward(loss / gradient_accumulation_steps * 50.0)
                                if self.accelerator.sync_gradients:
                                    if params_to_optimize:
                                        grad_norm = torch.tensor(float('inf'), device=self.device)
                                        valid_gradients = True
                                        try:
                                            raw_grad_norm = torch.norm(torch.stack([torch.norm(p.grad, 2) for p in params_to_optimize if p.grad is not None]), 2)
                                            self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Raw gradient norm: {raw_grad_norm.item():.4f}")
                                            grad_norm = self.accelerator.clip_grad_norm_(params_to_optimize, 0.5)
                                            self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Clipped gradient norm: {grad_norm.item():.4f}")
                                            scaler_state = self.accelerator.scaler.state_dict()
                                            self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Gradient scaler state - scale: {scaler_state['scale']}, growth_interval: {scaler_state['growth_interval']}")
                                        except ValueError as e:
                                            self.logger.warning(f"Fold {fold_idx+1}, Train Step {step+1}: Skipping optimizer step due to NaN/Inf gradients")
                                            valid_gradients = False
                                            self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Invalid gradients detected")
                                            scaler_state = self.accelerator.scaler.state_dict()
                                            self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Gradient scaler state on failure - scale: {scaler_state['scale']}, growth_interval: {scaler_state['growth_interval']}")
                                        if valid_gradients and not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                                            self.optimizer.step()
                                            self.optimizer.zero_grad()
                                        else:
                                            self.logger.debug(f"Fold {fold_idx+1}, Train Step {step+1}: Optimizer step skipped due to invalid gradients")
                                            num_skipped_steps += 1
                                train_loss_epoch += self.accelerator.gather(loss).mean().item() * gradient_accumulation_steps
                                num_train_batches_epoch += 1
                                if self.accelerator.is_main_process and global_step % 25 == 0:
                                    self.logger.info(f"Fold {fold_idx+1}, Epoch {self.current_epoch}, Step {global_step}/{max_train_steps}, Train Loss: {loss.item():.4f}, Skipped Steps: {num_skipped_steps}")
                                global_step += 1
                        except Exception as e:
                            self.logger.error(f"Fold {fold_idx+1}, Training step {step+1} failed: {e}\n{traceback.format_exc()}")
                            if "out of memory" in str(e).lower():
                                self.logger.error("OOM Error detected. Try smaller batch size or gradient accumulation.")
                            continue
                    avg_train_loss_epoch = train_loss_epoch / num_train_batches_epoch if num_train_batches_epoch > 0 else float('nan')
                    if self.accelerator.is_main_process:
                        self.logger.info(f"Fold {fold_idx+1}, Epoch {self.current_epoch} Finished - Avg Train Loss: {avg_train_loss_epoch:.4f}, Skipped Steps: {num_skipped_steps}/{num_train_batches_epoch}")
                    avg_val_loss = float('inf')
                    if val_dataloader:
                        if self.accelerator.is_main_process:
                            avg_val_loss = self.validate(val_dataloader, dataset_path)
                        self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        current_loss_for_comparison = avg_val_loss if val_dataloader else avg_train_loss_epoch
                        if not np.isnan(current_loss_for_comparison) and current_loss_for_comparison < self.best_val_loss:
                            self.best_val_loss = current_loss_for_comparison
                            self.best_epoch = self.current_epoch
                    self.accelerator.wait_for_everyone()
                    gc.collect()
                    torch.cuda.empty_cache()
                self.fold_val_losses.append(self.best_val_loss if not np.isnan(self.best_val_loss) else float('inf'))
                if self.best_val_loss < global_best_avg_val_loss:
                    global_best_avg_val_loss = self.best_val_loss
                    global_best_hyperparameters = hyperparameters
                    global_best_epoch = self.best_epoch
            except Exception as e:
                self.logger.error(f"Fold {fold_idx + 1} failed entirely: {e}\n{traceback.format_exc()}")
                self.fold_val_losses.append(float('inf'))
            self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            valid_losses = [loss for loss in self.fold_val_losses if loss != float('inf')]
            avg_val_loss = np.mean(valid_losses) if valid_losses else float('inf')
            self.logger.info(f"Training Finished - Avg validation loss: {avg_val_loss:.4f} across {len(valid_losses)} valid folds")
        model_state = {
            'unet': self.accelerator.get_state_dict(self.unet) if isinstance(self.unet, PeftModel) else None,
            'text_encoder': self.accelerator.get_state_dict(self.prior_pipeline.text_encoder) if hasattr(self.prior_pipeline, 'text_encoder') and isinstance(self.prior_pipeline.text_encoder, PeftModel) else None,
            'projection_layer': self.accelerator.get_state_dict(self.prior_pipeline.projection_layer) if hasattr(self.prior_pipeline, 'projection_layer') and any(p.requires_grad for p in self.prior_pipeline.projection_layer.parameters()) else None
        }
        return avg_val_loss, model_state, global_best_hyperparameters, global_best_epoch