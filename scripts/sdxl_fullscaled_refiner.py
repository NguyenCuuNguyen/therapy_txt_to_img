import os
import logging
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler, AutoencoderKL
import pandas as pd
import yaml
from PIL import Image
import numpy as np
import gc
import psutil
import torchvision
import transformers
import importlib.metadata
from accelerate import Accelerator
from transformers import T5Tokenizer, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from src.utils.openai_utils import OpenAIUtils  # Import OpenAIUtils for prompt refinement
from collections import OrderedDict  # For LRU cache implementation
from diffusers.utils import numpy_to_pil # Use standard numpy_to_pil

# Configure logging
log_dir = "/home/iris/Documents/deep_learning/src/logs"
log_file = os.path.join(log_dir, "sdxl_chunks_fullscaled_refiner_fixed.log") # New log file

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid conflicts
if logger.hasHandlers():
    logger.handlers.clear()

# Create a file handler
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Improved format
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Also log to console for easier debugging
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(file_formatter)
logger.addHandler(stream_handler)


# Log library versions for debugging
logger.info(f"torch version: {torch.__version__}")
logger.info(f"torchvision version: {torchvision.__version__}")
logger.info(f"transformers version: {transformers.__version__}")
logger.info(f"diffusers version: {importlib.metadata.version('diffusers')}")
try:
    logger.info(f"accelerate version: {importlib.metadata.version('accelerate')}")
except importlib.metadata.PackageNotFoundError:
    logger.info("accelerate not installed")
try:
    logger.info(f"peft version: {importlib.metadata.version('peft')}")
except importlib.metadata.PackageNotFoundError:
    logger.info("peft not installed")
try:
    import xformers
    logger.info(f"xformers version: {xformers.__version__}")
except ImportError:
    logger.info("xformers not installed")

# Set PyTorch memory allocation config to reduce fragmentation (optional)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"

def cleanup_memory():
    """Centralized function to clean up GPU and CPU memory."""
    logger.debug("Cleaning up memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    log_memory_usage()

def log_memory_usage(context=""):
    """Log both GPU and CPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GiB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GiB
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 # GiB
            free = total_memory - reserved
            logger.debug(f"{context} GPU Memory: Allocated {allocated:.2f} GiB, Reserved {reserved:.2f} GiB, Free {free:.2f} GiB (Total: {total_memory:.2f} GiB)")
        except Exception as e:
             logger.warning(f"Could not get GPU memory details: {e}")


    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_usage = mem_info.rss / 1024**3  # GiB
    logger.debug(f"{context} CPU Memory: RAM Usage {ram_usage:.2f} GiB")

def log_tensor_stats(tensor, name):
    """Log statistics of a tensor for debugging, including device."""
    if tensor is not None and isinstance(tensor, torch.Tensor):
        try:
            stats = (
                f"{name} stats: shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
                f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
                f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}"
            )
            logger.debug(stats)
        except Exception as e:
            logger.debug(f"Could not log stats for tensor {name}: {e}")
    elif tensor is None:
        logger.debug(f"{name} is None")
    else:
         logger.debug(f"{name} is not a tensor (type: {type(tensor)})")


def summarize_prompt(prompt, max_tokens=300):
    """Summarize a prompt to fit within a token limit while retaining key themes."""
    # Note: This is a basic word count summarizer. Token count might differ.
    words = prompt.split()
    if len(words) <= max_tokens:
        return prompt

    # Simple truncation for now, consider more sophisticated summarization if needed
    logger.warning(f"Prompt too long ({len(words)} words), truncating to {max_tokens} words.")
    return " ".join(words[:max_tokens])

# LRU Cache implementation to limit prompt_cache size
class LRUCache:
    def __init__(self, capacity=100):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Use base StableDiffusionXLPipeline and override encode_prompt
# No need for a full custom pipeline class if only overriding encode_prompt and __call__ logic
# However, the original code structure used a custom class, so we'll stick to that for minimal changes
# but simplify the __call__ method significantly.

class SDXLChunkingPipeline(StableDiffusionXLPipeline):
    # Keep __init__ largely the same but remove problematic overrides if any
    def __init__(self, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        self.max_chunks = 4  # Process up to ~300 tokens (4 * 75)
        self.chunk_size = 75 # Tokens per chunk (excluding BOS/EOS)
        # Gradient checkpointing can be enabled outside if needed (e.g., pipeline.enable_gradient_checkpointing())
        # self.unet.enable_gradient_checkpointing()
        # self.unet.eval() # Should be handled by pipeline context
        # self.vae.eval()

        logger.info("SDXLChunkingPipeline initialized. Using chunked prompt encoding.")
        # xFormers can be enabled via pipeline.enable_xformers_memory_efficient_attention() if installed and compatible
        # logger.info("xFormers memory-efficient attention is disabled by default in custom pipeline.")


    def _split_prompt(self, prompt: str):
        """Split a prompt into chunks for tokenizers."""
        # Use tokenizer to split correctly respecting token boundaries if possible
        # For simplicity, using word split first, then tokenize chunks
        # This might not be perfectly accurate wrt token limits per chunk but is simpler
        words = prompt.split()
        chunks = []
        current_chunk_words = []
        current_len = 0
        # Target 77 tokens per chunk (max_length for CLIP)
        # We use self.chunk_size=75 as a heuristic word limit per chunk
        for word in words:
            current_chunk_words.append(word)
            # Simple heuristic, might split mid-token group
            if len(current_chunk_words) >= self.chunk_size:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))

        return chunks[: self.max_chunks]


    def _encode_prompt_chunk(self, chunk, tokenizer, text_encoder, device, dtype):
        """Encode a single prompt chunk. Returns hidden states and pooled output."""
        with torch.no_grad():
            inputs = tokenizer(
                chunk,
                padding="max_length",
                max_length=tokenizer.model_max_length, # Should be 77
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(device)
            # logger.debug(f"Input IDs shape: {input_ids.shape}")

            # output_hidden_states=True is needed for SDXL's second-to-last layer output
            outputs = text_encoder(input_ids, output_hidden_states=True, return_dict=True)

            # SDXL uses hidden states from the penultimate layer
            # Ensure outputs.hidden_states is available and has enough layers
            if outputs.hidden_states is None or len(outputs.hidden_states) < 2:
                 raise ValueError("Text encoder did not return enough hidden states. Check model config.")
            prompt_embeds = outputs.hidden_states[-2] # Shape: (batch_size=1, seq_len=77, embed_dim)

            # Pooled output: text_encoder (CLIP) uses last_hidden_state[:, 0]
            # text_encoder_2 (CLIPWithProj) uses text_embeds (projection)
            if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                 pooled_prompt_embeds = outputs.text_embeds # Shape (batch_size=1, proj_dim)
            else:
                 # Fallback for standard CLIP text_encoder
                 pooled_prompt_embeds = outputs.last_hidden_state[:, 0, :] # Shape (batch_size=1, embed_dim)


        # logger.debug(f"Chunk embeddings shape: {prompt_embeds.shape}, Pooled shape: {pooled_prompt_embeds.shape}")
        # log_tensor_stats(prompt_embeds, "Chunk prompt_embeds")
        # log_tensor_stats(pooled_prompt_embeds, "Chunk pooled_embeds")
        return prompt_embeds.to(dtype=dtype), pooled_prompt_embeds.to(dtype=dtype)

    # Override encode_prompt completely
    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        lora_scale: float | None = None, # Added lora_scale
    ):
        """
        Encodes the prompt into text encoder hidden states and pooled embeddings. Handles chunking for long prompts.
        Based on diffusers StableDiffusionXLPipeline.encode_prompt but adds chunking.
        """
        dtype = self.text_encoder.dtype # Use encoder's dtype

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            # Assuming prompt_embeds are provided
            batch_size = prompt_embeds.shape[0]

        # logger.debug(f"encode_prompt - Batch size: {batch_size}, Num images per prompt: {num_images_per_prompt}")
        # logger.debug(f"Prompt type: {type(prompt)}")
        # if isinstance(prompt, str): logger.debug(f"Prompt: {prompt[:100]}...")
        # if isinstance(negative_prompt, str): logger.debug(f"Negative Prompt: {negative_prompt[:100]}...")

        # Tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]

        if prompt_embeds is None:
            prompt_text = prompt if isinstance(prompt, list) else [prompt] * batch_size
            prompt_embeds_list = []
            pooled_prompt_embeds_list = []

            for text in prompt_text:
                # Summarize prompt if too long (heuristic word count)
                summarized_text = summarize_prompt(text, max_tokens=self.max_chunks * self.chunk_size)

                prompt_chunks = self._split_prompt(summarized_text)
                # logger.debug(f"Split prompt into {len(prompt_chunks)} chunks.")

                chunk_embeds_list = []
                chunk_pooled_embeds_list = [] # Store pooled from text_encoder_2 for each chunk

                for chunk_idx, chunk in enumerate(prompt_chunks):
                    chunk_embeds_per_encoder = []
                    chunk_pooled_per_encoder = [] # Pooled from [encoder1, encoder2]

                    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                        embeds, pooled = self._encode_prompt_chunk(chunk, tokenizer, text_encoder, device, dtype)
                        chunk_embeds_per_encoder.append(embeds)
                        chunk_pooled_per_encoder.append(pooled)

                    # Concatenate hidden states from both encoders along the embedding dim
                    # Shape: (1, 77, 768 + 1280 = 2048)
                    chunk_concat_embeds = torch.cat(chunk_embeds_per_encoder, dim=-1)
                    chunk_embeds_list.append(chunk_concat_embeds)

                    # Store pooled output from text_encoder_2 (index 1) for this chunk
                    chunk_pooled_embeds_list.append(chunk_pooled_per_encoder[1])

                # Concatenate chunk hidden states along the sequence dimension
                # Shape: (1, num_chunks * 77, 2048)
                prompt_embeds = torch.cat(chunk_embeds_list, dim=1)

                # Use pooled output from the FIRST chunk of text_encoder_2
                # Shape: (1, 1280)
                pooled_prompt_embeds = chunk_pooled_embeds_list[0] if chunk_pooled_embeds_list else \
                    torch.zeros((1, self.text_encoder_2.config.hidden_size), dtype=dtype, device=device) # Fallback

                prompt_embeds_list.append(prompt_embeds)
                pooled_prompt_embeds_list.append(pooled_prompt_embeds)

            # Stack results from all prompts in the batch
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0) # Shape: (batch_size, num_chunks * 77, 2048)
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0) # Shape: (batch_size, 1280)
            # logger.debug(f"Final prompt_embeds shape: {prompt_embeds.shape}")
            # logger.debug(f"Final pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")

        # logger.debug(f"Prompt embeds dtype: {prompt_embeds.dtype}, Pooled embeds dtype: {pooled_prompt_embeds.dtype}")

        # Handle negative prompt encoding (similar chunking logic)
        if negative_prompt_embeds is None:
            if negative_prompt is None:
                negative_prompt = ""
            negative_prompt_text = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size

            uncond_embeds_list = []
            uncond_pooled_embeds_list = []

            for text in negative_prompt_text:
                # No need to summarize negative prompt usually, assume it's short
                # If needed: summarized_text = summarize_prompt(text, ...)
                # else:
                summarized_text = text

                # Allow empty negative prompt resulting in zero embeddings
                if not summarized_text:
                     # Use shapes from positive prompt, fill with zeros
                     num_chunks = prompt_embeds.shape[1] // tokenizers[0].model_max_length
                     seq_len = prompt_embeds.shape[1]
                     embed_dim = prompt_embeds.shape[2]
                     pooled_dim = pooled_prompt_embeds.shape[1]

                     negative_embeds = torch.zeros((1, seq_len, embed_dim), dtype=dtype, device=device)
                     negative_pooled = torch.zeros((1, pooled_dim), dtype=dtype, device=device)

                     uncond_embeds_list.append(negative_embeds)
                     uncond_pooled_embeds_list.append(negative_pooled)
                     continue # Skip encoding for empty string

                prompt_chunks = self._split_prompt(summarized_text)
                # logger.debug(f"Split negative prompt into {len(prompt_chunks)} chunks.")

                chunk_embeds_list = []
                chunk_pooled_embeds_list = []

                # Match the number of chunks used in the positive prompt for consistent sequence length?
                # No, SDXL handles variable lengths if padding/attention mask is correct. Let's keep it simple.

                for chunk_idx, chunk in enumerate(prompt_chunks):
                    chunk_embeds_per_encoder = []
                    chunk_pooled_per_encoder = []

                    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                        embeds, pooled = self._encode_prompt_chunk(chunk, tokenizer, text_encoder, device, dtype)
                        chunk_embeds_per_encoder.append(embeds)
                        chunk_pooled_per_encoder.append(pooled)

                    chunk_concat_embeds = torch.cat(chunk_embeds_per_encoder, dim=-1)
                    chunk_embeds_list.append(chunk_concat_embeds)
                    chunk_pooled_embeds_list.append(chunk_pooled_per_encoder[1])

                negative_embeds = torch.cat(chunk_embeds_list, dim=1)
                negative_pooled = chunk_pooled_embeds_list[0] if chunk_pooled_embeds_list else \
                    torch.zeros((1, self.text_encoder_2.config.hidden_size), dtype=dtype, device=device)

                uncond_embeds_list.append(negative_embeds)
                uncond_pooled_embeds_list.append(negative_pooled)

            negative_prompt_embeds = torch.cat(uncond_embeds_list, dim=0)
            negative_pooled_prompt_embeds = torch.cat(uncond_pooled_embeds_list, dim=0)
            # logger.debug(f"Final negative_prompt_embeds shape: {negative_prompt_embeds.shape}")
            # logger.debug(f"Final negative_pooled_prompt_embeds shape: {negative_pooled_prompt_embeds.shape}")

        # Pad shorter sequence (positive or negative) to match the longer one
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            max_seq_len = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])

            if prompt_embeds.shape[1] < max_seq_len:
                 padding_size = max_seq_len - prompt_embeds.shape[1]
                 padding = torch.zeros((prompt_embeds.shape[0], padding_size, prompt_embeds.shape[2]), dtype=dtype, device=device)
                 prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
                 logger.debug(f"Padded positive prompt embeds to shape: {prompt_embeds.shape}")


            if negative_prompt_embeds.shape[1] < max_seq_len:
                 padding_size = max_seq_len - negative_prompt_embeds.shape[1]
                 padding = torch.zeros((negative_prompt_embeds.shape[0], padding_size, negative_prompt_embeds.shape[2]), dtype=dtype, device=device)
                 negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)
                 logger.debug(f"Padded negative prompt embeds to shape: {negative_prompt_embeds.shape}")

        # Replicate embeds for multiple images per prompt
        if prompt_embeds is not None:
            bs_embed, seq_len, _ = prompt_embeds.shape
            if bs_embed != batch_size * num_images_per_prompt:
                 prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1) # Ensure correct view

        if pooled_prompt_embeds is not None:
             bs_embed_pooled, _ = pooled_prompt_embeds.shape
             if bs_embed_pooled != batch_size * num_images_per_prompt:
                  pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
             pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)


        if negative_prompt_embeds is not None:
             bs_embed_neg, seq_len_neg, _ = negative_prompt_embeds.shape
             if bs_embed_neg != batch_size * num_images_per_prompt:
                  negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
             negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len_neg, -1)

        if negative_pooled_prompt_embeds is not None:
             bs_embed_neg_pooled, _ = negative_pooled_prompt_embeds.shape
             if bs_embed_neg_pooled != batch_size * num_images_per_prompt:
                  negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
             negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        # logger.debug(f"Shape after replication: prompt_embeds {prompt_embeds.shape if prompt_embeds is not None else 'None'}")
        # logger.debug(f"Shape after replication: pooled_prompt_embeds {pooled_prompt_embeds.shape if pooled_prompt_embeds is not None else 'None'}")
        # logger.debug(f"Shape after replication: negative_prompt_embeds {negative_prompt_embeds.shape if negative_prompt_embeds is not None else 'None'}")
        # logger.debug(f"Shape after replication: negative_pooled_prompt_embeds {negative_pooled_prompt_embeds.shape if negative_pooled_prompt_embeds is not None else 'None'}")


        # For classifier-free guidance, concatenate the negative and positive embeddings.
        if do_classifier_free_guidance:
            if prompt_embeds is not None and negative_prompt_embeds is not None:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                log_tensor_stats(prompt_embeds, "Prompt embeds after CFG concat")
            if pooled_prompt_embeds is not None and negative_pooled_prompt_embeds is not None:
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                log_tensor_stats(pooled_prompt_embeds, "Pooled embeds after CFG concat")

        return prompt_embeds, pooled_prompt_embeds # Return both, __call__ will construct add_text_embeds


    # Simplified __call__ method, relying on Accelerator and base class logic where possible
    @torch.no_grad() # Ensure no gradients are computed in inference
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0, # Default SDXL guidance scale
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback: callable | None = None,
        callback_steps: int = 1,
        cross_attention_kwargs: dict[str, any] | None = None,
        guidance_rescale: float = 0.0,
        original_size: tuple[int, int] | None = None,
        crops_coords_top_left: tuple[int, int] = (0, 0),
        target_size: tuple[int, int] | None = None,
        negative_original_size: tuple[int, int] | None = None,
        negative_crops_coords_top_left: tuple[int, int] = (0, 0),
        negative_target_size: tuple[int, int] | None = None,
        aesthetic_score: float = 6.0, # SDXL specific param
        negative_aesthetic_score: float = 2.5, # SDXL specific param
        denoising_end: float | None = None, # For refiner split
        refiner: StableDiffusionXLImg2ImgPipeline | None = None, # Pass refiner instance
        **kwargs, # Catch any unexpected kwargs
    ):
        logger.info("Starting SDXLChunkingPipeline __call__")
        log_memory_usage("Start of __call__")

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds,
            pooled_prompt_embeds, negative_pooled_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0] // (2 if self.do_classifier_free_guidance else 1) # Infer from embeds

        device = self.device # Relies on Accelerator placing the pipeline correctly
        dtype = self.unet.dtype # Use the dtype of the main model component

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt using the overridden chunking method
        logger.info("Encoding prompt...")
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            # lora_scale=lora_scale, # Pass lora_scale if used
        )
        log_tensor_stats(prompt_embeds, "Encoded prompt_embeds")
        log_tensor_stats(pooled_prompt_embeds, "Encoded pooled_prompt_embeds")
        log_memory_usage("After prompt encoding")
        cleanup_memory() # Cleanup after encoding if needed

        # 4. Prepare added conditions for SDXL
        # Note: aesthetic scores are handled within add_time_ids in recent diffusers versions
        # Need to use the base class's method correctly
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
             original_size, crops_coords_top_left, target_size, aesthetic_score,
             negative_original_size, negative_crops_coords_top_left, negative_target_size, negative_aesthetic_score,
             batch_size * num_images_per_prompt, device, dtype
         )

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        add_time_ids = add_time_ids.to(device, dtype=prompt_embeds.dtype) # Ensure correct dtype and device
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        log_tensor_stats(add_text_embeds, "add_text_embeds")
        log_tensor_stats(add_time_ids, "add_time_ids")

        # 5. Prepare timesteps
        num_inference_steps_base = num_inference_steps
        timesteps_base = self.scheduler.timesteps # Will be set outside loop

        # If using refiner, adjust base steps
        refiner_start_step = num_inference_steps
        if refiner is not None and denoising_end is not None:
            num_inference_steps_base = int(num_inference_steps * denoising_end)
            refiner_start_step = num_inference_steps_base
            logger.info(f"Using refiner: Base steps = {num_inference_steps_base}, Refiner steps = {num_inference_steps - num_inference_steps_base}")

        self.scheduler.set_timesteps(num_inference_steps_base, device=device)
        timesteps_base = self.scheduler.timesteps
        logger.debug(f"Base timesteps: {timesteps_base}")


        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype, # Use embed dtype
            device,
            generator,
            latents,
        )
        log_tensor_stats(latents, "Initial latents")

        # 7. Prepare extra step kwargs. TODO: Logic should ideally be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop (Base)
        logger.info("Starting base denoising loop...")
        num_warmup_steps = len(timesteps_base) - num_inference_steps_base * self.scheduler.order
        with self.progress_bar(total=num_inference_steps_base) as progress_bar:
            for i, t in enumerate(timesteps_base):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # log_tensor_stats(latent_model_input, f"Step {i} latent_model_input")

                # predict the noise residual
                # Ensure components are on the correct device (Accelerator should handle)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0] # Get sample output
                # log_tensor_stats(noise_pred, f"Step {i} noise_pred")

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    # Optional guidance rescaling
                    if guidance_rescale > 0.0:
                         noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)


                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                # log_tensor_stats(latents, f"Step {i} updated latents")

                # call the callback, if provided
                if i == len(timesteps_base) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Store latents before potential refiner stage
        image_latents = latents
        log_memory_usage("After base loop")
        cleanup_memory() # Cleanup after base loop

        # 9. Refiner Denoising loop
        if refiner is not None and denoising_end is not None:
             logger.info("Starting refiner denoising loop...")
             # Ensure refiner is on the correct device (Accelerator should handle)
             refiner.scheduler.set_timesteps(num_inference_steps - refiner_start_step, device=device)
             timesteps_refiner = refiner.scheduler.timesteps
             logger.debug(f"Refiner timesteps: {timesteps_refiner}")

             # Use base prompt_embeds but potentially only the part corresponding to text_encoder_2?
             # Refiner typically uses only text_encoder_2 embeddings (1280 dim)
             # However, the Img2Img refiner pipeline might handle this internally or expect both.
             # Let's pass the *full* base prompt_embeds and added_cond_kwargs,
             # assuming the refiner pipeline's UNet handles selection or projection.
             # If errors occur, check refiner's expected input dims. The Img2Img refiner
             # does use the same `added_cond_kwargs` structure.

             # Refiner needs image latents as input, not noise latents
             # The loop above modifies `latents` in place, so `image_latents` has the result.

             # The refiner is StableDiffusionXLImg2ImgPipeline, needs `image` argument (latents)
             # We skip the initial steps of Img2Img (add_noise) and jump into its denoising loop
             # This requires careful state management OR just using the refiner's __call__
             # Reusing the loop structure like the original code for consistency:

             latents = image_latents # Start from base result
             num_warmup_steps = len(timesteps_refiner) - (num_inference_steps - refiner_start_step) * refiner.scheduler.order
             with self.progress_bar(total=num_inference_steps - refiner_start_step) as progress_bar:
                 for i, t in enumerate(timesteps_refiner):
                      latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                      latent_model_input = refiner.scheduler.scale_model_input(latent_model_input, t)

                      # Refiner UNet expects encoder_hidden_states dim 1280
                      # Let's slice the prompt_embeds from the base pipeline
                      # Shape: (2*bs, seq_len, 2048) -> (2*bs, seq_len, 1280)
                      refiner_prompt_embeds = prompt_embeds[..., -refiner.unet.config.cross_attention_dim :]
                      # log_tensor_stats(refiner_prompt_embeds, f"Refiner Step {i} prompt_embeds")

                      noise_pred = refiner.unet(
                          latent_model_input,
                          t,
                          encoder_hidden_states=refiner_prompt_embeds, # Use sliced embeds
                          cross_attention_kwargs=cross_attention_kwargs,
                          added_cond_kwargs=added_cond_kwargs, # Use same added conditions
                          return_dict=False,
                      )[0]
                      # log_tensor_stats(noise_pred, f"Refiner Step {i} noise_pred")

                      if do_classifier_free_guidance:
                          noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                          noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                          # Optional guidance rescaling
                          if guidance_rescale > 0.0:
                              noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)


                      latents = refiner.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                      # log_tensor_stats(latents, f"Refiner Step {i} updated latents")


                      if i == len(timesteps_refiner) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % refiner.scheduler.order == 0):
                          progress_bar.update()
                          # Add callback support if needed for refiner stage

             image_latents = latents # Update latents with refined result
             log_memory_usage("After refiner loop")
             cleanup_memory() # Cleanup after refiner loop


        # 10. Post-processing VAE decode
        logger.info("Decoding latents...")
        # Ensure VAE is on the correct device (Accelerator should handle)
        if output_type == "latent":
             image = image_latents
        else:
             # Handle VAE dtype for decode stability
             needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
             if needs_upcasting:
                  logger.debug("Upcasting VAE for decoding...")
                  self.upcast_vae() # Utility function from base class

             # Scale latents before decoding
             image_latents = image_latents / self.vae.config.scaling_factor
             log_tensor_stats(image_latents, "Latents before VAE decode")

             image = self.vae.decode(image_latents.to(self.vae.dtype), return_dict=False)[0] # Decode using VAE dtype
             log_tensor_stats(image, "VAE decoded image")

             if needs_upcasting:
                  logger.debug("Returning VAE to original dtype...")
                  self.vae.to(dtype=dtype) # Return VAE to original pipeline dtype

        # 11. Convert to PIL
        if output_type == "pil":
             image = self.image_processor.postprocess(image, output_type=output_type)
             # image = numpy_to_pil(image) # Use standard utility
        elif output_type == "numpy":
             image = self.image_processor.postprocess(image, output_type="np") # Use numpy for array output

        # Offload last model step to CPU? Handled by Accelerator if offloading enabled.
        log_memory_usage("After VAE decode")
        cleanup_memory()

        if not return_dict:
            return (image,)

        # Return using base class standard output format
        return StableDiffusionXLPipelineOutput(images=image) # Adjust if refiner changes output


    # Helper to get add_time_ids using base class logic if possible
    # Override of _get_add_time_ids might not be needed if using base class call properly
    # This depends on exact diffusers version features
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score, # Added aesthetic score
        negative_original_size, # Added negative versions
        negative_crops_coords_top_left,
        negative_target_size,
        negative_aesthetic_score,
        batch_size,
        device,
        dtype
    ):
         # In recent diffusers versions, this logic is often inside `prepare_extra_step_kwargs`
         # or directly handled in `__call__`. We mimic the structure needed by the UNet.
         # Standard implementation:
         add_time_ids = list(original_size + crops_coords_top_left + target_size)
         add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
         add_time_ids = add_time_ids.repeat(batch_size, 1)
         # Newer versions might concatenate aesthetic scores here too.
         # For simplicity, returning the basic 6-dim vector. UNet should handle it.
         # Check the UNet's expected `added_cond_kwargs` if errors persist.
         # Simpler approach: Use the base class implementation directly if possible
         try:
              # Attempt to call the base class method if it exists and handles new args
              base_ids = super()._get_add_time_ids(original_size, crops_coords_top_left, target_size,
                                                   aesthetic_score, negative_original_size, negative_crops_coords_top_left,
                                                   negative_target_size, negative_aesthetic_score, dtype)
              return base_ids.repeat(batch_size, 1).to(device=device) # Ensure device and repeat
         except TypeError:
              # Fallback to basic implementation if base class doesn't match
              logger.warning("Falling back to basic _get_add_time_ids implementation.")
              get_condition_embeddings = getattr(self, "_get_condition_embeddings", None) # Diffusers >= 0.26?
              if get_condition_embeddings:
                  add_time_ids = get_condition_embeddings(
                      original_size, crops_coords_top_left, target_size, aesthetic_score,
                      negative_original_size, negative_crops_coords_top_left, negative_target_size, negative_aesthetic_score,
                      batch_size, device, dtype
                  )
                  return add_time_ids # Already repeated and on device
              else: # Older diffusers fallback
                  add_time_ids = list(original_size + crops_coords_top_left + target_size)
                  add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
                  add_time_ids = add_time_ids.repeat(batch_size, 1)
                  return add_time_ids



def load_theory_prompts(yaml_path):
    """Load theory prompts from YAML file."""
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        prompts = config.get("theory_exp", {}).get("prompts", [])
        theory_prompts = {list(item.keys())[0]: list(item.values())[0] for item in prompts}
        logger.info(f"Loaded {len(theory_prompts)} theory prompts from {yaml_path}")
        return theory_prompts
    except Exception as e:
        logger.error(f"Failed to load YAML from {yaml_path}: {e}")
        raise

def load_id_list(id_list_path):
    """Load list of IDs from a text file."""
    try:
        with open(id_list_path, "r") as f:
            id_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(id_list)} IDs from {id_list_path}")
        return id_list
    except Exception as e:
        logger.error(f"Failed to load ID list from {id_list_path}: {e}")
        raise

def load_csv_rows_by_ids(csv_path, id_list, chunk_size=50):
    """Load CSV rows in chunks where 'file' matches values in id_list, excluding 'file' column."""
    id_list_set = set(str(id_val) for id_val in id_list) # Use set for faster lookup
    prompts = []
    total_rows_processed = 0
    logger.info(f"Loading CSV rows from {csv_path} for {len(id_list_set)} IDs...")
    try:
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)): # Added low_memory=False
            total_rows_processed += len(chunk)
            if "file" not in chunk.columns:
                raise ValueError("CSV does not contain a 'file' column")

            # Ensure 'file' column is string for comparison
            chunk['file'] = chunk['file'].astype(str)

            selected_rows = chunk[chunk["file"].isin(id_list_set)]
            if selected_rows.empty:
                del chunk
                # cleanup_memory() # Less frequent cleanup
                continue

            logger.info(f"Found {len(selected_rows)} matching IDs in chunk {i+1}")
            for _, row in selected_rows.iterrows():
                row_dict = row.to_dict()
                file_id = row_dict.get("file", "UNKNOWN_ID") # Get file ID safely
                prompt_parts = [
                    f"{key}: {value}"
                    for key, value in row_dict.items()
                    if key not in ["file"] and pd.notnull(value) and value != "" # Exclude empty values
                ]
                txt_prompt = "; ".join(prompt_parts)
                if not txt_prompt: # Handle cases where row only has ID
                    logger.warning(f"Row ID {file_id} has no descriptive data, skipping prompt generation.")
                    continue
                prompts.append((file_id, txt_prompt))

            del chunk, selected_rows # Explicitly delete to maybe help GC
            # cleanup_memory() # Less frequent cleanup

    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
        raise

    logger.info(f"Finished reading CSV. Processed {total_rows_processed} rows, found {len(prompts)} matching prompts.")
    if not prompts and id_list: # Check if IDs were provided but none found
        logger.warning(f"No rows found in CSV matching the provided IDs.")
        # raise ValueError(f"No rows found in CSV with IDs: {id_list}") # Don't raise error, allow running with no matches

    cleanup_memory()
    return prompts


def refine_prompt_with_openai(openai_util, prompt, tokenizer, target_max_tokens=300):
    """Refine the prompt using OpenAI API to add visual cues while keeping token count within limit."""
    system_prompt = (
        "You are an expert prompt engineer for image generation from a psychotherapy theory perspective. Given a psychotherapy theory and relevant information about a person, Your task is to refine the provided prompt by adding "
        "photorealistic descriptive visual cues (e.g., 'a woman sitting and thinking deeply about family, with images of school, children, and old parents in the background. In front of her are opened notebooks with F grades. Add a volcano in the background for anger.') "
        "to make the image more illustrative of the person's life, events, emotions, meaningful and visually coherent, while preserving the core themes and instructions. "
        "Focus on concrete visual elements. Ensure the output is ONLY the refined prompt, with no introductory text, explanations, or formatting like quotes or markdown."
    )
    user_prompt = (
        f"Refine the following prompt by adding photorealistic visual cues (e.g., a man feeding his dog with 2 children playing in the background) to enhance its visual coherence for image generation, "
        f"while keeping the core themes and instructions intact. Target around {target_max_tokens} tokens. Keep it concise and focused on visual details. "
        f"Prompt:\n\n{prompt}" # Removed quotes around prompt
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        logger.debug("Calling OpenAI API for prompt refinement...")
        response = openai_util.client.chat.completions.create(
            model="gpt-4o-mini", # Or another suitable model like gpt-3.5-turbo
            messages=messages,
            temperature=0.4, # Slightly increased temperature for creativity
            max_tokens=target_max_tokens + 100, # Generous buffer
            # top_p=0.9, # Consider using top_p sampling
            n=1, # Generate one completion
            stop=None # No specific stop sequences
        )
        refined_prompt = response.choices[0].message.content.strip()
        # Remove potential markdown or quotes added by the model
        refined_prompt = refined_prompt.replace("```", "").replace("\"", "").strip()

        # Use a placeholder tokenizer for word count estimate if T5 not available/needed
        # token_count = len(refined_prompt.split()) # Rough word count
        # logger.debug(f"Refined prompt word count: {token_count}")
        # if token_count > target_max_tokens:
        #     logger.warning(f"OpenAI refined prompt potentially too long ({token_count} words), summarizing...")
        #     refined_prompt = summarize_prompt(refined_prompt, max_tokens=target_max_tokens) # Use word count summarizer

        logger.debug(f"Refined prompt (first 100 chars): {refined_prompt[:100]}...")
        return refined_prompt
    except Exception as e:
        logger.error(f"Failed to refine prompt with OpenAI: {e}")
        # Fallback: add a simple visual cue
        logger.warning("Using fallback prompt.")
        # Ensure fallback is visually descriptive
        return f"Photorealistic scene illustrating the following themes: {prompt}. Visually detailed, focusing on concrete elements and emotions."


def generate_images_for_csv_rows(
    csv_path,
    yaml_path,
    id_list_path,
    output_dir="/home/iris/Documents/deep_learning/generated_images/sdxl_chunks_fullscaled_refiner_fixed", # New output dir
    chunk_size=1, # Process one row -> one theory at a time for simplicity
):
    """Generate abstract images for CSV rows with matching IDs using theory prompts with refiner."""
    # Initialize Accelerator
    # Let Accelerator handle device placement and mixed precision
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",  # Use FP16 on GPU if available
        # cpu=not torch.cuda.is_available(), # Let accelerator choose CPU if no CUDA
        log_with="tensorboard", # Optional: configure logging
        project_dir=os.path.join(log_dir, "accelerate_logs") # Optional
    )
    device = accelerator.device
    dtype = torch.float16 if accelerator.native_amp else torch.float32
    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Load models BEFORE prepare, specifying torch_dtype
    # Use low_cpu_mem_usage if loading large models on system with limited RAM
    logger.info("Loading SDXL base pipeline components...")
    try:
        # Load components individually for more control if needed, or load full pipeline
        base_pipeline_pretrained = StableDiffusionXLPipeline.from_pretrained(
             "stabilityai/stable-diffusion-xl-base-1.0",
             torch_dtype=dtype,
             use_safetensors=True,
             variant="fp16" if dtype == torch.float16 else None, # Load fp16 weights directly if using fp16
             # low_cpu_mem_usage=True # Use if RAM is limited
        )
        logger.info("Base pipeline loaded.")
    except Exception as e:
        logger.error(f"Failed to load base pipeline: {e}")
        raise

    logger.info("Loading SDXL refiner pipeline components...")
    try:
        refiner_pipeline_pretrained = StableDiffusionXLImg2ImgPipeline.from_pretrained(
             "stabilityai/stable-diffusion-xl-refiner-1.0",
             torch_dtype=dtype,
             use_safetensors=True,
             variant="fp16" if dtype == torch.float16 else None,
             # Share VAE and text_encoder_2 for efficiency? Refiner usually has its own TE2. VAE sharing is good.
             vae=base_pipeline_pretrained.vae, # Share the VAE instance
             # text_encoder_2=base_pipeline_pretrained.text_encoder_2, # No, refiner uses its own TE2
             # low_cpu_mem_usage=True
        )
        logger.info("Refiner pipeline loaded.")
    except Exception as e:
        logger.error(f"Failed to load refiner pipeline: {e}")
        raise

    # Create custom pipeline instance using components from the loaded base pipeline
    # Ensure a fresh scheduler instance is created from config
    custom_pipeline = SDXLChunkingPipeline(
        vae=base_pipeline_pretrained.vae,
        text_encoder=base_pipeline_pretrained.text_encoder,
        text_encoder_2=base_pipeline_pretrained.text_encoder_2,
        tokenizer=base_pipeline_pretrained.tokenizer,
        tokenizer_2=base_pipeline_pretrained.tokenizer_2,
        unet=base_pipeline_pretrained.unet,
        scheduler=DPMSolverMultistepScheduler.from_config(base_pipeline_pretrained.scheduler.config),
    )

    # Set scheduler for the refiner instance too
    refiner_pipeline_pretrained.scheduler = DPMSolverMultistepScheduler.from_config(refiner_pipeline_pretrained.scheduler.config)

    # Optional: Enable memory optimizations before prepare if needed
    # custom_pipeline.enable_model_cpu_offload() # If using Accelerator's offloading (requires >= 0.17.0)
    # refiner_pipeline_pretrained.enable_model_cpu_offload()
    # custom_pipeline.unet.enable_gradient_checkpointing() # Can save memory during training, less impact on inference
    # refiner_pipeline_pretrained.unet.enable_gradient_checkpointing()

    # Prepare the pipelines with Accelerator LAST
    # Accelerator handles moving models to the correct device(s) defined during init
    logger.info("Preparing pipelines with Accelerator...")
    custom_pipeline, refiner_pipeline = accelerator.prepare(custom_pipeline, refiner_pipeline_pretrained)
    logger.info("Pipelines prepared and placed on device.")
    log_memory_usage("After prepare")

    # Now use `custom_pipeline` and `refiner_pipeline` directly

    logger.info("Initializing OpenAIUtils for prompt refinement...")
    try:
        # Prioritize environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # Fallback to config file
            config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
            if os.path.exists(config_path):
                 config = yaml.safe_load(open(config_path, "r"))
                 api_key = config.get("openai", {}).get("api_key")
            else:
                 logger.warning(f"Config file not found at {config_path}")

        if not api_key:
            raise ValueError("OpenAI API key not found in environment variable 'OPENAI_API_KEY' or in config file.")

        openai_util = OpenAIUtils(api_key=api_key)
        # Load tokenizer if needed for accurate token counting, otherwise skip for performance
        # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        tokenizer = None # Using word count heuristic for now
        logger.info("OpenAIUtils initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAIUtils: {e}")
        raise

    theory_prompts = load_theory_prompts(yaml_path)
    id_list = load_id_list(id_list_path)
    if not id_list:
         logger.warning("ID list is empty. No images will be generated.")
         return

    row_prompts = load_csv_rows_by_ids(csv_path, id_list, chunk_size=100) # Larger CSV read chunk
    if not row_prompts:
         logger.warning("No matching rows found in CSV for the given IDs. No images will be generated.")
         return

    os.makedirs(output_dir, exist_ok=True)

    negative_prompt = (
        "text, words, letters, abstract concept, low quality, blurry, noisy, distorted, watermark, signature, username, writing, labels" # Enhanced negative
    )

    prompt_cache = LRUCache(capacity=50) # Cache for refined prompts
    total_images_generated = 0
    total_images_failed = 0

    # Process one row at a time, generate images for all theories for that row
    logger.info(f"Starting image generation for {len(row_prompts)} CSV rows...")
    for row_idx, (row_id, txt_prompt) in enumerate(row_prompts):
        logger.info(f"Processing row {row_idx + 1}/{len(row_prompts)} - ID: {row_id}")
        safe_id = str(row_id).replace("/", "_").replace("\\", "_") # Sanitize ID for filenames

        for theory, prompt_template in theory_prompts.items():
            logger.info(f"  Generating image for theory: {theory}")
            try:
                # Construct base prompt
                base_prompt = prompt_template.format(txt_prompt=txt_prompt)
                # Add visual style cues
                prompt = f"Photorealistic illustration representing {theory} themes in this person's life: {base_prompt}. Use vibrant colors and expressive patterns, detailed scene."

                # Refine prompt using cache or OpenAI
                refined_prompt = prompt_cache.get(prompt)
                if refined_prompt is None:
                    logger.debug("Refining prompt with OpenAI...")
                    refined_prompt = refine_prompt_with_openai(openai_util, prompt, tokenizer, target_max_tokens=250) # Slightly shorter target
                    prompt_cache.put(prompt, refined_prompt)
                    cleanup_memory() # Clean up after potential large API response
                else:
                    logger.debug("Using cached refined prompt.")

                if row_idx == 0: # Log first refined prompt fully once
                    logger.info(f"  Refined Prompt (Theory: {theory}): {refined_prompt}")
                else: # Log shorter version subsequently
                    logger.debug(f"  Refined Prompt (Theory: {theory}): {refined_prompt[:100]}...")

                # Generate image using the prepared pipeline
                # Use accelerator context for potential distributed training features (though not used here)
                with accelerator.accumulate(custom_pipeline): # Context manager might not be needed for inference
                     images_output = custom_pipeline(
                         prompt=refined_prompt,
                         height=1024, # Standard SDXL size
                         width=1024,
                         num_inference_steps=40, # Slightly fewer steps for speed
                         guidance_scale=7.0, # Standard SDXL guidance
                         negative_prompt=negative_prompt,
                         num_images_per_prompt=1,
                         denoising_end=0.8, # Proportion of steps for base model
                         refiner=refiner_pipeline, # Pass the prepared refiner pipeline
                         output_type="pil", # Get PIL images directly
                         return_dict=True # Get output object
                     )
                     images = images_output.images # Extract images from output object

                # Save image
                output_path = os.path.join(output_dir, safe_id, f"{safe_id}_{theory}.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                images[0].save(output_path)
                logger.info(f"  Saved image to {output_path}")
                total_images_generated += 1

                # Manual cleanup (important!)
                del images, images_output, refined_prompt
                cleanup_memory() # Clean up between theories

            except Exception as e:
                logger.error(f"  Failed to generate image for ID {row_id}, theory {theory}: {e}", exc_info=True) # Log traceback
                total_images_failed += 1
                # Don't stop execution, continue with next theory/row
                # Ensure memory is cleaned up after failure
                cleanup_memory()
            finally:
                 # Ensure memory is cleaned up even if there was no exception in the try block
                 cleanup_memory()
                 log_memory_usage(f"End of theory {theory}")

        # Optional: Clear cache per row if memory is extremely tight
        # prompt_cache.cache.clear()
        cleanup_memory() # Clean up between rows
        logger.info(f"Finished processing row ID: {row_id}. Generated: {total_images_generated}, Failed: {total_images_failed}")


    logger.info(f"Image generation complete. Total Generated: {total_images_generated}, Total Failed: {total_images_failed}")

if __name__ == "__main__":
    csv_path = "/home/iris/Documents/deep_learning/data/input_csv/FILE_SUPERTOPIC_DESCRIPTION.csv"
    yaml_path = "/home/iris/Documents/deep_learning/config/prompt_config.yaml"
    id_list_path = "/home/iris/Documents/deep_learning/data/sample_list.txt" # Ensure this list contains IDs present in the CSV

    # Add basic check for file existence
    if not os.path.exists(csv_path):
         logger.error(f"CSV file not found: {csv_path}")
    elif not os.path.exists(yaml_path):
         logger.error(f"YAML config file not found: {yaml_path}")
    elif not os.path.exists(id_list_path):
         logger.error(f"ID list file not found: {id_list_path}")
    else:
         try:
              generate_images_for_csv_rows(csv_path, yaml_path, id_list_path)
         except Exception as main_e:
              logger.critical(f"An unhandled error occurred in the main execution: {main_e}", exc_info=True)