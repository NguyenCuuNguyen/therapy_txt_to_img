import os
import logging
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import pandas as pd
import yaml
from PIL import Image
import numpy as np
import gc
import psutil
import importlib.metadata
import transformers
from accelerate import Accelerator
from transformers import T5Tokenizer
from src.utils.openai_utils import OpenAIUtils  # Import OpenAIUtils for prompt refinement
from collections import OrderedDict  # For LRU cache implementation

# Configure logging
log_dir = "/home/iris/Documents/deep_learning/src/logs"
log_file = os.path.join(log_dir, "sd15_chunks_noRefiner.log")

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Remove any existing handlers to avoid conflicts
logger.handlers = []

# Create a file handler
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Create a stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_formatter = logging.Formatter('%(asctime)s %(message)s')
stream_handler.setFormatter(stream_formatter)
logger.addHandler(stream_handler)

# Log library versions for debugging
logger.info(f"torch version: {torch.__version__}")
logger.info(f"transformers version: {transformers.__version__}")
logger.info(f"diffusers version: {importlib.metadata.version('diffusers')}")
try:
    logger.info(f"peft version: {importlib.metadata.version('peft')}")
except importlib.metadata.PackageNotFoundError:
    logger.info("peft not installed")
try:
    import xformers
    logger.info(f"xformers version: {xformers.__version__}")
except ImportError:
    logger.info("xformers not installed")

# Set PyTorch memory allocation config to reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def log_memory_usage():
    """Log both GPU and CPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GiB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GiB
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3  # GiB
        logger.debug(f"GPU Memory: Allocated {allocated:.2f} GiB, Reserved {reserved:.2f} GiB, Free {free:.2f} GiB")

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_usage = mem_info.rss / 1024**3  # GiB
    logger.debug(f"CPU Memory: RAM Usage {ram_usage:.2f} GiB")

def log_tensor_stats(tensor, name):
    """Log statistics of a tensor for debugging."""
    if tensor is not None:
        logger.debug(f"{name} stats: shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")

def summarize_prompt(prompt, max_tokens=300):
    """Summarize a prompt to fit within a token limit while retaining key themes."""
    words = prompt.split()
    if len(words) <= max_tokens:
        return prompt

    themes = [word for word in words if word.endswith(":")]
    if not themes:
        return " ".join(words[:max_tokens])

    summarized = []
    current_theme = None
    for word in words:
        if word.endswith(":"):
            current_theme = word
            summarized.append(word)
        elif current_theme and word != ";":
            summarized.append(word)
        if len(summarized) >= max_tokens:
            break
        if word == ";" and current_theme:
            current_theme = None

    return " ".join(summarized)

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

class SD15ChunkingPipeline(StableDiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,  # Disable safety checker to save memory
            feature_extractor=None,
            requires_safety_checker=False,
        )
        self.max_chunks = 4  # Process up to 300 tokens
        self.chunk_size = 75  # Tokens per chunk
        # Enable gradient checkpointing for UNet to save memory
        self.unet.enable_gradient_checkpointing()
        # Ensure evaluation mode
        self.unet.eval()
        self.vae.eval()

        # Do NOT enable xFormers due to compatibility issues
        logger.info("xFormers memory-efficient attention is disabled to avoid compatibility issues.")

        # Override the config
        self.register_to_config(
            _class_name="SD15ChunkingPipeline",
            _diffusers_version="0.29.2",
            vae=["diffusers", "AutoencoderKL"],
            text_encoder=["transformers", "CLIPTextModel"],
            tokenizer=["transformers", "CLIPTokenizer"],
            unet=["diffusers", "UNet2DConditionModel"],
            scheduler=["diffusers", "DPMSolverMultistepScheduler"],
        )

    @property
    def components(self):
        return {
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
            "unet": self.unet,
            "scheduler": self.scheduler,
        }

    def _split_prompt(self, prompt):
        """Split a prompt into chunks of chunk_size tokens."""
        words = prompt.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
        return chunks[: self.max_chunks]

    def _encode_prompt_chunk(self, chunk, tokenizer, text_encoder, device, dtype):
        """Encode a single prompt chunk through a text encoder."""
        inputs = tokenizer(
            chunk,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)
        with torch.no_grad():
            outputs = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
            embeddings = outputs.last_hidden_state  # Shape: (batch_size=1, seq_len, embed_dim)
        logger.debug(f"Chunk embeddings shape: {embeddings.shape}, Embeddings dtype: {embeddings.dtype}")
        log_tensor_stats(embeddings, "Chunk embeddings")
        return embeddings.to(dtype)

    def encode_prompt(
        self, prompt, device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None
    ):
        """Custom prompt encoding to handle chunking and concatenation."""
        batch_size = num_images_per_prompt
        dtype = torch.float32  # Use float32 for all operations
        logger.debug(f"Batch size: {batch_size}, Prompt: {prompt[:100]}..., Negative prompt: {negative_prompt[:100]}...")
        log_memory_usage()

        if negative_prompt is None:
            negative_prompt = ""

        # Summarize prompt if too long
        prompt = summarize_prompt(prompt, max_tokens=self.max_chunks * self.chunk_size)
        logger.debug(f"Summarized prompt: {prompt[:100]}...")

        # Handle prompt chunks
        prompt_chunks = (
            self._split_prompt(prompt)
            if isinstance(prompt, str)
            else [self._split_prompt(p) for p in prompt]
        )
        if isinstance(prompt, str):
            prompt_chunks = [prompt_chunks]
        logger.debug(f"Prompt chunks: {[len(chunks) for chunks in prompt_chunks]}")

        # Handle negative prompt chunks
        negative_chunks = (
            self._split_prompt(negative_prompt)
            if isinstance(negative_prompt, str)
            else [self._split_prompt(np) for np in negative_prompt]
        )
        if isinstance(negative_prompt, str):
            negative_chunks = [negative_chunks]
        logger.debug(f"Negative chunks: {[len(chunks) for chunks in negative_chunks]}")

        max_chunks = max(len(chunks) for chunks in prompt_chunks)
        logger.debug(f"Max chunks: {max_chunks}")

        for i in range(len(negative_chunks)):
            while len(negative_chunks[i]) < max_chunks:
                negative_chunks[i].append("")

        prompt_embeds_list = []
        negative_embeds_list = []

        for prompt_idx in range(batch_size):
            prompt_chunk_embeds = []
            chunks = prompt_chunks[prompt_idx % len(prompt_chunks)]
            logger.debug(f"Processing prompt_idx: {prompt_idx}, Chunks: {len(chunks)}")
            for chunk in chunks:
                embeds = self._encode_prompt_chunk(chunk, self.tokenizer, self.text_encoder, device, dtype)
                if embeds.dim() == 2:
                    embeds = embeds.unsqueeze(0)
                prompt_chunk_embeds.append(embeds)
            if prompt_chunk_embeds:
                prompt_embeds = torch.cat(prompt_chunk_embeds, dim=1)
            else:
                embed_dim = self.text_encoder.config.hidden_size
                prompt_embeds = torch.zeros((1, self.tokenizer.model_max_length, embed_dim), dtype=dtype, device=device)
            logger.debug(f"Prompt embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
            log_tensor_stats(prompt_embeds, "Prompt embeds")
            prompt_embeds_list.append(prompt_embeds)

            negative_chunk_embeds = []
            neg_chunks = negative_chunks[prompt_idx % len(negative_chunks)]
            logger.debug(f"Processing negative chunks: {len(neg_chunks)}")
            for chunk in neg_chunks:
                embeds = self._encode_prompt_chunk(chunk, self.tokenizer, self.text_encoder, device, dtype)
                if embeds.dim() == 2:
                    embeds = embeds.unsqueeze(0)
                negative_chunk_embeds.append(embeds)
            if negative_chunk_embeds:
                negative_embeds = torch.cat(negative_chunk_embeds, dim=1)
            else:
                negative_embeds = torch.zeros_like(prompt_embeds)
            logger.debug(f"Negative embeds shape: {negative_embeds.shape}, dtype: {negative_embeds.dtype}")
            log_tensor_stats(negative_embeds, "Negative embeds")
            negative_embeds_list.append(negative_embeds)

        prompt_embeds = torch.stack(prompt_embeds_list).to(dtype)
        if batch_size == 1:
            prompt_embeds = prompt_embeds.squeeze(0)
        logger.debug(f"Final prompt embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
        log_tensor_stats(prompt_embeds, "Final prompt embeds")

        negative_embeds = torch.stack(negative_embeds_list).to(dtype)
        if batch_size == 1:
            negative_embeds = negative_embeds.squeeze(0)
        logger.debug(f"Final negative embeds shape: {negative_embeds.shape}, dtype: {negative_embeds.dtype}")
        log_tensor_stats(negative_embeds, "Final negative embeds")

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
            logger.debug(f"After guidance, prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
            log_tensor_stats(prompt_embeds, "Prompt embeds after guidance")

        log_memory_usage()
        return prompt_embeds

    def __call__(
        self,
        prompt,
        height=512,  # Increased resolution
        width=512,
        num_inference_steps=30,  # Increased steps
        guidance_scale=7.5,  # Re-enable classifier-free guidance
        negative_prompt=None,
        num_images_per_prompt=1,
        **kwargs,
    ):
        """Generate images with chunked prompt processing."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = num_images_per_prompt
        dtype = torch.float32  # Use float32 for all operations
        do_classifier_free_guidance = guidance_scale > 1.0
        log_memory_usage()

        # Ensure all components are on the GPU
        self.unet.to(device, dtype=dtype)
        self.vae.to(device, dtype=dtype)
        self.text_encoder.to(device, dtype=dtype)

        prompt_embeds = self.encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Offload text encoder to CPU
        self.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        log_memory_usage()

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size,
            self.unet.config.in_channels,
            height,
            width,
            dtype,
            device,
            None,
        ).to(dtype)
        logger.debug(f"Latents shape: {latents.shape}, dtype: {latents.dtype}, requires_grad: {latents.requires_grad}")
        log_tensor_stats(latents, "Initial latents")
        log_memory_usage()

        # Move prompt_embeds to CPU initially, reload as needed
        prompt_embeds_cpu = prompt_embeds.to("cpu")
        del prompt_embeds
        torch.cuda.empty_cache()

        for t in timesteps:
            # Reload prompt_embeds to GPU for UNet inference
            prompt_embeds = prompt_embeds_cpu.to(device)

            # Move latents to GPU for this iteration
            latents = latents.to(device)

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).to(dtype)
            logger.debug(f"latent_model_input shape: {latent_model_input.shape}, dtype: {latent_model_input.dtype}, requires_grad: {latent_model_input.requires_grad}")
            log_tensor_stats(latent_model_input, "latent_model_input")

            # Clear cache before UNet inference to minimize fragmentation
            torch.cuda.empty_cache()

            noise_pred = self.unet(
                latent_model_input.to(device, dtype=dtype),
                t,
                encoder_hidden_states=prompt_embeds.to(device, dtype=dtype),
            ).sample
            logger.debug(f"noise_pred shape: {noise_pred.shape}, dtype: {noise_pred.dtype}, requires_grad: {noise_pred.requires_grad}")
            log_tensor_stats(noise_pred, "noise_pred")
            log_memory_usage()

            # Move prompt_embeds back to CPU to free GPU memory
            prompt_embeds = prompt_embeds.to("cpu")

            # Move latent_model_input to CPU to free GPU memory, keep noise_pred on GPU for scheduler.step
            latent_model_input = latent_model_input.to("cpu")
            torch.cuda.empty_cache()

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                log_tensor_stats(noise_pred, "noise_pred after guidance")

            noise_pred = noise_pred.to(dtype)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample.to(dtype)
            logger.debug(f"Updated latents shape: {latents.shape}, dtype: {latents.dtype}, requires_grad: {latents.requires_grad}")
            log_tensor_stats(latents, "Updated latents")

            # Move noise_pred and latents to CPU after scheduler.step
            noise_pred = noise_pred.to("cpu")
            latents = latents.to("cpu")
            del latent_model_input, noise_pred
            gc.collect()
            torch.cuda.empty_cache()

        # Reload latents to GPU for VAE decoding
        latents = latents.to(device)
        latents = (1 / self.vae.config.scaling_factor * latents).to(dtype)
        logger.debug(f"Latents before VAE decode shape: {latents.shape}, dtype: {latents.dtype}, requires_grad: {latents.requires_grad}")
        log_tensor_stats(latents, "Latents before VAE decode")
        with torch.no_grad():
            # Clear cache before VAE decoding
            torch.cuda.empty_cache()
            image = self.vae.decode(latents).sample
        logger.debug(f"VAE output image shape: {image.shape}, dtype: {image.dtype}, requires_grad: {image.requires_grad}")
        log_tensor_stats(image, "VAE output image")
        image = (image / 2 + 0.5).clamp(0, 1)
        log_tensor_stats(image, "Normalized image")
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        log_tensor_stats(torch.from_numpy(image), "Image before conversion to PIL")

        # Convert to PIL images (at 512x512 resolution)
        images = [Image.fromarray((img * 255).astype(np.uint8)) for img in image]
        log_memory_usage()
        del latents, image
        gc.collect()
        torch.cuda.empty_cache()
        return images

def load_theory_prompts(yaml_path):
    """Load theory prompts from YAML file."""
    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        prompts = config.get("theory_exp", {}).get("prompts", [])
        theory_prompts = {list(item.keys())[0]: list(item.values())[0] for item in prompts}
        return theory_prompts
    except Exception as e:
        logger.error(f"Failed to load YAML from {yaml_path}: {e}")
        raise

def load_id_list(id_list_path):
    """Load list of IDs from a text file."""
    try:
        with open(id_list_path, "r") as f:
            id_list = [line.strip() for line in f if line.strip()]
        return id_list
    except Exception as e:
        logger.error(f"Failed to load ID list from {id_list_path}: {e}")
        raise

def load_csv_rows_by_ids(csv_path, id_list, chunk_size=50):
    """Load CSV rows in chunks where 'file' matches values in id_list, excluding 'file' column."""
    id_list = [str(id_val) for id_val in id_list]
    prompts = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if "file" not in chunk.columns:
            raise ValueError("CSV does not contain a 'file' column")
        selected_rows = chunk[chunk["file"].astype(str).isin(id_list)]
        if selected_rows.empty:
            continue
        for _, row in selected_rows.iterrows():
            row_dict = row.to_dict()
            prompt_parts = [
                f"{key}: {value}"
                for key, value in row_dict.items()
                if key not in ["file"] and pd.notnull(value)
            ]
            txt_prompt = "; ".join(prompt_parts)
            prompts.append((row_dict["file"], txt_prompt))
        del chunk
        gc.collect()
    if not prompts:
        raise ValueError(f"No rows found in CSV with IDs: {id_list}")
    return prompts

def upscale_image(image_path, output_path, target_size=(1024, 1024)):
    """Upscale a single image to the target size and save it to the output path."""
    try:
        img = Image.open(image_path)
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_resized.save(output_path)
        logger.info(f"Upscaled and saved image to {output_path}")
        del img, img_resized
        gc.collect()
    except Exception as e:
        logger.error(f"Failed to upscale image {image_path}: {e}")


def refine_prompt_with_openai(openai_util, prompt, tokenizer, target_max_tokens=300):
    """Refine the prompt using OpenAI API to add visual cues while keeping token count within limit."""
    system_prompt = (
        "You are an expert prompt engineer for image generation. Your task is to refine the provided prompt by adding "
        "descriptive visual cues (e.g., 'with swirling shapes and a gradient of blues and reds to represent emotional depth') "
        "to make the image more meaningful and visually coherent, while preserving the core themes and instructions. "
        "Output only the refined prompt, with no additional explanations or formatting."
    )
    user_prompt = (
        f"Refine the following prompt by adding descriptive visual cues (e.g., swirling shapes, color gradients) to enhance its visual coherence for image generation, "
        f"while keeping the core themes and instructions intact. Target around {target_max_tokens} tokens. "
        f"Prompt:\n\n\"{prompt}\""
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = openai_util.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,  # Low temperature for focused refinement
            max_tokens=target_max_tokens + 50  # Buffer for token count
        )
        refined_prompt = response.choices[0].message.content.strip()
        token_count = len(tokenizer(refined_prompt, truncation=False)["input_ids"])
        logger.debug(f"Refined prompt token count: {token_count}")
        if token_count > target_max_tokens:
            refined_prompt = summarize_prompt(refined_prompt, max_tokens=target_max_tokens)
        return refined_prompt
    except Exception as e:
        logger.error(f"Failed to refine prompt with OpenAI: {e}")
        # Fallback: add a simple visual cue
        return f"{prompt}, with swirling shapes and a gradient of blues and reds to represent emotional depth"

def generate_images_for_csv_rows(
    csv_path,
    yaml_path,
    id_list_path,
    output_dir="/home/iris/Documents/deep_learning/generated_images/sd15_chunks_noRefiner",
    temp_dir="/home/iris/Documents/deep_learning/generated_images/sd15_chunks_noRefiner_temp",
    chunk_size=1,
):
    """Generate abstract images for CSV rows with matching IDs using theory prompts."""
    accelerator = Accelerator(
        cpu=False,
        mixed_precision="no",  # Disable mixed precision to use float32
        device_placement=True,  # Ensure device placement on GPU
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading Stable Diffusion 1.5 pipeline with Accelerator...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,  # Use float32 for all operations
        use_safetensors=True,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
        use_safetensors=True,
    )
    custom_pipeline = SD15ChunkingPipeline(
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
        tokenizer=pipeline.tokenizer,
        unet=pipeline.unet,
        scheduler=pipeline.scheduler,
    )
    custom_pipeline = accelerator.prepare(custom_pipeline)
    log_memory_usage()

    logger.info("Initializing OpenAIUtils for prompt refinement...")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        config = yaml.safe_load(open("/home/iris/Documents/deep_learning/config/config.yaml", "r"))
        api_key = config.get("openai", {}).get("api_key")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables or config file.")
    openai_util = OpenAIUtils(api_key=api_key)
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    theory_prompts = load_theory_prompts(yaml_path)
    id_list = load_id_list(id_list_path)

    row_prompts = load_csv_rows_by_ids(csv_path, id_list, chunk_size=50)
    os.makedirs(temp_dir, exist_ok=True)

    negative_prompt = (
        "literal therapy settings, text, words, letters, clinical environment, people in therapy, realistic depictions"
    )

    prompt_cache = LRUCache(capacity=50)
    i = 0

    chunked_rows = [row_prompts[i:i + chunk_size] for i in range(0, len(row_prompts), chunk_size)]
    for chunk_idx, chunk in enumerate(chunked_rows):
        logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunked_rows)} with {len(chunk)} rows...")
        for row_id, txt_prompt in chunk:
            logger.info(f"Processing row with ID: {row_id}")
            for theory, prompt_template in theory_prompts.items():
                base_prompt = prompt_template.format(txt_prompt=txt_prompt)
                prompt = f"An abstract painting representing {theory} themes: {base_prompt}, with vibrant colors and expressive patterns"
                refined_prompt = prompt_cache.get(prompt)
                if refined_prompt is None:
                    logger.debug(f"Base prompt before refinement: {prompt[:100]}...")
                    refined_prompt = refine_prompt_with_openai(openai_util, prompt, tokenizer, target_max_tokens=300)
                    prompt_cache.put(prompt, refined_prompt)
                    gc.collect()
                else:
                    logger.debug(f"Using cached refined prompt for: {prompt[:100]}...")
                logger.info(f"Generating image for theory: {theory}")
                if i == 0:
                    logger.info(f"Refined prompt: {refined_prompt[:100]}...")
                i += 1
                try:
                    images = custom_pipeline(
                        prompt=refined_prompt,
                        height=512,  # Increased resolution
                        width=512,
                        num_inference_steps=20,  # Increased steps
                        guidance_scale=7.5,  # Re-enable classifier-free guidance
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=1,
                    )
                    safe_id = str(row_id).replace("/", "_").replace("\\", "_")
                    temp_id_path = os.path.join(temp_dir, safe_id)
                    temp_image_path = os.path.join(temp_id_path, f"{safe_id}_{theory}.png")
                    output_path = os.path.join(output_dir, safe_id, f"{safe_id}_{theory}.png")
                    os.makedirs(temp_id_path, exist_ok=True)
                    images[0].save(temp_image_path)
                    logger.info(f"Saved temporary image (512x512) to {temp_image_path}")
                    upscale_image(temp_image_path, output_path, target_size=(1024, 1024))
                    try:
                        os.remove(temp_image_path)
                        os.rmdir(temp_id_path)
                    except Exception as e:
                        logger.warning(f"Failed to clean up {temp_image_path}: {e}")
                    del images
                    gc.collect()
                except Exception as e:
                    logger.error(f"Failed to generate image for ID {row_id}, theory {theory}: {e}")
                    accelerator.free_memory()
                    torch.cuda.empty_cache()
                    gc.collect()
                finally:
                    accelerator.free_memory()
                    torch.cuda.empty_cache()
                    gc.collect()
                    log_memory_usage()

        prompt_cache.cache.clear()
        gc.collect()
        del chunk
        gc.collect()

    logger.info(f"Cleaning up temporary directory: {temp_dir}")
    try:
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")

    logger.info("Image generation complete.")

if __name__ == "__main__":
    csv_path = "/home/iris/Documents/deep_learning/data/input_csv/FILE_SUPERTOPIC_DESCRIPTION.csv"
    yaml_path = "/home/iris/Documents/deep_learning/config/prompt_config.yaml"
    id_list_path = "/home/iris/Documents/deep_learning/data/sample_list.txt"
    generate_images_for_csv_rows(csv_path, yaml_path, id_list_path)