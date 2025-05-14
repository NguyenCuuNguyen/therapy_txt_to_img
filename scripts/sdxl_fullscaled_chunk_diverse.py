import os
import logging
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import pandas as pd
import yaml
from PIL import Image
import numpy as np
import gc
import re
import psutil
import torchvision
import transformers
import importlib.metadata
from accelerate import Accelerator
from transformers import T5Tokenizer
from src.utils.openai_utils import OpenAIUtils
from collections import OrderedDict

# Configure logging
log_dir = "/home/iris/Documents/deep_learning/src/logs"
log_file = os.path.join(log_dir, "sdxl_chunks_fullscaled_noRefiner_diverse.log")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info(f"torch version: {torch.__version__}")
logger.info(f"torchvision version: {torchvision.__version__}")
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,garbage_collection_threshold:0.6"

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    log_memory_usage()

def log_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        logger.debug(f"GPU Memory: Allocated {allocated:.2f} GiB, Reserved {reserved:.2f} GiB, Free {free:.2f} GiB")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_usage = mem_info.rss / 1024**3
    logger.debug(f"CPU Memory: RAM Usage {ram_usage:.2f} GiB")

def log_tensor_stats(tensor, name):
    if tensor is not None:
        logger.debug(f"{name} stats: shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")

def summarize_prompt(prompt, max_tokens=500):
    words = prompt.split()
    if len(words) <= max_tokens:
        return prompt
    priority_keywords = ["family", "relationship", "event", "obstacle", "self", "goal", "narrative", "conflict", "anxiety", "frustration", "emotion"]
    sentences = prompt.split(". ")
    prioritized_sentences = []
    other_sentences = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in priority_keywords):
            prioritized_sentences.append(sentence)
        else:
            other_sentences.append(sentence)
    summarized = []
    current_tokens = 0
    for sentence in prioritized_sentences + other_sentences:
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens <= max_tokens:
            summarized.append(sentence)
            current_tokens += sentence_tokens
        else:
            break
    return ". ".join(summarized) + ("." if summarized else "")

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

class SDXLChunkingPipeline(StableDiffusionXLPipeline):
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
        self.max_chunks = 4
        self.chunk_size = 100
        self.unet.enable_gradient_checkpointing()
        self.unet.eval()
        self.vae.eval()
        logger.info("xFormers memory-efficient attention is disabled to avoid compatibility issues.")
        self.register_to_config(
            _class_name="SDXLChunkingPipeline",
            _diffusers_version="0.29.2",
            vae=["diffusers", "AutoencoderKL"],
            text_encoder=["transformers", "CLIPTextModel"],
            text_encoder_2=["transformers", "CLIPTextModelWithProjection"],
            tokenizer=["transformers", "CLIPTokenizer"],
            tokenizer_2=["transformers", "CLIPTokenizer"],
            unet=["diffusers", "UNet2DConditionModel"],
            scheduler=["diffusers", "DPMSolverMultistepScheduler"],
        )

    @property
    def components(self):
        return {
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "unet": self.unet,
            "scheduler": self.scheduler,
        }

    def _split_prompt(self, prompt):
        words = prompt.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
        return chunks[: self.max_chunks]

    def _encode_prompt_chunk(self, chunk, tokenizer, text_encoder, device, dtype):
        with torch.no_grad():
            inputs = tokenizer(
                chunk,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(device)
            outputs = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
            embeddings = outputs.hidden_states[-2]
            pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]
        logger.debug(f"Chunk embeddings shape: {embeddings.shape}, Pooled shape: {pooled.shape}, Embeddings dtype: {embeddings.dtype}")
        log_tensor_stats(embeddings, "Chunk embeddings")
        log_tensor_stats(pooled, "Pooled embeddings")
        return embeddings.to(dtype), pooled.to(dtype)

    def encode_prompt(
        self, prompt, device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None
    ):
        batch_size = num_images_per_prompt
        dtype = torch.float32
        logger.debug(f"Batch size: {batch_size}, Prompt: {prompt[:100]}..., Negative prompt: {negative_prompt[:100]}...")
        log_memory_usage()
        if negative_prompt is None:
            negative_prompt = ""
        prompt = summarize_prompt(prompt, max_tokens=self.max_chunks * self.chunk_size)
        logger.debug(f"Summarized prompt: {prompt[:100]}...")
        prompt_chunks = (
            self._split_prompt(prompt)
            if isinstance(prompt, str)
            else [self._split_prompt(p) for p in prompt]
        )
        if isinstance(prompt, str):
            prompt_chunks = [prompt_chunks]
        logger.debug(f"Prompt chunks: {[len(chunks) for chunks in prompt_chunks]}")
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
        pooled_prompt_embeds_list = []
        negative_embeds_list = []
        negative_pooled_list = []
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]
        for prompt_idx in range(batch_size):
            prompt_chunk_embeds = []
            prompt_chunk_pooled = []
            chunks = prompt_chunks[prompt_idx % len(prompt_chunks)]
            logger.debug(f"Processing prompt_idx: {prompt_idx}, Chunks: {len(chunks)}")
            for chunk in chunks:
                chunk_embeds_per_encoder = []
                chunk_pooled_per_encoder = []
                for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                    embeds, pooled = self._encode_prompt_chunk(chunk, tokenizer, text_encoder, device, dtype)
                    if embeds.dim() == 2:
                        embeds = embeds.unsqueeze(0)
                    if pooled.dim() == 1:
                        pooled = pooled.unsqueeze(0)
                    chunk_embeds_per_encoder.append(embeds)
                    chunk_pooled_per_encoder.append(pooled)
                chunk_embeds = torch.cat(chunk_embeds_per_encoder, dim=-1)
                logger.debug(f"Chunk embeds shape after concat: {chunk_embeds.shape}, dtype: {chunk_embeds.dtype}")
                prompt_chunk_embeds.append(chunk_embeds)
                prompt_chunk_pooled.append(chunk_pooled_per_encoder[1])
            if prompt_chunk_embeds:
                prompt_embeds = torch.cat(prompt_chunk_embeds, dim=1)
            else:
                embed_dim = self.text_encoder.config.hidden_size + self.text_encoder_2.config.hidden_size
                prompt_embeds = torch.zeros((1, self.tokenizer.model_max_length, embed_dim), dtype=dtype, device=device)
            logger.debug(f"Prompt embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
            log_tensor_stats(prompt_embeds, "Prompt embeds")
            prompt_embeds_list.append(prompt_embeds)
            pooled = prompt_chunk_pooled[-1] if prompt_chunk_pooled else torch.zeros((1, self.text_encoder_2.config.hidden_size), dtype=dtype, device=device)
            logger.debug(f"Pooled prompt shape: {pooled.shape}, dtype: {pooled.dtype}")
            log_tensor_stats(pooled, "Pooled prompt")
            pooled_prompt_embeds_list.append(pooled)
            negative_chunk_embeds = []
            negative_chunk_pooled = []
            neg_chunks = negative_chunks[prompt_idx % len(negative_chunks)]
            logger.debug(f"Processing negative chunks: {len(neg_chunks)}")
            for chunk in neg_chunks:
                chunk_embeds_per_encoder = []
                chunk_pooled_per_encoder = []
                for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                    embeds, pooled = self._encode_prompt_chunk(chunk, tokenizer, text_encoder, device, dtype)
                    if embeds.dim() == 2:
                        embeds = embeds.unsqueeze(0)
                    if pooled.dim() == 1:
                        pooled = pooled.unsqueeze(0)
                    chunk_embeds_per_encoder.append(embeds)
                    chunk_pooled_per_encoder.append(pooled)
                chunk_embeds = torch.cat(chunk_embeds_per_encoder, dim=-1)
                logger.debug(f"Negative chunk embeds shape after concat: {chunk_embeds.shape}, dtype: {chunk_embeds.dtype}")
                negative_chunk_embeds.append(chunk_embeds)
                negative_chunk_pooled.append(chunk_pooled_per_encoder[1])
            if negative_chunk_embeds:
                negative_embeds = torch.cat(negative_chunk_embeds, dim=1)
            else:
                negative_embeds = torch.zeros_like(prompt_embeds)
            logger.debug(f"Negative embeds shape: {negative_embeds.shape}, dtype: {negative_embeds.dtype}")
            log_tensor_stats(negative_embeds, "Negative embeds")
            negative_embeds_list.append(negative_embeds)
            pooled = negative_chunk_pooled[-1] if negative_chunk_pooled else torch.zeros((1, self.text_encoder_2.config.hidden_size), dtype=dtype, device=device)
            logger.debug(f"Pooled negative shape: {pooled.shape}, dtype: {pooled.dtype}")
            log_tensor_stats(pooled, "Pooled negative")
            negative_pooled_list.append(pooled)
        prompt_embeds = torch.stack(prompt_embeds_list).to(dtype)
        if batch_size == 1:
            prompt_embeds = prompt_embeds.squeeze(0)
        logger.debug(f"Final prompt embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
        log_tensor_stats(prompt_embeds, "Final prompt embeds")
        pooled_prompt_embeds = torch.stack(pooled_prompt_embeds_list).to(dtype)
        if batch_size == 1:
            pooled_prompt_embeds = pooled_prompt_embeds.squeeze(0)
        logger.debug(f"Final pooled prompt embeds shape: {pooled_prompt_embeds.shape}, dtype: {pooled_prompt_embeds.dtype}")
        log_tensor_stats(pooled_prompt_embeds, "Final pooled prompt embeds")
        negative_embeds = torch.stack(negative_embeds_list).to(dtype)
        if batch_size == 1:
            negative_embeds = negative_embeds.squeeze(0)
        logger.debug(f"Final negative embeds shape: {negative_embeds.shape}, dtype: {negative_embeds.dtype}")
        log_tensor_stats(negative_embeds, "Final negative embeds")
        negative_pooled_embeds = torch.stack(negative_pooled_list).to(dtype)
        if batch_size == 1:
            negative_pooled_embeds = negative_pooled_embeds.squeeze(0)
        logger.debug(f"Final pooled negative embeds shape: {negative_pooled_embeds.shape}, dtype: {negative_pooled_embeds.dtype}")
        log_tensor_stats(negative_pooled_embeds, "Final pooled negative embeds")
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)
            logger.debug(f"After guidance, prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
            log_tensor_stats(prompt_embeds, "Prompt embeds after guidance")
            pooled_prompt_embeds = torch.cat([negative_pooled_embeds, pooled_prompt_embeds], dim=0)
            logger.debug(f"After guidance, pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}, dtype: {pooled_prompt_embeds.dtype}")
            log_tensor_stats(pooled_prompt_embeds, "Pooled prompt embeds after guidance")
        log_memory_usage()
        return prompt_embeds, {"text_embeds": pooled_prompt_embeds}

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        logger.debug(f"_get_add_time_ids inputs: original_size={original_size}, crops_coords_top_left={crops_coords_top_left}, target_size={target_size}")
        original_size = original_size or (1024, 1024)
        crops_coords_top_left = crops_coords_top_left or (0, 0)
        target_size = target_size or (1024, 1024)
        original_height, original_width = original_size
        crop_top, crop_left = crops_coords_top_left
        target_height, target_width = target_size
        original_height = original_height or 1024
        original_width = original_width or 1024
        crop_top = crop_top or 0
        crop_left = crop_left or 0
        target_height = target_height or 1024
        target_width = target_width or 1024
        logger.debug(f"Validated inputs: original_size=({original_height}, {original_width}), crops_coords_top_left=({crop_top}, {crop_left}), target_size=({target_height}, {target_width})")
        add_time_ids = list((original_height, original_width) + (crop_top, crop_left) + (target_height, target_width))
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        logger.debug(f"add_time_ids shape: {add_time_ids.shape}, dtype: {add_time_ids.dtype}")
        log_tensor_stats(add_time_ids, "add_time_ids")
        return add_time_ids

    def __call__(
        self,
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=150,
        guidance_scale=12.0,
        negative_prompt=None,
        num_images_per_prompt=1,
        **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = num_images_per_prompt
        dtype = torch.float32
        log_memory_usage()
        self.unet.to(device, dtype=dtype)
        self.vae.to(device, dtype=dtype)
        self.text_encoder.to(device, dtype=dtype)
        self.text_encoder_2.to(device, dtype=dtype)
        with torch.no_grad():
            prompt_embeds, added_cond_kwargs = self.encode_prompt(
                prompt, device, num_images_per_prompt, guidance_scale > 1.0, negative_prompt
            )
        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")
        cleanup_memory()
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
        add_time_ids = self._get_add_time_ids(
            (height, width), (0, 0), (height, width), dtype=dtype
        ).to(device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        if guidance_scale > 1.0:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        logger.debug(f"add_time_ids shape: {add_time_ids.shape}, dtype: {add_time_ids.dtype}, requires_grad: {add_time_ids.requires_grad}")
        log_tensor_stats(add_time_ids, "add_time_ids")
        added_cond_kwargs["time_ids"] = add_time_ids.to(device, dtype=dtype)
        prompt_embeds_cpu = prompt_embeds.to("cpu")
        added_cond_kwargs_cpu = {k: v.to("cpu") for k, v in added_cond_kwargs.items()}
        del prompt_embeds, added_cond_kwargs
        cleanup_memory()
        for t in timesteps:
            prompt_embeds = prompt_embeds_cpu.to(device)
            added_cond_kwargs = {k: v.to(device) for k, v in added_cond_kwargs_cpu.items()}
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t).to(dtype)
            logger.debug(f"latent_model_input shape: {latent_model_input.shape}, dtype: {latent_model_input.dtype}, requires_grad: {latent_model_input.requires_grad}")
            log_tensor_stats(latent_model_input, "latent_model_input")
            cleanup_memory()
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input.to(device, dtype=dtype),
                    t,
                    encoder_hidden_states=prompt_embeds.to(device, dtype=dtype),
                    added_cond_kwargs={k: v.to(device, dtype=dtype) for k, v in added_cond_kwargs.items()},
                ).sample
            logger.debug(f"noise_pred shape: {noise_pred.shape}, dtype: {noise_pred.dtype}, requires_grad: {noise_pred.requires_grad}")
            log_tensor_stats(noise_pred, "noise_pred")
            log_memory_usage()
            prompt_embeds = prompt_embeds.to("cpu")
            added_cond_kwargs = {k: v.to("cpu") for k, v in added_cond_kwargs.items()}
            latent_model_input = latent_model_input.to("cpu")
            cleanup_memory()
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                log_tensor_stats(noise_pred, "noise_pred after guidance")
            noise_pred = noise_pred.to(dtype)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample.to(dtype)
            logger.debug(f"Updated latents shape: {latents.shape}, dtype: {latents.dtype}, requires_grad: {latents.requires_grad}")
            log_tensor_stats(latents, "Updated latents")
            noise_pred = noise_pred.to("cpu")
            del latent_model_input, noise_pred
            cleanup_memory()
        latents = (1 / self.vae.config.scaling_factor * latents).to(dtype)
        logger.debug(f"Latents before VAE decode shape: {latents.shape}, dtype: {latents.dtype}, requires_grad: {latents.requires_grad}")
        log_tensor_stats(latents, "Latents before VAE decode")
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        logger.debug(f"VAE output image shape: {image.shape}, dtype: {image.dtype}, requires_grad: {image.requires_grad}")
        log_tensor_stats(image, "VAE output image")
        image = (image / 2 + 0.5).clamp(0, 1)
        log_tensor_stats(image, "Normalized image")
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        log_tensor_stats(torch.from_numpy(image), "Image before conversion to PIL")
        images = [Image.fromarray((img * 255).astype(np.uint8)) for img in image]
        log_memory_usage()
        del latents, image
        cleanup_memory()
        return images

def load_theory_prompts(yaml_path):
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
    try:
        with open(id_list_path, "r") as f:
            id_list = [line.strip() for line in f if line.strip()]
        return id_list
    except Exception as e:
        logger.error(f"Failed to load ID list from {id_list_path}: {e}")
        raise

def load_csv_rows_by_ids(csv_path, id_list, chunk_size=50):
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
            key_themes = {
                "emotions": [],
                "events": [],
                "relationships": [],
                "themes": [],
                "priority_details": []
            }
            priority_fields = [
                "family_relationships", "relationship_issues", "life_events", "personal_obstacles",
                "self_perception", "personal_goals", "narrative", "inner_conflict", "anxiety", "frustration",
                "communication", "support_systems", "support_networks", "mental_health_management",
                "coping_strategies", "identity"
            ]
            for key, value in row_dict.items():
                if key not in ["file"] and pd.notnull(value):
                    value = value.strip()
                    if len(value.split()) > 20:
                        value = " ".join(value.split()[:20]) + "..."
                    if key in ["anxiety", "emotional_health", "frustration", "inner_conflict", "fear"]:
                        key_themes["emotions"].append(f"{key}: {value}")
                    elif key in ["life_events", "employment", "parenting", "therapy_process"]:
                        key_themes["events"].append(f"{key}: {value}")
                    elif key in ["family_relationships", "relationship_issues", "communication", "support_systems", "support_networks"]:
                        key_themes["relationships"].append(f"{key}: {value}")
                    else:
                        key_themes["themes"].append(f"{key}: {value}")
                    if key in priority_fields:
                        key_themes["priority_details"].append(f"{key}: {value}")
            txt_prompt = "; ".join([f"{key}: {', '.join(values) if isinstance(values, list) else values}" for key, values in key_themes.items() if values and key != "priority_details"])
            prompts.append((row_dict["file"], txt_prompt, key_themes))
        del chunk
        cleanup_memory()
    if not prompts:
        raise ValueError(f"No rows found in CSV with IDs: {id_list}")
    return prompts

def extract_theory_essence(prompt):
    match = re.search(r'\{txt_prompt\}\.\s*(.*?)\s*Do not include', prompt)
    if match:
        return match.group(1)
    return ""

def generate_metaphors_with_openai(openai_util, key_themes, theory, guideline, tokenizer, max_tokens=200):
    system_prompt = (
        f"You are an expert in visual storytelling for {theory} psychotherapy. Generate unique, photorealistic visual metaphors for the provided transcript details, ensuring each metaphor vividly reflects the specific content and emotional tone. "
        f"Based on the transcript’s tone (e.g., chaotic, hopeful), decide whether the metaphors form a fragmented collage or a cohesive scene. "
        f"Align metaphors with the {theory} perspective, using {guideline['metaphor_style']} and incorporating {', '.join(guideline['key_elements'])}. "
        f"Output a list of metaphors, one for each priority detail, in the format: '[Detail]: [Metaphor description].' Avoid generic imagery (e.g., oceans, cliffs). Target ~{max_tokens} tokens."
    )
    user_prompt = (
        f"Generate visual metaphors for these transcript details in a {theory} perspective:\n"
        f"Priority Details: {', '.join(key_themes['priority_details']) if key_themes['priority_details'] else 'not specified'}.\n"
        f"Emotions: {', '.join(key_themes['emotions']) if key_themes['emotions'] else 'not specified'}.\n"
        f"Events: {', '.join(key_themes['events']) if key_themes['events'] else 'not specified'}.\n"
        f"Relationships: {', '.join(key_themes['relationships']) if key_themes['relationships'] else 'not specified'}.\n"
        f"Themes: {', '.join(key_themes['themes']) if key_themes['themes'] else 'not specified'}.\n"
        f"Guidelines:\n"
        f"- Focus: {guideline['focus']}.\n"
        f"- Tone: {guideline['tone']}.\n"
        f"- Metaphor Style: {guideline['metaphor_style']}.\n"
        f"- Key Elements: {', '.join(guideline['key_elements'])}.\n"
        f"Output metaphors as a list, ensuring uniqueness and specificity. Target ~{max_tokens} tokens."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = openai_util.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens + 50
        )
        metaphor_text = response.choices[0].message.content.strip()
        metaphors = [line.strip() for line in metaphor_text.split('\n') if line.strip() and ': ' in line]
        token_count = len(tokenizer(metaphor_text, truncation=False)["input_ids"])
        logger.debug(f"Generated metaphors token count: {token_count}")
        return metaphors
    except Exception as e:
        logger.error(f"Failed to generate metaphors with OpenAI: {e}")
        return []

def refine_prompt_with_openai(openai_util, base_prompt, tokenizer, key_themes, theory, target_max_tokens=500):
    theory_guidelines = {
        "cbt": {
            "focus": "interplay of thoughts, emotions, behaviors",
            "tone": "dynamic, structured, high-contrast",
            "metaphor_style": "technological, urban, or mechanical elements to represent cognitive processes",
            "key_elements": ["balance (e.g., scales, bridges)", "distortion (e.g., mirrors, warped surfaces)", "clarity (e.g., light, lenses)"]
        },
        "psychodynamic": {
            "focus": "past relationships influencing present",
            "tone": "ethereal, layered, introspective",
            "metaphor_style": "historical, organic, or dreamlike elements to evoke memory and emotion",
            "key_elements": ["repressed elements (e.g., locked objects, hidden paths)", "connections (e.g., roots, threads)", "reflections (e.g., pools, mirrors)"]
        },
        "narrative": {
            "focus": "unfolding story of identity and transformation",
            "tone": "vibrant, flowing, narrative-driven",
            "metaphor_style": "journey-based, artistic, or storytelling elements to represent life arcs",
            "key_elements": ["paths (e.g., roads, rivers)", "stories (e.g., books, tapestries)", "guidance (e.g., lights, stars)"]
        }
    }
    guideline = theory_guidelines.get(theory, {
        "focus": "general emotional landscape",
        "tone": "neutral",
        "metaphor_style": "abstract",
        "key_elements": ["generic emotional symbols"]
    })
    metaphor_mappings = generate_metaphors_with_openai(openai_util, key_themes, theory, guideline, tokenizer, max_tokens=200)
    system_prompt = (
        f"You are an expert prompt engineer specializing in visual storytelling for {theory} psychotherapy. Craft a symbolic, photorealistic image prompt that vividly captures the client’s unique emotional landscape, inner conflicts, relationships, and life themes, prioritizing the specific details from 'Priority Details,' 'Emotions,' 'Events,' and 'Relationships.' "
        f"Use the provided transcript-specific metaphors and guidelines to create a highly detailed, evocative scene or collage, reflecting the transcript’s emotional tone. Each visual element must directly reflect the transcript’s unique content, weaving in short, impactful phrases from the details to inspire vivid imagery. "
        f"Apply the theory guidelines as a stylization layer, ensuring the output is coherent, fits a 1024x1024 image without cropping, and excludes text, therapy settings, or generic imagery (e.g., oceans, cliffs, forests)."
    )
    user_prompt = (
        f"Enhance the following base prompt for a {theory} perspective, ensuring it {guideline['focus'].lower()}. "
        f"Prioritize these transcript details:\n"
        f"Priority Details: {', '.join(key_themes['priority_details']) if key_themes['priority_details'] else 'not specified'}.\n"
        f"Emotions: {', '.join(key_themes['emotions']) if key_themes['emotions'] else 'not specified'}.\n"
        f"Events: {', '.join(key_themes['events']) if key_themes['events'] else 'not specified'}.\n"
        f"Relationships: {', '.join(key_themes['relationships']) if key_themes['relationships'] else 'not specified'}.\n"
        f"Themes: {', '.join(key_themes['themes']) if key_themes['themes'] else 'not specified'}.\n"
        f"Guidelines:\n"
        f"- Focus: {guideline['focus']}.\n"
        f"- Tone: {guideline['tone']}.\n"
        f"- Metaphor Style: {guideline['metaphor_style']}.\n"
        f"- Key Elements: Include {', '.join(guideline['key_elements'])}.\n"
        f"- Specific Metaphors: {', '.join(metaphor_mappings) if metaphor_mappings else 'generate unique metaphors based on the details, weaving in phrases from Priority Details'}.\n"
        f"Requirements:\n"
        f"- Craft a unique scene or collage that vividly reflects the transcript’s content, integrating short phrases from Priority Details into the narrative flow.\n"
        f"- Decide whether a collage or cohesive scene based on the transcript’s emotional tone (e.g., collage for chaos, scene for hope).\n"
        f"- Avoid therapy settings, text, or generic scenes.\n"
        f"- Ensure each visual element ties directly to the transcript for maximum specificity.\n"
        f"Target ~{target_max_tokens} tokens.\n\nBase Prompt:\n{base_prompt}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = openai_util.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=target_max_tokens + 50
        )
        refined_prompt = response.choices[0].message.content.strip()
        token_count = len(tokenizer(refined_prompt, truncation=False)["input_ids"])
        logger.debug(f"Refined prompt token count: {token_count}")
        if token_count > target_max_tokens:
            refined_prompt = summarize_prompt(refined_prompt, max_tokens=target_max_tokens)
        second_user_prompt = (
            f"Further refine the following prompt to make it even more specific to the transcript details, ensuring each visual element vividly reflects the unique 'Priority Details,' 'Emotions,' 'Events,' and 'Relationships.' "
            f"Integrate short phrases from these details into a cohesive narrative or collage, maintaining the {theory} perspective and {guideline['tone']} tone. "
            f"Avoid generic imagery and ensure the prompt fits a 1024x1024 image without cropping.\n\nPrompt:\n{refined_prompt}"
        )
        second_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": second_user_prompt}
        ]
        response = openai_util.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=second_messages,
            temperature=0.5,
            max_tokens=target_max_tokens + 50
        )
        final_prompt = response.choices[0].message.content.strip()
        token_count = len(tokenizer(final_prompt, truncation=False)["input_ids"])
        logger.debug(f"Final refined prompt token count: {token_count}")
        if token_count > target_max_tokens:
            final_prompt = summarize_prompt(final_prompt, max_tokens=target_max_tokens)
        return final_prompt
    except Exception as e:
        logger.error(f"Failed to refine prompt with OpenAI: {e}")
        return base_prompt

def generate_images_for_csv_rows(
    csv_path,
    yaml_path,
    id_list_path,
    output_dir="/home/iris/Documents/deep_learning/generated_images/sdxl/chunk_fullscaled_diverse_gpt_iter",
    temp_dir="/home/iris/Documents/deep_learning/generated_images/sdxl/chunk_fullscaled_diverse_gpt_iter_temp",
    chunk_size=1,
):
    accelerator = Accelerator(
        cpu=False,
        mixed_precision="no",
        device_placement=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading SDXL pipeline with Accelerator...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler",
        use_safetensors=True,
    )
    custom_pipeline = SDXLChunkingPipeline(
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
        text_encoder_2=pipeline.text_encoder_2,
        tokenizer=pipeline.tokenizer,
        tokenizer_2=pipeline.tokenizer_2,
        unet=pipeline.unet,
        scheduler=pipeline.scheduler,
    )
    custom_pipeline = accelerator.prepare(custom_pipeline)
    custom_pipeline.unet.to(device)
    custom_pipeline.vae.to(device)
    custom_pipeline.text_encoder.to(device)
    custom_pipeline.text_encoder_2.to(device)
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
    theory_essences = {theory: extract_theory_essence(prompt) for theory, prompt in theory_prompts.items()}
    id_list = load_id_list(id_list_path)
    row_prompts = load_csv_rows_by_ids(csv_path, id_list, chunk_size=50)
    os.makedirs(output_dir, exist_ok=True)
    negative_prompt = (
        "text, words, letters, low quality, blurry, distorted, generic landscapes, repetitive imagery, overly abstract, oceans, cliffs, forests, therapy settings, clinical environments"
    )
    prompt_cache = LRUCache(capacity=100)
    chunked_rows = [row_prompts[i:i + chunk_size] for i in range(0, len(row_prompts), chunk_size)]
    for chunk_idx, chunk in enumerate(chunked_rows):
        logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunked_rows)} with {len(chunk)} rows...")
        for row_id, txt_prompt, key_themes in chunk:
            logger.info(f"Processing row with ID: {row_id}")
            for theory, prompt_template in theory_prompts.items():
                base_prompt = prompt_template.format(txt_prompt=txt_prompt)
                refined_prompt = prompt_cache.get(base_prompt)
                if refined_prompt is None:
                    logger.debug(f"Base prompt before refinement: {base_prompt}...")
                    refined_prompt = refine_prompt_with_openai(openai_util, base_prompt, tokenizer, key_themes, theory, target_max_tokens=500)
                    prompt_cache.put(base_prompt, refined_prompt)
                    cleanup_memory()
                else:
                    logger.debug(f"Using cached refined prompt for: {base_prompt}...")
                logger.info(f"Generating image for theory: {theory}")
                logger.info(f"Refined prompt: {refined_prompt}...")
                try:
                    images = custom_pipeline(
                        prompt=refined_prompt,
                        height=1024,
                        width=1024,
                        num_inference_steps=150,
                        guidance_scale=12.0,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=1,
                    )
                    safe_id = str(row_id).replace("/", "_").replace("\\", "_")
                    output_path = os.path.join(output_dir, safe_id, f"{safe_id}_{theory}.png")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    images[0].save(output_path)
                    logger.info(f"Saved image to {output_path}")
                    del images
                    cleanup_memory()
                except Exception as e:
                    logger.error(f"Failed to generate image for ID {row_id}, theory {theory}: {e}")
                    accelerator.free_memory()
                    cleanup_memory()
                finally:
                    accelerator.free_memory()
                    cleanup_memory()
                    log_memory_usage()
        prompt_cache.cache.clear()
        cleanup_memory()
        del chunk
        cleanup_memory()
    logger.info("Image generation complete.")

if __name__ == "__main__":
    csv_path = "/home/iris/Documents/deep_learning/data/input_csv/FILE_SUPERTOPIC_DESCRIPTION.csv"
    yaml_path = "/home/iris/Documents/deep_learning/config/prompt_config.yaml"
    id_list_path = "/home/iris/Documents/deep_learning/data/sample_list.txt"
    generate_images_for_csv_rows(csv_path, yaml_path, id_list_path)