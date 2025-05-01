import os
import torch
import logging
import yaml
import json
import pandas as pd
import gc
import re # For finding epoch numbers
import traceback
from pathlib import Path # For easier path handling
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline, # For SDXL Refiner
    DiffusionPipeline,
    AutoencoderKL,
    VQModel,
    DPMSolverMultistepScheduler,
    KandinskyV22PriorPipeline, # Import specific Kandinsky classes
    KandinskyV22Pipeline,
    UNet2DConditionModel # Needed for component loading
)
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer
# from peft import PeftModel # Not needed if using load_adapter/load_lora_weights

#for summarizing prompt
from transformers import pipeline as hf_pipeline # Use alias to avoid conflict if pipeline var used elsewhere
import math # For ceiling division
# Global variable to cache the summarizer pipeline
summarizer_pipeline = None

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/iris/Documents/deep_learning/src/logs/iter2_image_generation.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        raise

def load_prompt_variations(yaml_path):
    """Loads prompt variations from a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            prompt_list = config.get('theory_exp', {}).get('prompts', [])
            variations = {k: v for item in prompt_list for k, v in item.items()}
            if not variations: logger.warning(f"No prompt variations found in {yaml_path}")
            else: logger.info(f"Loaded {len(variations)} prompt variations from {yaml_path}")
            return variations
    except FileNotFoundError: logger.error(f"Prompt variations file not found: {yaml_path}"); return {}
    except Exception as e: logger.error(f"Failed to load prompt variations from {yaml_path}: {e}"); return {}

def load_sample_ids(txt_path):
    """Loads a list of IDs from a text file (one ID per line)."""
    try:
        with open(txt_path, 'r') as f:
            ids = {str(line.strip()) for line in f if line.strip()}
            logger.info(f"Loaded {len(ids)} sample IDs from {txt_path}")
            return ids
    except FileNotFoundError: logger.error(f"Sample ID list file not found: {txt_path}"); return set()
    except Exception as e: logger.error(f"Failed to load sample IDs from {txt_path}: {e}"); return set()

def find_checkpoint_dirs(base_model_dir):
    """Finds 'best' and 'last' checkpoint directories."""
    checkpoints = {"best": None, "last": None}
    best_epoch = -1; last_epoch = -1
    if not os.path.isdir(base_model_dir): logger.warning(f"Base directory not found: {base_model_dir}"); return checkpoints
    for dirname in os.listdir(base_model_dir):
        dirpath = os.path.join(base_model_dir, dirname)
        if os.path.isdir(dirpath):
            best_match = re.match(r'best_epoch_(\d+)', dirname)
            last_match = re.match(r'last_epoch_(\d+)', dirname)
            if best_match:
                try:
                    epoch = int(best_match.group(1))
                    if epoch > best_epoch: checkpoints["best"] = dirpath; best_epoch = epoch
                except ValueError: logger.warning(f"Could not parse epoch from dir: {dirname}")
            elif last_match:
                 try:
                    epoch = int(last_match.group(1))
                    if epoch > last_epoch: checkpoints["last"] = dirpath; last_epoch = epoch
                 except ValueError: logger.warning(f"Could not parse epoch from dir: {dirname}")
    if checkpoints["best"]: logger.info(f"Found best checkpoint: {checkpoints['best']} (Epoch {best_epoch})")
    else: logger.warning(f"No 'best' checkpoint directory found in {base_model_dir}")
    if checkpoints["last"]: logger.info(f"Found last checkpoint: {checkpoints['last']} (Epoch {last_epoch})")
    else: logger.warning(f"No 'last' checkpoint directory found in {base_model_dir}")
    return checkpoints

# !--- Modified load_pipeline_with_lora ---!
def load_pipeline_with_lora(model_name, base_model_ids, lora_checkpoint_dir, device, target_dtype):
    """Loads the base pipeline(s) and attaches LoRA weights."""
    logger.info(f"Loading base model components for {model_name}...")
    pipeline = None
    prior_pipeline = None
    lora_weight_name = "adapter_model.safetensors"
    lora_weight_name_bin = "adapter_model.bin" # Fallback

    try:
        # --- Load Base Models ---
        if model_name == "sdxl":
            base_model_id = base_model_ids["sdxl"]
            logger.info(f"Loading SDXL Pipeline components in {target_dtype}...")
            """ Load the entire pipeline, including VAE, in the target dtype (fp16)
            # This ensures VAE input/output match during decode"""
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                base_model_id,
                torch_dtype=target_dtype, # Load ALL components in target_dtype
                variant="fp16", # Use fp16 variant if available
                use_safetensors=True,
                # VAE will be loaded in target_dtype by the pipeline
            )
            if hasattr(pipeline, 'vae'):
                 logger.info(f"Pipeline VAE loaded with dtype: {pipeline.vae.dtype}") # Should be target_dtype
                 pipeline.vae.eval() # Ensure eval mode
            else:
                 logger.warning("Pipeline loaded without a VAE attribute.")
            logger.info("Loaded SDXL base pipeline.")

            # --- Load SDXL LoRA using pipeline method ---
            logger.info(f"Loading SDXL LoRA weights from: {lora_checkpoint_dir}")
            # Check if the main weight file exists in the checkpoint directory itself
            main_safe_path = os.path.join(lora_checkpoint_dir, lora_weight_name)
            main_bin_path = os.path.join(lora_checkpoint_dir, lora_weight_name_bin)

            if os.path.exists(main_safe_path) or os.path.exists(main_bin_path):
                 weight_to_load = lora_weight_name if os.path.exists(main_safe_path) else lora_weight_name_bin
                 logger.info(f"Found main LoRA file '{weight_to_load}', loading from base directory: {lora_checkpoint_dir}")
                 pipeline.load_lora_weights(lora_checkpoint_dir, weight_name=weight_to_load)
                 logger.info("Loaded LoRA weights into SDXL pipeline from base directory.")
            else:
                 # Fallback: Try loading from subfolders if main file not found
                 logger.warning(f"Main LoRA weight file not found in {lora_checkpoint_dir}. Attempting to load from subfolders (unet_lora, text_encoder_lora, etc.)...")
                 unet_lora_path = os.path.join(lora_checkpoint_dir, "unet_lora")
                 te1_lora_path = os.path.join(lora_checkpoint_dir, "text_encoder_lora")
                 te2_lora_path = os.path.join(lora_checkpoint_dir, "text_encoder_2_lora")

                 def get_lora_weight_name_sub(path):
                     safetensors_path = os.path.join(path, lora_weight_name)
                     bin_path = os.path.join(path, lora_weight_name_bin)
                     if os.path.exists(safetensors_path): return lora_weight_name
                     elif os.path.exists(bin_path): return lora_weight_name_bin
                     else: return None

                 # Load UNet LoRA from subfolder
                 if os.path.exists(unet_lora_path):
                     weight_file = get_lora_weight_name_sub(unet_lora_path)
                     if weight_file:
                         try:
                             logger.info(f"Loading SDXL UNet LoRA from subfolder {unet_lora_path} using {weight_file}...")
                             pipeline.load_lora_weights(unet_lora_path, weight_name=weight_file)
                             logger.info("Loaded SDXL UNet LoRA from subfolder.")
                         except Exception as e: logger.error(f"Failed to load SDXL UNet LoRA from subfolder: {e}")
                     else: logger.warning(f"No weight file found in {unet_lora_path}")

                 # Load TE1 LoRA from subfolder
                 if os.path.exists(te1_lora_path) and hasattr(pipeline, 'text_encoder'):
                     weight_file = get_lora_weight_name_sub(te1_lora_path)
                     if weight_file:
                         try:
                             logger.info(f"Loading SDXL TE1 LoRA from subfolder {te1_lora_path} using {weight_file}...")
                             pipeline.load_lora_weights(te1_lora_path, weight_name=weight_file, text_encoder=pipeline.text_encoder)
                             logger.info("Loaded SDXL Text Encoder 1 LoRA from subfolder.")
                         except Exception as e: logger.error(f"Failed to load SDXL TE1 LoRA from subfolder: {e}")
                     else: logger.warning(f"No weight file found in {te1_lora_path}")

                 # Load TE2 LoRA from subfolder
                 if os.path.exists(te2_lora_path) and hasattr(pipeline, 'text_encoder_2'):
                     weight_file = get_lora_weight_name_sub(te2_lora_path)
                     if weight_file:
                         try:
                             logger.info(f"Loading SDXL TE2 LoRA from subfolder {te2_lora_path} using {weight_file}...")
                             pipeline.load_lora_weights(te2_lora_path, weight_name=weight_file, text_encoder=pipeline.text_encoder_2)
                             logger.info("Loaded SDXL Text Encoder 2 LoRA from subfolder.")
                         except Exception as e: logger.error(f"Failed to load SDXL TE2 LoRA from subfolder: {e}")
                     else: logger.warning(f"No weight file found in {te2_lora_path}")


        elif model_name == "kandinsky":
            # --- Kandinsky Loading (using load_adapter) ---
            prior_id = base_model_ids["kandinsky_prior"]
            decoder_id = base_model_ids["kandinsky_decoder"]
            logger.info(f"Loading Kandinsky Prior Pipeline ({prior_id})...")
            prior_pipeline = KandinskyV22PriorPipeline.from_pretrained(prior_id, torch_dtype=target_dtype)
            logger.info(f"Loading Kandinsky Decoder Pipeline ({decoder_id})...")
            pipeline = KandinskyV22Pipeline.from_pretrained(decoder_id, torch_dtype=target_dtype, use_safetensors=True)
            try:
                # !--- Load Kandinsky VAE in target_dtype (fp16) ---!
                logger.info(f"Loading Kandinsky VAE (MoVQ) in {target_dtype} from {decoder_id}/movq")
                vae = VQModel.from_pretrained(decoder_id, subfolder="movq", torch_dtype=target_dtype) # Load in fp16
                pipeline.movq = vae # Assign the fp16 VAE
                logger.info(f"Loaded and assigned Kandinsky VQ VAE ({pipeline.movq.dtype}).")
            except Exception as e: logger.warning(f"Could not load separate VQ VAE for Kandinsky: {e}")
            logger.info("Loaded Kandinsky base pipelines.")

            # --- Load LoRA Adapters for Kandinsky ---
            logger.info(f"Loading LoRA adapters from: {lora_checkpoint_dir}")
            unet_lora_path = os.path.join(lora_checkpoint_dir, "unet_lora")
            te1_lora_path = os.path.join(lora_checkpoint_dir, "text_encoder_lora") # Prior's TE

            # Helper for load_adapter
            def _load_adapter_safely(component, adapter_path):
                if not component or not hasattr(component, 'load_adapter'): logger.warning(f"Component {component.__class__.__name__ if component else 'None'} does not support load_adapter."); return False
                if not os.path.isdir(adapter_path): logger.warning(f"Adapter path not found: {adapter_path}"); return False
                lora_weight_name_safe = "adapter_model.safetensors"; lora_weight_name_bin = "adapter_model.bin"
                if not (os.path.exists(os.path.join(adapter_path, lora_weight_name_safe)) or os.path.exists(os.path.join(adapter_path, lora_weight_name_bin))):
                    logger.warning(f"No adapter weight file found in {adapter_path}"); return False
                try:
                    logger.info(f"Loading adapter from {adapter_path} into {component.__class__.__name__}")
                    component.load_adapter(adapter_path) # Finds weights automatically
                    logger.info(f"Successfully loaded adapter into {component.__class__.__name__}")
                    return True
                except Exception as e: logger.error(f"Failed to load adapter from {adapter_path}: {e}"); return False

            _load_adapter_safely(pipeline.unet, unet_lora_path)
            _load_adapter_safely(prior_pipeline.text_encoder, te1_lora_path)
            # -----------------------------------------

        else:
            raise ValueError(f"Unsupported model_name for generation: {model_name}")

        # Move pipelines to device
        if pipeline: pipeline.to(device)
        if prior_pipeline: prior_pipeline.to(device)

        logger.info(f"Finished loading pipeline(s) and LoRA for {model_name}.")
        return pipeline, prior_pipeline

    except Exception as e:
        logger.error(f"Failed during model or LoRA loading for {model_name} from {lora_checkpoint_dir}: {e}\n{traceback.format_exc()}")
        return None, None

def summarize_long_prompt(prompt: str, tokenizer, max_length: int = 77, min_summary_tokens: int = 20):
    """
    Checks prompt length using the provided tokenizer and summarizes if it exceeds max_length.

    Args:
        prompt (str): The input text prompt.
        tokenizer: The tokenizer instance (e.g., CLIPTokenizer) to check length accurately.
        max_length (int): The target maximum token length (e.g., 77 for CLIP).
        min_summary_tokens (int): Minimum desired length for the summary in tokens.

    Returns:
        str: The original prompt or a summarized version.
    """
    global summarizer_pipeline # Allow modification of the global cache
    logger.info(f"the prompt is {prompt}")
    if not prompt or not isinstance(prompt, str):
        logger.warning("Invalid prompt passed to summarizer.")
        return ""

    # 1. Check length using the target model's tokenizer
    # We subtract 2 for potential start/end tokens the tokenizer might add
    target_max_tokens = max_length - 2
    token_ids = tokenizer(prompt, max_length=max_length, truncation=False)["input_ids"]

    if len(token_ids) <= target_max_tokens:
        logger.debug(f"Prompt length ({len(token_ids)} tokens) is within limit ({target_max_tokens}). No summarization needed.")
        return prompt
    else:
        logger.warning(f"Prompt length ({len(token_ids)} tokens) exceeds limit ({target_max_tokens}). Attempting summarization.")

        # 2. Summarize if too long
        try:
            # Load summarizer pipeline only if needed and not already loaded
            if summarizer_pipeline is None:
                logger.info("Loading summarization pipeline (facebook/bart-large-cnn)...")
                # Ensure CUDA device is managed properly if GPU is available
                device_id = 0 if torch.cuda.is_available() else -1
                summarizer_pipeline = hf_pipeline("summarization", model="facebook/bart-large-cnn", device=device_id)
                logger.info("Summarization pipeline loaded.")

            # Estimate target summary length in words (very rough approximation)
            # Aim for a summary significantly shorter than the max token limit
            # BART expects max_length in terms of tokens, not words.
            # Let's target roughly 60% of the max token length for the summary.
            summary_max_len = math.ceil(target_max_tokens * 0.7) # Target token length for summary
            summary_min_len = min(min_summary_tokens, summary_max_len - 5) # Ensure min is reasonable

            logger.debug(f"Summarizing with min_length={summary_min_len}, max_length={summary_max_len}")

            # Summarize the original *full* prompt
            summary_list = summarizer_pipeline(prompt, max_length=summary_max_len, min_length=summary_min_len, do_sample=False)

            if summary_list and isinstance(summary_list, list) and 'summary_text' in summary_list[0]:
                summary = summary_list[0]['summary_text'].strip()
                # Check summary length again (optional but good practice)
                summary_token_ids = tokenizer(summary, max_length=max_length, truncation=False)["input_ids"]
                logger.info(f"Summarized prompt: '{summary}' ({len(summary_token_ids)} tokens)")
                if len(summary_token_ids) > target_max_tokens:
                     logger.warning("Summarized prompt still exceeds token limit! Truncating summary.")
                     # Force truncation by the tokenizer during the actual pipeline call later
                     return summary # Return the summary, it will be truncated by CLIPTokenizer later
                return summary
            else:
                logger.error("Summarization failed to produce valid output.")
                # Fallback: Truncate the original prompt manually (less ideal)
                truncated_prompt = tokenizer.decode(token_ids[:target_max_tokens], skip_special_tokens=True)
                logger.warning(f"Falling back to manual truncation: '{truncated_prompt}'")
                return truncated_prompt

        except Exception as e:
            logger.error(f"Failed during summarization or length check: {e}\n{traceback.format_exc()}")
            # Fallback: Truncate the original prompt manually
            try:
                truncated_prompt = tokenizer.decode(token_ids[:target_max_tokens], skip_special_tokens=True)
                logger.warning(f"Falling back to manual truncation due to error: '{truncated_prompt}'")
                return truncated_prompt
            except Exception: # If even decoding fails
                logger.warning(f"Failed to summarize prompt")
                return prompt[:max_length*5] # Very rough character limit fallback


def generate_images(
    prior_pipeline, #: KandinskyV22PriorPipeline, # Kandinsky specific
    decoder_pipeline, #: StableDiffusionXLPipeline or KandinskyV22Pipeline, # Main pipeline
    refiner_pipeline, #: StableDiffusionXLImg2ImgPipeline, # SDXL specific
    model_name, #: str,
    prompts_df, #: pd.DataFrame,
    output_base_dir, #: str,
    checkpoint_label, #: str,
    gen_params_label, # Added label for generation hyperparams
    sample_ids, #: set,
    prompt_variations, #: dict
    gen_hyperparams # Added generation hyperparameters
    ):
    """
    Generates images for specified prompts and variations using the loaded pipeline(s).
    Includes SDXL refiner stage and quality improvements.

    Args:
        prior_pipeline: The loaded prior pipeline (used for Kandinsky). None otherwise.
        decoder_pipeline: The loaded main/decoder pipeline (SDXL Base or Kandinsky Decoder).
        refiner_pipeline: The loaded SDXL Refiner pipeline. None otherwise.
        model_name (str): Name of the model (e.g., 'sdxl', 'kandinsky').
        prompts_df (pd.DataFrame): DataFrame containing prompts and details ('file', 'prompt_details').
        output_base_dir (str): Base directory to save generated images.
        checkpoint_label (str): Label identifying the model config and checkpoint type.
        gen_params_label, # Added label for generation hyperparams
        sample_ids (set): A set of string file IDs for which to generate images.
        prompt_variations (dict): Dictionary where keys are variation names (e.g., 'cbt')
                                  and values are prompt template strings.
        gen_hyperparams (dict): Dictionary of generation hyperparameters (e.g., num_inference_steps, guidance_scale).
    """
    if decoder_pipeline is None: logger.error("Main/Decoder pipeline is None."); return
    if model_name == "kandinsky" and prior_pipeline is None: logger.error("Kandinsky requires prior_pipeline."); return
    # Refiner is optional for SDXL, but recommended
    if model_name == "sdxl" and refiner_pipeline is None: logger.warning("SDXL Refiner pipeline not provided, image quality may be suboptimal.")
    if not prompt_variations: logger.error("No prompt variations provided."); return

    if decoder_pipeline is None: logger.error("Main/Decoder pipeline is None."); return

    # --- Tokenizer for length check ---
    # Get the primary tokenizer (SDXL uses tokenizer, Kandinsky uses prior's)
    if model_name == "sdxl":
        tokenizer_for_check = decoder_pipeline.tokenizer # SDXL's first tokenizer
    elif model_name == "kandinsky":
        if prior_pipeline: tokenizer_for_check = prior_pipeline.tokenizer
        else: logger.error("Kandinsky prior pipeline missing for tokenizer check."); return
    else:
        # Fallback or add logic for other models if needed
        tokenizer_for_check = decoder_pipeline.tokenizer if hasattr(decoder_pipeline, 'tokenizer') else None

    if tokenizer_for_check is None:
        logger.error(f"Could not determine tokenizer for length check for model {model_name}"); return

    # --- Extract generation hyperparameters ---
    num_inference_steps = gen_hyperparams.get("num_inference_steps", 40) # Default 40
    guidance_scale = gen_hyperparams.get("guidance_scale", 7.5) # Default 7.5
    # SDXL Refiner specific params
    high_noise_frac = gen_hyperparams.get("high_noise_frac", 0.8) # Default 0.8 for refiner split
    num_refiner_steps = gen_hyperparams.get("num_refiner_steps", 25) # Default 25 for refiner

    logger.info(f"Generating images for {len(sample_ids)} specified IDs using {len(prompt_variations)} prompt variations.")
    img_size = 1024 if model_name == "sdxl" else 512
    device = decoder_pipeline.device

    # --- Quality prompt additions ---
    pos_quality_boost = ", sharp focus, highly detailed, intricate details, clear, high resolution, masterpiece, 8k"
    neg_quality_boost = "blurry, blurred, smudged, low quality, worst quality, unclear, fuzzy, out of focus, text, words, letters, signature, watermark, username, artist name, deformed, distorted"
    # ---

    if sample_ids:
        sample_id_list = list(sample_ids)
        logger.debug(f"Sample file IDs to generate (first 5): {sample_id_list[:5]}")
        if sample_id_list: logger.debug(f"Type of first sample file ID: {type(sample_id_list[0])}")
    else: logger.warning("Sample ID list is empty!")

    prompts_processed_count = 0
    for index, row in prompts_df.iterrows():
        file_id = str(row.get('file', f'row_{index}')).strip()
        logger.debug(f"Checking file ID: '{file_id}' (Type: {type(file_id)}) against sample_ids set.")
        if file_id not in sample_ids: logger.debug(f"Skipping file ID '{file_id}'."); continue

        prompts_processed_count += 1
        prompt_details = row.get('prompt_details', '')
        logger.info(f"Processing File ID: {file_id} (Row {index})")

        for variation_name, prompt_template in prompt_variations.items():
            logger.info(f"  Generating for variation: {variation_name}")
            # !--- Create output dir including gen_params label ---!
            variation_output_dir = os.path.join(output_base_dir, model_name, checkpoint_label, gen_params_label, variation_name)
            os.makedirs(variation_output_dir, exist_ok=True)

            try:
                base_final_prompt = prompt_template.format(txt_prompt=prompt_details)
                # Add quality boosters
                final_prompt = base_final_prompt + pos_quality_boost
                negative_prompt = neg_quality_boost # Start negative prompt with quality terms
            except Exception as fmt_err: logger.error(f"  Error formatting prompt: {fmt_err}"); continue
            
            # !--- Summarize the prompt BEFORE passing to pipeline ---!
            # Use the boosted prompt for summarization check
            final_prompt = summarize_long_prompt(
                final_prompt,
                tokenizer=tokenizer_for_check, # Pass the relevant tokenizer
                max_length=77 # CLIP's typical limit
            )
            # !-------------------------------------------------------!
            
            logger.debug(f"  Formatted Prompt w/ Boosters: {final_prompt}")
            logger.debug(f"  Negative Prompt w/ Boosters: {negative_prompt}")

            try:
                generator = torch.Generator(device=device).manual_seed(42 + index + hash(variation_name))
                image = None

                with torch.no_grad():
                    if model_name == "kandinsky":
                        # --- Kandinsky Stage 1: Prior ---

                        # Use hyperparameters for Kandinsky if specified differently, otherwise use defaults
                        k_prior_steps = gen_hyperparams.get("kandinsky_prior_steps", 25)
                        k_decoder_steps = gen_hyperparams.get("kandinsky_decoder_steps", 50)
                        k_guidance = gen_hyperparams.get("kandinsky_guidance", 4.0)

                        logger.debug(f"  Running Kandinsky Prior (Steps: {k_prior_steps})...")
                        prior_output = prior_pipeline(
                            prompt=final_prompt, # Use boosted prompt
                            negative_prompt=negative_prompt,
                            num_inference_steps=k_prior_steps, # Keep prior steps relatively low
                            generator=generator
                        )
                        image_embeds = prior_output.image_embeds
                        negative_image_embeds = prior_output.negative_image_embeds
                        logger.debug("  Finished Kandinsky Prior.")

                        # --- Kandinsky Stage 2: Decoder ---
                        logger.debug("  Running Kandinsky Decoder...")
                        image = decoder_pipeline(
                            prompt=final_prompt, # Pass boosted prompt again
                            image_embeds=image_embeds,
                            negative_image_embeds=negative_image_embeds,
                            height=img_size,
                            width=img_size,
                            num_inference_steps=k_decoder_steps, # Can increase decoder steps
                            guidance_scale=k_guidance,
                            generator=generator
                        ).images[0]
                        logger.debug("  Finished Kandinsky Decoder.")

                    elif model_name == "sdxl":
                        # --- SDXL Stage 1: Base ---
                        logger.debug(f"  Running SDXL Base (Steps: {num_inference_steps}, Guidance: {guidance_scale}, Denoising End: {high_noise_frac if refiner_pipeline else None})...")
                        # n_steps = 40 # Increased base steps
                        # high_noise_frac = 0.8 # Fraction of steps for base model when using refiner

                        latents = decoder_pipeline( # Use the main pipeline (decoder_pipeline var)
                            prompt=final_prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=num_inference_steps, 
                            guidance_scale=guidance_scale, # Slightly increased guidance
                            generator=generator,
                            # If using refiner, output latents and potentially denoise only partially
                            output_type="latent" if refiner_pipeline else "pil", # Output latents only if refiner exists
                            denoising_end=high_noise_frac if refiner_pipeline else None # Stop base early if using refiner
                        ).images # Output is latents or PIL image

                        # --- SDXL Stage 2: Refiner ---
                        if refiner_pipeline and isinstance(latents, torch.Tensor): # Check if we got latents
                            logger.debug(f"  SDXL Base finished, latent shape: {latents.shape}. Running Refiner...")
                            # Refiner needs the same prompt and the base latents
                            image = refiner_pipeline(
                                prompt=final_prompt,
                                negative_prompt=negative_prompt,
                                image=latents, # Pass base latents as 'image' input
                                num_inference_steps=num_refiner_steps, # Can use different steps for refiner # Use total steps here too
                                denoising_start=high_noise_frac, # Start refining from where base left off
                                guidance_scale=guidance_scale, # Can use same guidance
                                generator=generator,
                            ).images[0]
                            logger.debug("  SDXL Refiner finished.")
                        elif refiner_pipeline and not isinstance(latents, torch.Tensor):
                             logger.error("  SDXL Base pipeline did not return latents, cannot run refiner.")
                             image = None # Mark as failed
                        elif not refiner_pipeline and isinstance(latents, Image.Image):
                             # Base pipeline already produced an image (no refiner used)
                             logger.info("  SDXL Refiner not used, using base output directly.")
                             image = latents # The 'latents' variable actually holds the PIL image here
                        else:
                             logger.error("  Unexpected state after SDXL base pipeline. Cannot proceed.")
                             image = None


                # --- End Generation Logic ---

                if image:
                    output_filename = f"{model_name}_{checkpoint_label}_{gen_params_label}_{variation_name}_{file_id}.png"
                    output_path = os.path.join(variation_output_dir, output_filename)
                    image.save(output_path)
                    logger.info(f"  Saved image: {output_path}")
                else: logger.warning(f"  Image generation failed for {file_id}, variation {variation_name}")

            except Exception as e: logger.error(f"  Failed generation for file ID {file_id}, variation {variation_name}: {e}\n{traceback.format_exc()}")

            # Optional periodic cleanup
            if (prompts_processed_count * len(prompt_variations) + list(prompt_variations.keys()).index(variation_name)) % 5 == 0:
                 gc.collect(); torch.cuda.empty_cache()

    logger.info(f"Finished generating images. Processed {prompts_processed_count} matching file IDs from sample list.")


# --- main function ---
def main():
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    prompts_csv_path = "/home/iris/Documents/deep_learning/data/input_csv/FILE_SUPERTOPIC_DESCRIPTION.csv"
    sample_list_path = "/home/iris/Documents/deep_learning/data/sample_list.txt"
    prompt_variations_path = "/home/iris/Documents/deep_learning/config/prompt_config.yaml"
    generation_output_base = "/home/iris/Documents/deep_learning/generated_images/iter2"

    config = load_config(config_path)
    if config is None: return

    sample_ids = load_sample_ids(sample_list_path)
    if not sample_ids: logger.error("Sample ID list empty."); return

    prompt_variations = load_prompt_variations(prompt_variations_path)
    if not prompt_variations: logger.error("Prompt variations empty."); return

    if not os.path.exists(prompts_csv_path): logger.error(f"Prompts CSV not found: {prompts_csv_path}"); return
    try:
        prompts_df = pd.read_csv(prompts_csv_path)
        if 'file' not in prompts_df.columns: logger.error("CSV missing 'file' column."); return
        prompts_df['file'] = prompts_df['file'].astype(str)
        logger.info(f"Loaded {len(prompts_df)} total prompts from {prompts_csv_path}")
    except Exception as e: logger.error(f"Failed to load prompts CSV: {e}"); return

    base_output_dir = config.get("base_output_dir", "/home/iris/Documents/deep_learning/experiments/custom_finetuned")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_dtype = torch.float16

    base_model_ids = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "kandinsky_prior": "kandinsky-community/kandinsky-2-2-prior",
        "kandinsky_decoder": "kandinsky-community/kandinsky-2-2-decoder",
    }
    models_to_generate = config.get("models_to_generate", ["sdxl", "kandinsky"])

    # !--- Define Generation Hyperparameter Sets to Test ---!
    generation_param_sets = [
        # Set 0: Defaults / Previous settings
        {"guidance_scale": 7.5, "num_inference_steps": 40, "high_noise_frac": 0.8, "num_refiner_steps": 25, "kandinsky_guidance": 4.0, "kandinsky_decoder_steps": 50},
        # Set 1: Increased Steps
        {"guidance_scale": 7.5, "num_inference_steps": 50, "high_noise_frac": 0.8, "num_refiner_steps": 30, "kandinsky_guidance": 4.0, "kandinsky_decoder_steps": 60},
        # Set 2: Lower Guidance
        {"guidance_scale": 6.5, "num_inference_steps": 40, "high_noise_frac": 0.8, "num_refiner_steps": 25, "kandinsky_guidance": 3.5, "kandinsky_decoder_steps": 50},
        # Set 3: Higher Guidance
        {"guidance_scale": 8.5, "num_inference_steps": 40, "high_noise_frac": 0.8, "num_refiner_steps": 25, "kandinsky_guidance": 5.0, "kandinsky_decoder_steps": 50},
        # Set 4: Different Refiner Split
        {"guidance_scale": 7.5, "num_inference_steps": 40, "high_noise_frac": 0.7, "num_refiner_steps": 25, "kandinsky_guidance": 4.0, "kandinsky_decoder_steps": 50},
        # Add more sets as needed
    ]

    # --- Pre-load Refiner if needed ---
    refiner_pipeline = None
    if "sdxl" in models_to_generate:
        logger.info("Pre-loading SDXL Refiner pipeline...")
        try:
            refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                base_model_ids["sdxl_refiner"],
                torch_dtype=target_dtype, use_safetensors=True, variant="fp16"
            ).to(device)
            logger.info("SDXL Refiner loaded successfully.")
        except Exception as e: logger.error(f"Failed to load SDXL Refiner: {e}.")

    for model_name in models_to_generate:
        model_finetune_base = os.path.join(base_output_dir, model_name)
        logger.info(f"Processing model: {model_name} (Fine-tune base: {model_finetune_base})")
        if not os.path.isdir(model_finetune_base): logger.warning(f"Dir not found: {model_finetune_base}."); continue

        current_base_ids = {}
        if model_name == "sdxl": current_base_ids["sdxl"] = base_model_ids["sdxl"]
        elif model_name == "kandinsky":
            current_base_ids["kandinsky_prior"] = base_model_ids["kandinsky_prior"]
            current_base_ids["kandinsky_decoder"] = base_model_ids["kandinsky_decoder"]
        else: logger.warning(f"Base model IDs not defined for {model_name}."); continue

        for config_dir_name in os.listdir(model_finetune_base):
            if config_dir_name.startswith("hyperparam_config_"):
                config_dir_path = os.path.join(model_finetune_base, config_dir_name)
                logger.info(f"--- Processing Config Directory: {config_dir_name} ---")
                checkpoints = find_checkpoint_dirs(config_dir_path)

                for save_type, checkpoint_dir in checkpoints.items():
                    if checkpoint_dir:
                        logger.info(f"--- Generating images for '{save_type}' checkpoint ---")
                        # !--- Loop through Generation Hyperparameters ---!
                        for gen_idx, gen_params in enumerate(generation_param_sets):
                            gen_params_label = f"gen_params_{gen_idx}"
                            logger.info(f"--- Running Generation with Params Set {gen_idx}: {gen_params} ---")

                            pipeline, prior_pipeline_k = None, None
                            try:
                                pipeline, prior_pipeline_k = load_pipeline_with_lora(
                                    model_name=model_name,
                                    base_model_ids=current_base_ids,
                                    lora_checkpoint_dir=checkpoint_dir,
                                    device=device,
                                    target_dtype=target_dtype
                                )
                                current_refiner = refiner_pipeline if model_name == "sdxl" else None
                                if pipeline or prior_pipeline_k:
                                    generate_images(
                                        prior_pipeline=prior_pipeline_k,
                                        decoder_pipeline=pipeline,
                                        refiner_pipeline=current_refiner, # Pass refiner
                                        model_name=model_name,
                                        prompts_df=prompts_df,
                                        output_base_dir=generation_output_base,
                                        # Include gen_params_label in checkpoint label for output path
                                        checkpoint_label=f"{config_dir_name}_{save_type}",
                                        gen_params_label=gen_params_label, # Pass gen param label
                                        sample_ids=sample_ids,
                                        prompt_variations=prompt_variations,
                                        gen_hyperparams=gen_params # Pass the hyperparams dict
                                    )
                                else:
                                    logger.error(f"Skipping generation for {save_type}/{gen_params_label} due to pipeline loading failure.")
                            except Exception as e:
                                logger.error(f"Failed generation pipeline for {model_name}/{config_dir_name}/{save_type}/{gen_params_label}: {e}\n{traceback.format_exc()}")
                            finally:
                                logger.debug(f"Cleaning up after {save_type}/{gen_params_label}...")
                                del pipeline; del prior_pipeline_k
                                gc.collect(); torch.cuda.empty_cache()
                        # --- End Generation Hyperparameter Loop ---
                    else:
                        logger.warning(f"No '{save_type}' checkpoint found for {config_dir_name}. Skipping.")
                logger.info(f"--- Finished processing Config Directory: {config_dir_name} ---")
        logger.info(f"========== Finished processing model: {model_name} ==========")

    logger.info("Image generation script finished.")

    # Cleanup refiner at the very end
    del refiner_pipeline
    gc.collect(); torch.cuda.empty_cache()
    logger.info("Image generation script finished.")


if __name__ == "__main__":
    main()



# --- generate_images function (remains the same) ---
# def generate_images(prior_pipeline, decoder_pipeline, model_name, prompts_df, output_base_dir, checkpoint_label, sample_ids, prompt_variations):
#     """Generates images for specified prompts and variations using the loaded pipeline(s)."""
#     # ... (Implementation from previous step - no changes needed here) ...
#     if decoder_pipeline is None: logger.error("Main/Decoder pipeline is None."); return
#     if model_name == "kandinsky" and prior_pipeline is None: logger.error("Kandinsky requires prior_pipeline."); return
#     if not prompt_variations: logger.error("No prompt variations provided."); return

#     logger.info(f"Generating images for {len(sample_ids)} specified IDs using {len(prompt_variations)} prompt variations.")
#     img_size = 1024 if model_name == "sdxl" else 512
#     device = decoder_pipeline.device

#     if sample_ids:
#         sample_id_list = list(sample_ids)
#         logger.debug(f"Sample file IDs to generate (first 5): {sample_id_list[:5]}")
#         if sample_id_list: logger.debug(f"Type of first sample file ID: {type(sample_id_list[0])}")
#     else: logger.warning("Sample ID list is empty!")

#     prompts_processed_count = 0
#     for index, row in prompts_df.iterrows():
#         file_id = str(row.get('file', f'row_{index}')).strip()
#         logger.debug(f"Checking file ID: '{file_id}' (Type: {type(file_id)}) against sample_ids set.")
#         if file_id not in sample_ids: logger.debug(f"Skipping file ID '{file_id}'."); continue

#         prompts_processed_count += 1
#         prompt_details = row.get('prompt_details', '')
#         logger.info(f"Processing File ID: {file_id} (Row {index})")

#         for variation_name, prompt_template in prompt_variations.items():
#             logger.info(f"  Generating for variation: {variation_name}")
#             variation_output_dir = os.path.join(output_base_dir, model_name, checkpoint_label, variation_name)
#             os.makedirs(variation_output_dir, exist_ok=True)

#             try: final_prompt = prompt_template.format(txt_prompt=prompt_details)
#             except Exception as fmt_err: logger.error(f"  Error formatting prompt: {fmt_err}"); continue
#             logger.debug(f"  Formatted Prompt: {final_prompt}")

#             try:
#                 generator = torch.Generator(device=device).manual_seed(42 + index + hash(variation_name))
#                 negative_prompt = "low quality, bad quality, blurry, text, words, letters, signature"
#                 image = None
#                 with torch.no_grad():
#                     if model_name == "kandinsky":
#                         logger.debug("  Running Kandinsky Prior...")
#                         prior_output = prior_pipeline(prompt=final_prompt, negative_prompt=negative_prompt, num_inference_steps=25, generator=generator)
#                         image_embeds = prior_output.image_embeds; negative_image_embeds = prior_output.negative_image_embeds
#                         logger.debug("  Running Kandinsky Decoder...")
#                         image = decoder_pipeline(prompt=final_prompt, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=img_size, width=img_size, num_inference_steps=50, guidance_scale=4.0, generator=generator).images[0]
#                     elif model_name == "sdxl":
#                         image = decoder_pipeline(
#                             prompt=final_prompt, negative_prompt=negative_prompt, height=img_size, width=img_size,
#                             num_inference_steps=30, guidance_scale=7.5, generator=generator, num_images_per_prompt=1
#                         ).images[0]

#                 if image:
#                     output_filename = f"{model_name}_{checkpoint_label}_{variation_name}_{file_id}.png"
#                     output_path = os.path.join(variation_output_dir, output_filename)
#                     image.save(output_path)
#                     logger.info(f"  Saved image: {output_path}")
#                 else: logger.warning(f"  Image generation failed for {file_id}, variation {variation_name}")

#             except Exception as e: logger.error(f"  Failed generation for file ID {file_id}, variation {variation_name}: {e}\n{traceback.format_exc()}")

#             if (prompts_processed_count * len(prompt_variations) + list(prompt_variations.keys()).index(variation_name)) % 5 == 0:
#                  gc.collect(); torch.cuda.empty_cache()

#     logger.info(f"Finished generating images. Processed {prompts_processed_count} matching file IDs from sample list.")