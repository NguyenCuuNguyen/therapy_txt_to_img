import os
import torch
import logging
import yaml
import json
import pandas as pd
import gc
import re
import traceback
from pathlib import Path
from PIL import Image
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel
)
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
import math
from peft import PeftModel

# Import the custom pipeline from the training script
from train_sdxl_base_refiner import StableDiffusionXLPipelineWithT5

# Attempt to import OpenAIUtils, with a dummy fallback
try:
    from src.utils.openai_utils import OpenAIUtils
except ImportError:
    logging.error("Could not import OpenAIUtils. Using dummy class.")
    class OpenAIUtils:
        def __init__(self, *args, **kwargs): pass
        def summarize(self, text, target_max_tokens, tokenizer):
            logging.warning("OpenAIUtils not found, falling back to truncation.")
            token_ids = tokenizer(text, max_length=target_max_tokens, truncation=True, return_tensors="pt")["input_ids"]
            return tokenizer.decode(token_ids[0], skip_special_tokens=True)

# --- Explicit Logging Configuration ---
log_file_path = "/home/iris/Documents/deep_learning/src/logs/img_generate_sdxl_t5_refiner.log"
log_level = logging.INFO

logger = logging.getLogger()
logger.setLevel(log_level)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)
    handler.close()

try:
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")

    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    print(f"Logging to file: {log_file_path}")
except Exception as e:
    print(f"ERROR: Failed to configure file logging to {log_file_path}: {e}")

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# --- Configuration ---
def load_config(config_path):
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
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            prompt_list = config.get('theory_exp', {}).get('prompts', [])
            variations = {k: v for item in prompt_list for k, v in item.items()}
            if not variations:
                logger.warning(f"No prompt variations found in {yaml_path}")
            else:
                logger.info(f"Loaded {len(variations)} prompt variations from {yaml_path}")
            return variations
    except FileNotFoundError:
        logger.error(f"Prompt variations file not found: {yaml_path}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load prompt variations from {yaml_path}: {e}")
        return {}

def load_sample_ids(txt_path):
    try:
        with open(txt_path, 'r') as f:
            ids = {str(line.strip()) for line in f if line.strip()}
            logger.info(f"Loaded {len(ids)} sample IDs from {txt_path}")
            return ids
    except FileNotFoundError:
        logger.error(f"Sample ID list file not found: {txt_path}")
        return set()
    except Exception as e:
        logger.error(f"Failed to load sample IDs from {txt_path}: {e}")
        return set()

def find_checkpoint_dirs(base_model_dir):
    checkpoints = {"final": None}
    if not os.path.isdir(base_model_dir):
        logger.warning(f"Base directory not found: {base_model_dir}")
        return checkpoints
    for dirname in os.listdir(base_model_dir):
        dirpath = os.path.join(base_model_dir, dirname)
        if os.path.isdir(dirpath) and dirname.startswith("final_model"):
            checkpoints["final"] = dirpath
            logger.info(f"Found final checkpoint: {dirpath}")
            break
    if not checkpoints["final"]:
        logger.warning(f"No 'final_model' checkpoint directory found in {base_model_dir}")
    return checkpoints

def load_pipeline_with_lora(model_name, base_model_ids, lora_checkpoint_dir, device, target_dtype):
    logger.info(f"Loading base and refiner model components for {model_name}...")
    pipeline = None

    try:
        if model_name == "sdxl_base_refiner":
            t5_model_name = "t5-base"
            base_model_id = base_model_ids["sdxl"]
            refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

            logger.info("Loading base components...")
            tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            gc.collect()
            torch.cuda.empty_cache()

            text_encoder = T5EncoderModel.from_pretrained(t5_model_name)
            gc.collect()
            torch.cuda.empty_cache()

            vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
            gc.collect()
            torch.cuda.empty_cache()

            unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet")
            gc.collect()
            torch.cuda.empty_cache()

            refiner_unet = UNet2DConditionModel.from_pretrained(refiner_model_id, subfolder="unet")
            gc.collect()
            torch.cuda.empty_cache()

            scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model_id, subfolder="scheduler")
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"Checking for projection layer files in {lora_checkpoint_dir}...")
            projection_layer_files = [f for f in os.listdir(lora_checkpoint_dir) if f.endswith("_projection_layer.pth")]
            pool_projection_layer_files = [f for f in os.listdir(lora_checkpoint_dir) if f.endswith("_pool_projection_layer.pth")]

            projection_layer = None
            pool_projection_layer = None

            if projection_layer_files:
                projection_layer_path = os.path.join(lora_checkpoint_dir, projection_layer_files[0])
                logger.info(f"Loading projection layer from {projection_layer_path}")
                projection_layer = torch.nn.Linear(768, 2048)
                projection_layer.load_state_dict(torch.load(projection_layer_path))
                projection_layer.to(device, dtype=target_dtype)
                logger.info("Loaded projection layer.")
            else:
                logger.info("No projection layer file found; using default initialization.")
                projection_layer = torch.nn.Linear(768, 2048).to(device, dtype=target_dtype)

            if pool_projection_layer_files:
                pool_projection_layer_path = os.path.join(lora_checkpoint_dir, pool_projection_layer_files[0])
                logger.info(f"Loading pool projection layer from {pool_projection_layer_path}")
                pool_projection_layer = torch.nn.Linear(768, 1280)
                pool_projection_layer.load_state_dict(torch.load(pool_projection_layer_path))
                pool_projection_layer.to(device, dtype=target_dtype)
                logger.info("Loaded pool projection layer.")
            else:
                logger.info("No pool projection layer file found; using default initialization.")
                pool_projection_layer = torch.nn.Linear(768, 1280).to(device, dtype=target_dtype)

            logger.info("Initializing SDXL T5 pipeline with base and refiner UNets...")
            pipeline = StableDiffusionXLPipelineWithT5(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                projection_layer=projection_layer,
                pool_projection_layer=pool_projection_layer,
                refiner_unet=refiner_unet,
                logger_instance=logger
            )

            # Clear memory before loading LoRA weights
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"Searching for LoRA weight directories in: {lora_checkpoint_dir}")
            unet_lora_dirs = [d for d in os.listdir(lora_checkpoint_dir) if os.path.isdir(os.path.join(lora_checkpoint_dir, d)) and d.endswith("_unet_lora")]
            refiner_unet_lora_dirs = [d for d in os.listdir(lora_checkpoint_dir) if os.path.isdir(os.path.join(lora_checkpoint_dir, d)) and d.endswith("_refiner_unet_lora")]

            if unet_lora_dirs:
                unet_lora_path = os.path.join(lora_checkpoint_dir, unet_lora_dirs[0])
                logger.info(f"Found base UNet LoRA directory: {unet_lora_path}")
                logger.info(f"Loading base UNet LoRA weights using PeftModel...")
                pipeline.unet = PeftModel.from_pretrained(pipeline.unet, unet_lora_path)
                logger.info("Loaded base UNet LoRA weights.")
            else:
                logger.warning(f"No base UNet LoRA directory found in {lora_checkpoint_dir}")

            if refiner_unet_lora_dirs:
                refiner_unet_lora_path = os.path.join(lora_checkpoint_dir, refiner_unet_lora_dirs[0])
                logger.info(f"Found refiner UNet LoRA directory: {refiner_unet_lora_path}")
                logger.info(f"Loading refiner UNet LoRA weights using PeftModel...")
                pipeline.refiner_unet = PeftModel.from_pretrained(pipeline.refiner_unet, refiner_unet_lora_path)
                logger.info("Loaded refiner UNet LoRA weights.")
            else:
                logger.warning(f"No refiner UNet LoRA directory found in {lora_checkpoint_dir}")

            # Move pipeline to GPU
            pipeline.to(device)
            logger.info("Pipeline moved to GPU.")

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        return pipeline

    except Exception as e:
        logger.error(f"Failed during model or LoRA loading for {model_name} from {lora_checkpoint_dir}: {e}\n{traceback.format_exc()}")
        return None

def summarize_long_prompt(prompt: str, tokenizer, max_length: int = 512):
    logger.info(f"Prompt: {prompt}")
    if not prompt or not isinstance(prompt, str):
        logger.warning("Invalid prompt passed to summarizer.")
        return ""

    target_max_tokens = max_length - 2
    token_ids = tokenizer(prompt, max_length=max_length, truncation=False)["input_ids"]

    if len(token_ids) <= target_max_tokens:
        logger.debug(f"Prompt length ({len(token_ids)} tokens) is within limit ({target_max_tokens}).")
        return prompt
    else:
        logger.warning(f"Prompt length ({len(token_ids)}) exceeds limit ({target_max_tokens}). Using manual truncation.")
        truncated_prompt = tokenizer.decode(token_ids[:target_max_tokens], skip_special_tokens=True)
        logger.info(f"Truncated prompt: '{truncated_prompt}'")
        return truncated_prompt

def generate_images(
    pipeline,
    model_name,
    prompts_df,
    output_base_dir,
    checkpoint_label,
    gen_params_label,
    sample_ids,
    prompt_variations,
    gen_hyperparams,
    openai_utils
):
    if pipeline is None:
        logger.error("Pipeline is None; cannot generate images.")
        return
    if not prompt_variations:
        logger.error("No prompt variations provided.")
        return
    if openai_utils is None:
        logger.error("OpenAIUtils instance not provided.")
        return

    logger.info(f"Generating images for {len(sample_ids)} IDs using {len(prompt_variations)} variations with params: {gen_params_label}")
    img_size = 1024
    device = pipeline.device

    num_inference_steps = gen_hyperparams.get("num_inference_steps", 30)
    guidance_scale = gen_hyperparams.get("guidance_scale", 7.5)
    refiner_steps = gen_hyperparams.get("refiner_steps", 10)  # Number of refiner steps

    pos_quality_boost = ", sharp focus, highly detailed, intricate details, clear, high resolution, masterpiece, 8k"
    neg_quality_boost = "blurry, blurred, smudged, low quality, worst quality, unclear, fuzzy, out of focus, text, words, letters, signature, watermark, username, artist name, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limbs, missing limbs"

    tokenizer_for_check = pipeline.tokenizer
    t5_max_length = tokenizer_for_check.model_max_length

    prompts_processed_count = 0
    all_columns = prompts_df.columns.tolist()
    topic_columns = [col for col in all_columns if col not in ['file', 'id', 'prompt_details']]
    logger.info(f"Using columns for topic dictionary: {topic_columns}")
    if not topic_columns:
        logger.error("No topic columns identified!")
        return

    for index, row in prompts_df.iterrows():
        file_id = str(row.get('file', f'row_{index}')).strip()
        if file_id not in sample_ids:
            continue

        prompts_processed_count += 1
        try:
            topic_dict = row[topic_columns].dropna().to_dict()
            topic_string = ", ".join([f"'{k}': '{str(v).strip()}'" for k, v in topic_dict.items() if isinstance(v, str) and str(v).strip()])
            topic_string_formatted = f"{{{topic_string}}}" if topic_string else "{}"
        except Exception as e:
            logger.error(f"Error creating topic string for {file_id}: {e}")
            continue
        logger.info(f"Processing File ID: {file_id} (Row {index})")
        logger.debug(f"Topic String: {topic_string_formatted[:200]}...")

        parent_output_dir = os.path.join(output_base_dir, file_id)
        os.makedirs(parent_output_dir, exist_ok=True)

        for variation_name, prompt_template in prompt_variations.items():
            logger.info(f"Generating for variation: {variation_name}")
            try:
                base_final_prompt = prompt_template.format(txt_prompt=topic_string_formatted)
                final_prompt_boosted = base_final_prompt + pos_quality_boost
                negative_prompt_boosted = neg_quality_boost
            except KeyError:
                logger.error(f"Template missing '{{txt_prompt}}'.")
                continue
            except Exception as fmt_err:
                logger.error(f"Error formatting prompt: {fmt_err}")
                continue

            prompt_to_generate = summarize_long_prompt(final_prompt_boosted, tokenizer_for_check, t5_max_length)

            logger.debug(f"Prompt for generation: {prompt_to_generate}")
            logger.debug(f"Negative Prompt: {negative_prompt_boosted}")

            try:
                generator = torch.Generator(device=device).manual_seed(42 + index + hash(variation_name))
                image = None
                with torch.no_grad():
                    logger.info(f"Running base UNet for {num_inference_steps} steps and refiner UNet for {refiner_steps} steps")
                    image = pipeline(
                        prompt=prompt_to_generate,
                        negative_prompt=negative_prompt_boosted,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        refiner_steps=refiner_steps,
                        height=img_size,
                        width=img_size,
                        generator=generator,
                        output_type="pil"
                    ).images[0]

                if image:
                    output_filename = f"{file_id}_{variation_name}.png"
                    output_path = os.path.join(parent_output_dir, output_filename)
                    image.save(output_path)
                    logger.info(f"Saved image: {output_path}")
                else:
                    logger.warning(f"Image generation failed for {file_id}, variation {variation_name}")

            except Exception as e:
                logger.error(f"Failed generation for file ID {file_id}, variation {variation_name}: {e}\n{traceback.format_exc()}")

            # Clear memory after each image generation
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"Finished generating images. Processed {prompts_processed_count} matching file IDs.")

def main():
    # Set PyTorch CUDA allocation configuration to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Clear GPU memory at the start
    gc.collect()
    torch.cuda.empty_cache()

    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    prompts_csv_path = "/home/iris/Documents/deep_learning/data/input_csv/FILE_SUPERTOPIC_DESCRIPTION.csv"
    sample_list_path = "/home/iris/Documents/deep_learning/data/sample_list.txt"
    prompt_variations_path = "/home/iris/Documents/deep_learning/config/prompt_config.yaml"
    generation_output_base = "/home/iris/Documents/deep_learning/generated_images/iter2/sdxl_base_refiner"

    config = load_config(config_path)
    if config is None:
        return

    openai_config = config.get("openai", {})
    openai_api_key = openai_config.get("api_key")
    if not openai_api_key:
        logger.error("OpenAI API key not found in config.yaml under 'openai_api_key'.")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OpenAI API key also not found in environment variable OPENAI_API_KEY.")
            openai_utils = None
        else:
            logger.info("Loaded OpenAI API key from environment variable.")
            openai_utils = OpenAIUtils(api_key=openai_api_key, logger=logger)
    else:
        logger.info("Loaded OpenAI API key from config file.")
        openai_utils = OpenAIUtils(api_key=openai_api_key, logger=logger)

    sample_ids = load_sample_ids(sample_list_path)
    if not sample_ids:
        logger.error("Sample ID list empty.")
        return

    prompt_variations = load_prompt_variations(prompt_variations_path)
    if not prompt_variations:
        logger.error("Prompt variations empty.")
        return

    if not os.path.exists(prompts_csv_path):
        logger.error(f"Prompts CSV not found: {prompts_csv_path}")
        return
    try:
        prompts_df = pd.read_csv(prompts_csv_path)
        if 'file' not in prompts_df.columns:
            logger.error("CSV missing 'file' column.")
            return
        prompts_df['file'] = prompts_df['file'].astype(str)
        logger.info(f"Loaded {len(prompts_df)} total prompts from {prompts_csv_path}")
    except Exception as e:
        logger.error(f"Failed to load prompts CSV: {e}")
        return

    base_output_dir = "/home/iris/Documents/deep_learning/experiments/trained_sdxl_t5_refiner"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_dtype = torch.float16

    base_model_ids = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    }
    models_to_generate = ["sdxl_base_refiner"]

    generation_param_sets = [
        {
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "refiner_steps": 10  # Added refiner steps
        },
    ]

    for model_name in models_to_generate:
        model_finetune_base = os.path.join(base_output_dir, "hyperparam_config_0")
        logger.info(f"Processing model: {model_name} (Fine-tune base: {model_finetune_base})")
        if not os.path.isdir(model_finetune_base):
            logger.warning(f"Dir not found: {model_finetune_base}.")
            continue

        logger.info(f"Processing Config Directory: hyperparam_config_0")
        checkpoints = find_checkpoint_dirs(model_finetune_base)

        for save_type, checkpoint_dir in checkpoints.items():
            if checkpoint_dir:
                logger.info(f"Generating images for '{save_type}' checkpoint")
                for gen_idx, gen_params in enumerate(generation_param_sets):
                    gen_params_label = f"gen_params_{gen_idx}"
                    logger.info(f"Running Generation with Params Set {gen_idx}: {gen_params}")

                    pipeline = None
                    try:
                        pipeline = load_pipeline_with_lora(
                            model_name=model_name,
                            base_model_ids=base_model_ids,
                            lora_checkpoint_dir=checkpoint_dir,
                            device=device,
                            target_dtype=target_dtype
                        )
                        if pipeline:
                            generate_images(
                                pipeline=pipeline,
                                model_name=model_name,
                                prompts_df=prompts_df,
                                output_base_dir=generation_output_base,
                                checkpoint_label=f"hyperparam_config_0_{save_type}",
                                gen_params_label=gen_params_label,
                                sample_ids=sample_ids,
                                prompt_variations=prompt_variations,
                                gen_hyperparams=gen_params,
                                openai_utils=openai_utils
                            )
                        else:
                            logger.error(f"Skipping generation for {save_type}/{gen_params_label} due to pipeline loading failure.")
                    except Exception as e:
                        logger.error(f"Failed generation pipeline for {model_name}/hyperparam_config_0/{save_type}/{gen_params_label}: {e}\n{traceback.format_exc()}")
                    finally:
                        logger.debug(f"Cleaning up after {save_type}/{gen_params_label}...")
                        del pipeline
                        gc.collect()
                        torch.cuda.empty_cache()
            else:
                logger.warning(f"No '{save_type}' checkpoint found for hyperparam_config_0. Skipping.")
        logger.info(f"Finished processing model: {model_name}")

    logger.info("Image generation script finished.")

if __name__ == "__main__":
    main()