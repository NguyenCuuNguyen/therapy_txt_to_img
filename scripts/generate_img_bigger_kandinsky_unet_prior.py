import os
import torch
import logging
import yaml
import json
import pandas as pd
import gc
import traceback
from pathlib import Path
from PIL import Image
from src.utils.openai_utils import OpenAIUtils
from diffusers import (
    KandinskyV22PriorPipeline,
    KandinskyV22Pipeline,
    VQModel,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel
)
from transformers import T5EncoderModel, T5Tokenizer
from peft import PeftModel

# --- Logging Configuration ---
log_file_path = "/home/iris/Documents/deep_learning/src/logs/iter2_bigger_kandinsky_image_generation_unet_prior.log"
log_level = logging.INFO

logger = logging.getLogger()
logger.setLevel(log_level)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)
    handler.close()

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

console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

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
    """Loads a list of IDs from a text file (one ID per line)."""
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

def load_pipeline_with_lora(model_name, base_model_ids, lora_checkpoint_dir, device, target_dtype):
    """Loads the Kandinsky pipeline with fine-tuned T5 encoder and UNet LoRA weights."""
    logger.info(f"Loading fine-tuned Kandinsky model from {lora_checkpoint_dir}...")
    pipeline = None
    prior_pipeline = None

    try:
        if model_name != "kandinsky":
            raise ValueError(f"Only 'kandinsky' model is supported, got {model_name}")

        prior_id = base_model_ids["kandinsky_prior"]
        decoder_id = base_model_ids["kandinsky_decoder"]

        # Load T5-base text encoder
        logger.info("Loading T5-base text encoder (google/flan-t5-base)...")
        t5_model_name = "google/flan-t5-base"
        t5_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        text_encoder = T5EncoderModel.from_pretrained(t5_model_name, torch_dtype=t5_dtype).to(device)
        text_encoder.eval()
        tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        logger.info(f"Loaded T5-base text encoder on {device} with dtype {t5_dtype}")

        # Load text encoder LoRA weights
        text_encoder_lora_path = os.path.join(lora_checkpoint_dir, "best_unet_lora")
        if not os.path.exists(text_encoder_lora_path):
            raise FileNotFoundError(f"Text encoder LoRA weights not found at {text_encoder_lora_path}")
        text_encoder = PeftModel.from_pretrained(text_encoder, text_encoder_lora_path, torch_dtype=t5_dtype).to(device)
        text_encoder.eval()
        logger.info(f"Loaded text encoder LoRA weights from {text_encoder_lora_path}")

        # Load Kandinsky prior pipeline
        logger.info(f"Loading Kandinsky Prior Pipeline ({prior_id})...")
        prior_pipeline = KandinskyV22PriorPipeline.from_pretrained(
            prior_id,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=target_dtype
        ).to(device)
        logger.info("Loaded Kandinsky prior pipeline with fine-tuned T5 encoder")

        # Load Kandinsky decoder pipeline
        logger.info(f"Loading Kandinsky Decoder Pipeline ({decoder_id})...")
        pipeline = KandinskyV22Pipeline.from_pretrained(
            decoder_id,
            torch_dtype=target_dtype,
            use_safetensors=True
        )

        # Load VAE in target_dtype
        logger.info(f"Loading Kandinsky VAE (MoVQ) in {target_dtype} from {decoder_id}/movq")
        vae = VQModel.from_pretrained(decoder_id, subfolder="movq", torch_dtype=target_dtype).to(device)
        pipeline.movq = vae
        logger.info(f"Loaded and assigned Kandinsky VQ VAE ({pipeline.movq.dtype})")

        # Load LoRA weights for UNet using PeftModel
        logger.info(f"Loading UNet LoRA weights from {lora_checkpoint_dir}/best_unet_lora")
        unet_lora_path = os.path.join(lora_checkpoint_dir, "best_unet_lora")
        if not os.path.exists(unet_lora_path):
            raise FileNotFoundError(f"UNet LoRA weights not found at {unet_lora_path}")
        
        unet = UNet2DConditionModel.from_pretrained(
            decoder_id,
            subfolder="unet",
            torch_dtype=target_dtype,
            use_safetensors=True
        ).to(device)
        unet = PeftModel.from_pretrained(unet, unet_lora_path, torch_dtype=target_dtype).to(device)
        pipeline.unet = unet
        logger.info("Loaded UNet LoRA weights using PeftModel")

        # Move pipeline to device
        pipeline.to(device)
        logger.info(f"Finished loading fine-tuned Kandinsky pipeline")
        return pipeline, prior_pipeline

    except Exception as e:
        logger.error(f"Failed to load Kandinsky pipeline: {e}\n{traceback.format_exc()}")
        return None, None

def generate_images(
    prior_pipeline,  # KandinskyV22PriorPipeline
    decoder_pipeline,  # KandinskyV22Pipeline
    refiner_pipeline,  # None for Kandinsky
    model_name,  # str
    prompts_df,  # pd.DataFrame
    output_base_dir,  # str
    checkpoint_label,  # str
    gen_params_label,  # str
    sample_ids,  # set
    prompt_variations,  # dict
    gen_hyperparams,  # dict
    openai_utils  # OpenAIUtils
):
    """Generates images using the fine-tuned Kandinsky model with GPT-4o-mini summarization."""
    if decoder_pipeline is None:
        logger.error("Decoder pipeline is None.")
        return
    if model_name != "kandinsky":
        logger.error(f"Only 'kandinsky' model is supported, got {model_name}")
        return
    if prior_pipeline is None:
        logger.error("Kandinsky requires prior_pipeline.")
        return
    if not prompt_variations:
        logger.error("No prompt variations provided.")
        return
    if openai_utils is None:
        logger.error("OpenAIUtils instance not provided.")
        return

    logger.info(f"Generating images for {len(sample_ids)} IDs using {len(prompt_variations)} variations with params: {gen_params_label}")
    img_size = 512  # Kandinsky uses 512x512
    device = decoder_pipeline.device

    # Extract generation hyperparameters
    k_prior_steps = gen_hyperparams.get("kandinsky_prior_steps", 25)
    k_decoder_steps = gen_hyperparams.get("kandinsky_decoder_steps", 50)
    k_guidance = gen_hyperparams.get("kandinsky_guidance", 4.0)

    # Quality prompt additions
    pos_quality_boost = ", sharp focus, highly detailed, intricate details, clear, high resolution, masterpiece, 8k"
    neg_quality_boost = "blurry, blurred, smudged, low quality, worst quality, unclear, fuzzy, out of focus, text, words, letters, signature, watermark, username, artist name, deformed, distorted, disfigured, poorly drawn, bad anatomy, extra limbs, missing limbs"

    # Tokenizer for length check (T5-base)
    tokenizer_for_check = prior_pipeline.tokenizer
    t5_max_length = tokenizer_for_check.model_max_length  # 512 for T5-base

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
        logger.debug(f"  Topic String: {topic_string_formatted[:200]}...")

        for variation_name, prompt_template in prompt_variations.items():
            logger.info(f"  Generating for variation: {variation_name}")
            variation_output_dir = os.path.join(output_base_dir, model_name, checkpoint_label, gen_params_label, variation_name)
            os.makedirs(variation_output_dir, exist_ok=True)

            try:
                base_final_prompt = prompt_template.format(txt_prompt=topic_string_formatted)
                final_prompt_boosted = base_final_prompt + pos_quality_boost
                negative_prompt_boosted = neg_quality_boost
            except KeyError:
                logger.error(f"  Template missing '{{txt_prompt}}'.")
                continue
            except Exception as fmt_err:
                logger.error(f"  Error formatting prompt: {fmt_err}")
                continue

            # Check length and summarize with GPT-4o-mini if needed
            target_max_tokens = t5_max_length - 2
            token_ids = tokenizer_for_check(final_prompt_boosted, max_length=t5_max_length, truncation=False)["input_ids"]

            if len(token_ids) <= target_max_tokens:
                prompt_to_generate = final_prompt_boosted
                logger.debug(f"Prompt within T5-base length limit ({len(token_ids)}/{target_max_tokens} tokens).")
            else:
                logger.warning(f"Prompt length ({len(token_ids)}) exceeds T5-base limit ({target_max_tokens}). Using GPT-4o-mini to summarize.")
                try:
                    prompt_to_generate = openai_utils.summarize_text_with_openai(
                        text_to_summarize=final_prompt_boosted,
                        target_max_tokens=target_max_tokens,
                        tokenizer=tokenizer_for_check,
                        model="gpt-4o-mini"
                    )
                    token_count = len(tokenizer_for_check(prompt_to_generate, max_length=t5_max_length, truncation=False)["input_ids"])
                    logger.info(f"  GPT-4o-mini summary: '{prompt_to_generate[:100]}...' ({token_count} tokens)")
                except Exception as e:
                    logger.error(f"Failed to summarize prompt with GPT-4o-mini: {e}\n{traceback.format_exc()}")
                    # Fallback: Truncate prompt manually
                    prompt_to_generate = tokenizer_for_check.decode(
                        token_ids[:target_max_tokens],
                        skip_special_tokens=True
                    )
                    token_count = len(tokenizer_for_check(prompt_to_generate, max_length=t5_max_length, truncation=False)["input_ids"])
                    logger.warning(f"Falling back to truncated prompt: '{prompt_to_generate[:100]}...' ({token_count} tokens)")

            logger.debug(f"  Prompt for generation: {prompt_to_generate[:200]}...")
            logger.debug(f"  Negative Prompt: {negative_prompt_boosted[:200]}...")

            try:
                generator = torch.Generator(device=device).manual_seed(42 + index + hash(variation_name))
                with torch.no_grad():
                    prior_output = prior_pipeline(
                        prompt=prompt_to_generate,
                        negative_prompt=negative_prompt_boosted,
                        num_inference_steps=k_prior_steps,
                        generator=generator
                    )
                    image_embeds = prior_output.image_embeds
                    negative_image_embeds = prior_output.negative_image_embeds
                    image = decoder_pipeline(
                        prompt=prompt_to_generate,
                        image_embeds=image_embeds,
                        negative_image_embeds=negative_image_embeds,
                        height=img_size,
                        width=img_size,
                        num_inference_steps=k_decoder_steps,
                        guidance_scale=k_guidance,
                        generator=generator
                    ).images[0]

                output_filename = f"{model_name}_{checkpoint_label}_{gen_params_label}_{variation_name}_{file_id}.png"
                output_path = os.path.join(variation_output_dir, output_filename)
                image.save(output_path)
                logger.info(f"  Saved image: {output_path}")

            except Exception as e:
                logger.error(f"  Failed generation for file ID {file_id}, variation {variation_name}: {e}\n{traceback.format_exc()}")

            if (prompts_processed_count * len(prompt_variations) + list(prompt_variations.keys()).index(variation_name)) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    logger.info(f"Finished generating images. Processed {prompts_processed_count} matching file IDs.")

def main():
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    prompts_csv_path = "/home/iris/Documents/deep_learning/data/input_csv/FILE_SUPERTOPIC_DESCRIPTION.csv"
    sample_list_path = "/home/iris/Documents/deep_learning/data/sample_list.txt"
    prompt_variations_path = "/home/iris/Documents/deep_learning/config/prompt_config.yaml"
    generation_output_base = "/home/iris/Documents/deep_learning/generated_images/iter2"
    lora_checkpoint_dir = "/home/iris/Documents/deep_learning/experiments/bigger_kandinsky/best_t5_kandinsky_unet_prior/config_0_loss_0.1713"

    config = load_config(config_path)
    if config is None:
        return

    # Load OpenAI API key
    openai_config = config.get("openai", {})
    openai_api_key = openai_config.get("api_key")
    if not openai_api_key:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OpenAI API key not found in config.yaml or environment variable OPENAI_API_KEY.")
            return
        logger.info("Loaded OpenAI API key from environment variable.")
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_dtype = torch.float16

    base_model_ids = {
        "kandinsky_prior": "kandinsky-community/kandinsky-2-2-prior",
        "kandinsky_decoder": "kandinsky-community/kandinsky-2-2-decoder"
    }

    # Define generation hyperparameter sets
    generation_param_sets = [
        {"kandinsky_prior_steps": 25, "kandinsky_decoder_steps": 50, "kandinsky_guidance": 4.0},
        {"kandinsky_prior_steps": 30, "kandinsky_decoder_steps": 60, "kandinsky_guidance": 4.0},
        {"kandinsky_prior_steps": 25, "kandinsky_decoder_steps": 50, "kandinsky_guidance": 3.5},
        {"kandinsky_prior_steps": 25, "kandinsky_decoder_steps": 50, "kandinsky_guidance": 5.0}
    ]

    model_name = "kandinsky"
    logger.info(f"Processing fine-tuned Kandinsky model from {lora_checkpoint_dir}")

    for gen_idx, gen_params in enumerate(generation_param_sets):
        gen_params_label = f"gen_params_{gen_idx}"
        logger.info(f"--- Running Generation with Params Set {gen_idx}: {gen_params} ---")

        pipeline, prior_pipeline = None, None
        try:
            pipeline, prior_pipeline = load_pipeline_with_lora(
                model_name=model_name,
                base_model_ids=base_model_ids,
                lora_checkpoint_dir=lora_checkpoint_dir,
                device=device,
                target_dtype=target_dtype
            )
            if pipeline and prior_pipeline:
                generate_images(
                    prior_pipeline=prior_pipeline,
                    decoder_pipeline=pipeline,
                    refiner_pipeline=None,  # No refiner for Kandinsky
                    model_name=model_name,
                    prompts_df=prompts_df,
                    output_base_dir=generation_output_base,
                    checkpoint_label="best_t5_kandinsky",
                    gen_params_label=gen_params_label,
                    sample_ids=sample_ids,
                    prompt_variations=prompt_variations,
                    gen_hyperparams=gen_params,
                    openai_utils=openai_utils
                )
            else:
                logger.error(f"Skipping generation for {gen_params_label} due to pipeline loading failure.")
        except Exception as e:
            logger.error(f"Failed generation for {model_name}/{gen_params_label}: {e}\n{traceback.format_exc()}")
        finally:
            logger.debug(f"Cleaning up after {gen_params_label}...")
            del pipeline
            del prior_pipeline
            gc.collect()
            torch.cuda.empty_cache()

    logger.info("Image generation script finished.")

if __name__ == "__main__":
    main()