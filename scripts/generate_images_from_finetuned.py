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

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/iris/Documents/deep_learning/src/logs/image_generation_filtered.log", mode='w'),
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
            logger.info("Loading SDXL VAE in FP32...")
            vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float32)
            logger.info("Loading SDXL Pipeline components...")
            # Load the base pipeline *without* LoRA first
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                base_model_id, vae=vae, torch_dtype=target_dtype,
                variant="fp16", use_safetensors=True,
            )
            logger.info("Loaded SDXL base pipeline.")

            # --- Load SDXL LoRA using pipeline method ---
            logger.info(f"Loading SDXL LoRA weights from: {lora_checkpoint_dir}")
            # Check if the main weight file exists in the checkpoint directory itself
            # (diffusers >= 0.20.0 standard) or fallback to checking subfolders
            main_safe_path = os.path.join(lora_checkpoint_dir, lora_weight_name)
            main_bin_path = os.path.join(lora_checkpoint_dir, lora_weight_name_bin)

            if os.path.exists(main_safe_path) or os.path.exists(main_bin_path):
                 # Load directly from the checkpoint directory
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

                 # Helper to get weight name from subfolder
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
                logger.info(f"Loading Kandinsky VAE (MoVQ) in FP32 from {decoder_id}/movq")
                vae = VQModel.from_pretrained(decoder_id, subfolder="movq", torch_dtype=torch.float32)
                pipeline.movq = vae
                logger.info("Loaded and assigned Kandinsky VQ VAE in FP32.")
            except Exception as e: logger.warning(f"Could not load separate VQ VAE for Kandinsky: {e}")
            logger.info("Loaded Kandinsky base pipelines.")

            # --- Load LoRA Adapters for Kandinsky ---
            logger.info(f"Loading LoRA adapters from: {lora_checkpoint_dir}")
            unet_lora_path = os.path.join(lora_checkpoint_dir, "unet_lora")
            te1_lora_path = os.path.join(lora_checkpoint_dir, "text_encoder_lora") # Prior's TE

            # Helper for load_adapter
            def _load_adapter_safely(component, adapter_path):
                if not component or not hasattr(component, 'load_adapter'): logger.warning(...); return False
                if not os.path.isdir(adapter_path): logger.warning(...); return False
                lora_weight_name_safe = "adapter_model.safetensors"
                lora_weight_name_bin = "adapter_model.bin"
                if not (os.path.exists(os.path.join(adapter_path, lora_weight_name_safe)) or os.path.exists(os.path.join(adapter_path, lora_weight_name_bin))):
                    logger.warning(f"No adapter weight file found in {adapter_path}"); return False
                try:
                    logger.info(f"Loading adapter from {adapter_path} into {component.__class__.__name__}")
                    component.load_adapter(adapter_path, adapter_name="default") # Use default adapter name
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

# --- generate_images function (remains the same) ---
def generate_images(prior_pipeline, decoder_pipeline, model_name, prompts_df, output_base_dir, checkpoint_label, sample_ids, prompt_variations):
    """Generates images for specified prompts and variations using the loaded pipeline(s)."""
    # ... (Implementation from previous step - no changes needed here) ...
    if decoder_pipeline is None: logger.error("Main/Decoder pipeline is None."); return
    if model_name == "kandinsky" and prior_pipeline is None: logger.error("Kandinsky requires prior_pipeline."); return
    if not prompt_variations: logger.error("No prompt variations provided."); return

    logger.info(f"Generating images for {len(sample_ids)} specified IDs using {len(prompt_variations)} prompt variations.")
    img_size = 1024 if model_name == "sdxl" else 512
    device = decoder_pipeline.device

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
            variation_output_dir = os.path.join(output_base_dir, model_name, checkpoint_label, variation_name)
            os.makedirs(variation_output_dir, exist_ok=True)

            try: final_prompt = prompt_template.format(txt_prompt=prompt_details)
            except Exception as fmt_err: logger.error(f"  Error formatting prompt: {fmt_err}"); continue
            logger.debug(f"  Formatted Prompt: {final_prompt}")

            try:
                generator = torch.Generator(device=device).manual_seed(42 + index + hash(variation_name))
                negative_prompt = "low quality, bad quality, blurry, text, words, letters, signature"
                image = None
                with torch.no_grad():
                    if model_name == "kandinsky":
                        logger.debug("  Running Kandinsky Prior...")
                        prior_output = prior_pipeline(prompt=final_prompt, negative_prompt=negative_prompt, num_inference_steps=25, generator=generator)
                        image_embeds = prior_output.image_embeds; negative_image_embeds = prior_output.negative_image_embeds
                        logger.debug("  Running Kandinsky Decoder...")
                        image = decoder_pipeline(prompt=final_prompt, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=img_size, width=img_size, num_inference_steps=50, guidance_scale=4.0, generator=generator).images[0]
                    elif model_name == "sdxl":
                        image = decoder_pipeline(
                            prompt=final_prompt, negative_prompt=negative_prompt, height=img_size, width=img_size,
                            num_inference_steps=30, guidance_scale=7.5, generator=generator, num_images_per_prompt=1
                        ).images[0]

                if image:
                    output_filename = f"{model_name}_{checkpoint_label}_{variation_name}_{file_id}.png"
                    output_path = os.path.join(variation_output_dir, output_filename)
                    image.save(output_path)
                    logger.info(f"  Saved image: {output_path}")
                else: logger.warning(f"  Image generation failed for {file_id}, variation {variation_name}")

            except Exception as e: logger.error(f"  Failed generation for file ID {file_id}, variation {variation_name}: {e}\n{traceback.format_exc()}")

            if (prompts_processed_count * len(prompt_variations) + list(prompt_variations.keys()).index(variation_name)) % 5 == 0:
                 gc.collect(); torch.cuda.empty_cache()

    logger.info(f"Finished generating images. Processed {prompts_processed_count} matching file IDs from sample list.")


# --- main function ---
def main():
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    prompts_csv_path = "/home/iris/Documents/deep_learning/data/input_csv/FILE_SUPERTOPIC_DESCRIPTION.csv"
    sample_list_path = "/home/iris/Documents/deep_learning/data/sample_list.txt"
    prompt_variations_path = "/home/iris/Documents/deep_learning/config/prompt_config.yaml"
    generation_output_base = "/home/iris/Documents/deep_learning/generated_images/iter1"

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
                        pipeline, prior_pipeline = None, None
                        try:
                            pipeline, prior_pipeline = load_pipeline_with_lora(
                                model_name=model_name,
                                base_model_ids=current_base_ids,
                                lora_checkpoint_dir=checkpoint_dir,
                                device=device,
                                target_dtype=target_dtype
                            )
                            if pipeline or (model_name == "kandinsky" and prior_pipeline):
                                generate_images(
                                    prior_pipeline=prior_pipeline,
                                    decoder_pipeline=pipeline,
                                    model_name=model_name,
                                    prompts_df=prompts_df,
                                    output_base_dir=generation_output_base,
                                    checkpoint_label=f"{config_dir_name}_{save_type}",
                                    sample_ids=sample_ids,
                                    prompt_variations=prompt_variations
                                )
                            else:
                                logger.error(f"Skipping generation for {save_type} due to pipeline loading failure.")
                        except Exception as e:
                            logger.error(f"Failed generation pipeline for {model_name}/{config_dir_name}/{save_type}: {e}\n{traceback.format_exc()}")
                        finally:
                            logger.info(f"Cleaning up after {save_type} checkpoint...")
                            del pipeline; del prior_pipeline
                            gc.collect(); torch.cuda.empty_cache()
                    else:
                        logger.warning(f"No '{save_type}' checkpoint found for {config_dir_name}. Skipping.")
                logger.info(f"--- Finished processing Config Directory: {config_dir_name} ---")
        logger.info(f"========== Finished processing model: {model_name} ==========")

    logger.info("Image generation script finished.")

if __name__ == "__main__":
    main()
