import os
import torch
import logging
import yaml
import json
import pandas as pd
import gc
import re # For finding epoch numbers
import traceback
import re
from pathlib import Path # For easier path handling
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    AutoencoderKL,
    VQModel,
    DPMSolverMultistepScheduler, # Example scheduler
    KandinskyV22PriorPipeline,
    KandinskyV22Pipeline,
)
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer
# from peft import PeftModel # Use PeftModel for loading adapters. # Removed PeftModel import as pipeline.load_lora_weights is used

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/iris/Documents/deep_learning/src/logs/image_generation.log", mode='w'),
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
        return None # Return None to indicate failure
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        raise

def load_prompt_variations(yaml_path):
    """Loads prompt variations from a YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            # Extract the list of prompts under the 'theory_exp' key
            prompt_list = config.get('theory_exp', {}).get('prompts', [])
            # Convert the list of single-key dicts into a single dict
            variations = {k: v for item in prompt_list for k, v in item.items()}
            if not variations:
                 logger.warning(f"No prompt variations found or structure incorrect in {yaml_path}")
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
            # Read lines, strip whitespace, convert to string, filter empty lines
            ids = {str(line.strip()) for line in f if line.strip()}
            logger.info(f"Loaded {len(ids)} sample IDs from {txt_path}")
            return ids
    except FileNotFoundError:
        logger.error(f"Sample ID list file not found: {txt_path}")
        return set() # Return empty set on error
    except Exception as e:
        logger.error(f"Failed to load sample IDs from {txt_path}: {e}")
        return set()
    

def find_checkpoint_dirs(base_model_dir):
    """Finds 'best' and 'last' checkpoint directories."""
    checkpoints = {"best": None, "last": None}
    best_epoch = -1
    last_epoch = -1

    if not os.path.isdir(base_model_dir):
        logger.warning(f"Base directory not found: {base_model_dir}")
        return checkpoints

    for dirname in os.listdir(base_model_dir):
        dirpath = os.path.join(base_model_dir, dirname)
        if os.path.isdir(dirpath):
            # Match directories starting with 'best_epoch_' or 'last_epoch_'
            best_match = re.match(r'best_epoch_(\d+)', dirname)
            last_match = re.match(r'last_epoch_(\d+)', dirname)

            if best_match:
                try:
                    epoch = int(best_match.group(1))
                    # Keep the highest epoch number found for 'best'
                    if epoch > best_epoch:
                        checkpoints["best"] = dirpath
                        best_epoch = epoch
                except ValueError:
                    logger.warning(f"Could not parse epoch number from directory: {dirname}")
            elif last_match:
                 try:
                    epoch = int(last_match.group(1))
                    # Keep the highest epoch number found for 'last'
                    if epoch > last_epoch:
                        checkpoints["last"] = dirpath
                        last_epoch = epoch
                 except ValueError:
                    logger.warning(f"Could not parse epoch number from directory: {dirname}")

    if checkpoints["best"]: logger.info(f"Found best checkpoint: {checkpoints['best']} (Epoch {best_epoch})")
    else: logger.warning(f"No 'best' checkpoint directory found in {base_model_dir}")

    if checkpoints["last"]: logger.info(f"Found last checkpoint: {checkpoints['last']} (Epoch {last_epoch})")
    else: logger.warning(f"No 'last' checkpoint directory found in {base_model_dir}")

    return checkpoints


def load_pipeline_with_lora(model_name, base_model_ids, lora_checkpoint_dir, device, target_dtype):
    """Loads the base pipeline(s) and attaches LoRA weights."""
    logger.info(f"Loading base model components for {model_name}...")
    pipeline = None
    prior_pipeline = None
    # Define the expected LoRA weight filename (prioritize safetensors)
    lora_weight_name = "adapter_model.safetensors"
    lora_weight_name_bin = "adapter_model.bin" # Fallback

    try:
        # --- Load Base Models ---
        if model_name == "sdxl":
            base_model_id = base_model_ids["sdxl"]
            logger.info("Loading SDXL VAE in FP32...")
            vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float32)
            logger.info("Loading SDXL Pipeline components...")
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                base_model_id, vae=vae, torch_dtype=target_dtype,
                variant="fp16", use_safetensors=True,
            )
            # Assign text encoders explicitly if needed for LoRA loading by subfolder
            pipeline.text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder", torch_dtype=target_dtype, variant="fp16")
            pipeline.text_encoder_2 = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder_2", torch_dtype=target_dtype, variant="fp16")

        elif model_name == "kandinsky":
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

        else:
            raise ValueError(f"Unsupported model_name for generation: {model_name}")

        if pipeline is None and prior_pipeline is None:
            raise RuntimeError(f"Failed to load base pipeline(s) for {model_name}")

        # --- Load LoRA Weights ---
        logger.info(f"Loading LoRA weights from: {lora_checkpoint_dir}")
        unet_lora_path = os.path.join(lora_checkpoint_dir, "unet_lora")
        te1_lora_path = os.path.join(lora_checkpoint_dir, "text_encoder_lora")
        te2_lora_path = os.path.join(lora_checkpoint_dir, "text_encoder_2_lora")

        # Helper to determine the weight file name
        def get_lora_weight_name(path):
            safetensors_path = os.path.join(path, lora_weight_name)
            bin_path = os.path.join(path, lora_weight_name_bin)
            if os.path.exists(safetensors_path):
                return lora_weight_name
            elif os.path.exists(bin_path):
                return lora_weight_name_bin
            else:
                return None

        # Load UNet LoRA
        if os.path.exists(unet_lora_path) and hasattr(pipeline, 'unet'):
            weight_file = get_lora_weight_name(unet_lora_path)
            if weight_file:
                try:
                    logger.info(f"Loading UNet LoRA weights from {unet_lora_path} using {weight_file}...")
                    # Use load_lora_weights for pipelines that support it (like SDXL)
                    # For others like Kandinsky, load adapter directly into the component
                    if hasattr(pipeline, 'load_lora_weights'):
                         pipeline.load_lora_weights(unet_lora_path, weight_name=weight_file)
                    elif hasattr(pipeline.unet, 'load_adapter'):
                         pipeline.unet.load_adapter(unet_lora_path, weight_name=weight_file) # load_adapter needs path, not dir+subfolder
                    else:
                         logger.warning(f"Cannot load UNet LoRA for {model_name}: No suitable loading method found.")
                    logger.info("Loaded UNet LoRA.")
                except Exception as e:
                    logger.error(f"Failed to load UNet LoRA from {unet_lora_path}: {e}")
            else:
                 logger.warning(f"No weight file found in {unet_lora_path}")


        # Load Text Encoder LoRA(s)
        if model_name == "sdxl":
            if os.path.exists(te1_lora_path) and hasattr(pipeline, 'text_encoder'):
                weight_file = get_lora_weight_name(te1_lora_path)
                if weight_file:
                    try:
                        logger.info(f"Loading SDXL Text Encoder 1 LoRA weights from {te1_lora_path} using {weight_file}...")
                        pipeline.load_lora_weights(te1_lora_path, weight_name=weight_file, text_encoder=pipeline.text_encoder)
                        logger.info("Loaded Text Encoder 1 LoRA.")
                    except Exception as e: logger.error(f"Failed to load TE1 LoRA: {e}")
                else: logger.warning(f"No weight file found in {te1_lora_path}")

            if os.path.exists(te2_lora_path) and hasattr(pipeline, 'text_encoder_2'):
                weight_file = get_lora_weight_name(te2_lora_path)
                if weight_file:
                    try:
                        logger.info(f"Loading SDXL Text Encoder 2 LoRA weights from {te2_lora_path} using {weight_file}...")
                        pipeline.load_lora_weights(te2_lora_path, weight_name=weight_file, text_encoder=pipeline.text_encoder_2)
                        logger.info("Loaded Text Encoder 2 LoRA.")
                    except Exception as e: logger.error(f"Failed to load TE2 LoRA: {e}")
                else: logger.warning(f"No weight file found in {te2_lora_path}")

        elif model_name == "kandinsky":
            # Load into the prior pipeline's text encoder using load_adapter
            if os.path.exists(te1_lora_path) and prior_pipeline and hasattr(prior_pipeline, 'text_encoder') and hasattr(prior_pipeline.text_encoder, 'load_adapter'):
                # load_adapter doesn't need weight_name specified if using default names
                try:
                    logger.info(f"Loading Kandinsky Prior Text Encoder adapter from {te1_lora_path}...")
                    prior_pipeline.text_encoder.load_adapter(te1_lora_path) # No weight_name needed
                    logger.info("Loaded Prior Text Encoder LoRA adapter.")
                except Exception as e:
                    logger.error(f"Failed to load Prior Text Encoder adapter from {te1_lora_path}: {e}")
            elif not (prior_pipeline and hasattr(prior_pipeline, 'text_encoder')):
                 logger.warning("Prior pipeline or its text encoder not found for Kandinsky LoRA loading.")


        # Move pipelines to device
        if pipeline: pipeline.to(device)
        if prior_pipeline: prior_pipeline.to(device)

        logger.info(f"Finished loading pipeline(s) and LoRA for {model_name}.")
        return pipeline, prior_pipeline

    except Exception as e:
        logger.error(f"Failed during model or LoRA loading for {model_name} from {lora_checkpoint_dir}: {e}\n{traceback.format_exc()}")
        return None, None # Indicate failure
    

def generate_images(prior_pipeline, decoder_pipeline, model_name, prompts_df, output_base_dir, checkpoint_label, sample_ids, prompt_variations):
    """
    Generates images for specified prompts and variations using the loaded pipeline(s).

    Args:
        prior_pipeline: The loaded prior pipeline (used for Kandinsky). None otherwise.
        decoder_pipeline: The loaded main/decoder pipeline (SDXL or Kandinsky Decoder).
        model_name (str): Name of the model (e.g., 'sdxl', 'kandinsky').
        prompts_df (pd.DataFrame): DataFrame containing prompts and details.
        output_base_dir (str): Base directory to save generated images.
        checkpoint_label (str): Label identifying the model config and checkpoint type.
        sample_ids (set): A set of string IDs for which to generate images.
        prompt_variations (dict): Dictionary of prompt templates.
    """
    if decoder_pipeline is None: # Need at least the main/decoder pipeline
        logger.error("Main/Decoder pipeline is None, cannot generate images.")
        return
    if model_name == "kandinsky" and prior_pipeline is None:
        logger.error("Kandinsky generation requires prior_pipeline, but it's None.")
        return

    if not prompt_variations: logger.error("No prompt variations provided."); return

    logger.info(f"Generating images for {len(sample_ids)} specified IDs using {len(prompt_variations)} prompt variations.")

    img_size = 1024 if model_name == "sdxl" else 512
    device = decoder_pipeline.device # Get device from the pipeline

    # --- DEBUGGING: Print sample IDs and their type ---
    if sample_ids:
        sample_id_list = list(sample_ids)
        logger.debug(f"Sample IDs to generate (first 5): {sample_id_list[:5]}")
        logger.debug(f"Type of first sample ID: {type(sample_id_list[0])}")
    else:
        logger.warning("Sample ID list is empty!")
    # --- END DEBUGGING ---

    prompts_processed_count = 0
    for index, row in prompts_df.iterrows():
        prompt_id = str(row.get('file', f'prompt_{index}'))
        if prompt_id not in sample_ids: continue

        # --- DEBUGGING: Print current prompt ID and type being checked ---
        logger.debug(f"Checking prompt ID: '{prompt_id}' (Type: {type(prompt_id)}) against sample_ids set.")
        # --- END DEBUGGING ---

        # --- If ID matches, proceed ---
        prompts_processed_count += 1
        prompt_details = row.get('prompt_details', '')
        logger.info(f"Processing Prompt ID: {prompt_id} (Row {index})") # Log row index too

        for variation_name, prompt_template in prompt_variations.items():
            logger.info(f"  Generating for variation: {variation_name}")
            variation_output_dir = os.path.join(output_base_dir, model_name, checkpoint_label, variation_name)
            os.makedirs(variation_output_dir, exist_ok=True)

            try:
                final_prompt = prompt_template.format(txt_prompt=prompt_details)
                logger.debug(f"  Formatted Prompt: {final_prompt}")
            except Exception as fmt_err: logger.error(f"  Error formatting prompt: {fmt_err}"); continue

            try:
                generator = torch.Generator(device=device).manual_seed(42 + index + hash(variation_name))
                # Define a generic negative prompt (can be customized)
                negative_prompt = "low quality, bad quality, blurry, text, words, letters, signature"

                image = None
                with torch.no_grad():
                    """two pipelines for Kandinsky."""
                    if model_name == "kandinsky":
                        # --- Kandinsky Stage 1: Prior ---
                        logger.debug("  Running Kandinsky Prior...")
                        prior_output = prior_pipeline(
                            prompt=final_prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=25, # Prior steps
                            generator=generator
                        )
                        image_embeds = prior_output.image_embeds
                        negative_image_embeds = prior_output.negative_image_embeds
                        logger.debug("  Finished Kandinsky Prior.")

                        # --- Kandinsky Stage 2: Decoder ---
                        logger.debug("  Running Kandinsky Decoder...")
                        # Pass image embeds to the decoder pipeline
                        image = decoder_pipeline(
                            prompt=final_prompt,
                            image_embeds=image_embeds,
                            negative_image_embeds=negative_image_embeds,
                            height=img_size,
                            width=img_size,
                            num_inference_steps=50, # Decoder steps (often more)
                            guidance_scale=4.0,     # Kandinsky often uses lower guidance
                            generator=generator
                        ).images[0]
                        logger.debug("  Finished Kandinsky Decoder.")

                    elif model_name == "sdxl":
                        # --- SDXL Generation (Single Stage) ---
                        image = decoder_pipeline( # Use the main pipeline directly
                            prompt=final_prompt,
                            negative_prompt=negative_prompt,
                            height=img_size,
                            width=img_size,
                            num_inference_steps=30,
                            guidance_scale=7.5,
                            generator=generator
                        ).images[0]

                    # Add elif for other models if needed

                # Save the image if generation was successful
                if image:
                    output_filename = f"{model_name}_{checkpoint_label}_{variation_name}_{prompt_id}.png"
                    output_path = os.path.join(variation_output_dir, output_filename)
                    image.save(output_path)
                    logger.info(f"  Saved image: {output_path}")
                else:
                     logger.warning(f"  Image generation did not produce an output for {prompt_id}, variation {variation_name}")

            except Exception as e:
                logger.error(f"  Failed generation for prompt ID {prompt_id}, variation {variation_name}: {e}\n{traceback.format_exc()}")

            # Optional periodic cleanup
            if (index * len(prompt_variations) + list(prompt_variations.keys()).index(variation_name)) % 5 == 0:
                 gc.collect(); torch.cuda.empty_cache()

    logger.info(f"Finished generating images for {prompts_processed_count} prompt IDs.")


def main():
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml" # Or get from args
    prompts_csv_path = "/home/iris/Documents/deep_learning/data/input_csv/FILE_SUPERTOPIC_DESCRIPTION.csv" # Path to your prompts CSV
    generation_output_base = "/home/iris/Documents/deep_learning/generated_images/iter1" # Base dir for generated images
    sample_list_path = "/home/iris/Documents/deep_learning/data/sample_list.txt"
    prompt_variations_path = "/home/iris/Documents/deep_learning/config/prompt_config.yaml"

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

    # Updated base model IDs
    base_model_ids = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "kandinsky_prior": "kandinsky-community/kandinsky-2-2-prior",
        "kandinsky_decoder": "kandinsky-community/kandinsky-2-2-decoder",
    }
    models_to_generate = config.get("models_to_generate", ["sdxl", "kandinsky"])

    for model_name in models_to_generate: # e.g., "sdxl", "kandinsky"
        model_finetune_base = os.path.join(base_output_dir, model_name)
        logger.info(f"Processing model: {model_name} (Fine-tune base: {model_finetune_base})")
        if not os.path.isdir(model_finetune_base): logger.warning(f"Dir not found: {model_finetune_base}."); continue

        # Determine correct base ID(s) for loading
        current_base_ids = {}
        if model_name == "sdxl":
            current_base_ids["sdxl"] = base_model_ids["sdxl"]
        elif model_name == "kandinsky":
            current_base_ids["kandinsky_prior"] = base_model_ids["kandinsky_prior"]
            current_base_ids["kandinsky_decoder"] = base_model_ids["kandinsky_decoder"]
        else:
             logger.warning(f"Base model IDs not fully defined for {model_name}. Skipping.")
             continue

        for config_dir_name in os.listdir(model_finetune_base):
            if config_dir_name.startswith("hyperparam_config_"):
                config_dir_path = os.path.join(model_finetune_base, config_dir_name)
                logger.info(f"--- Processing Config Directory: {config_dir_name} ---")
                checkpoints = find_checkpoint_dirs(config_dir_path)

                for save_type, checkpoint_dir in checkpoints.items():
                    if checkpoint_dir:
                        logger.info(f"--- Generating images for '{save_type}' checkpoint ---")
                        # Initialize pipelines to None for cleanup
                        pipeline = None
                        prior_pipeline = None
                        try:
                            # Pass the dictionary of base IDs needed for the specific model
                            pipeline, prior_pipeline = load_pipeline_with_lora(
                                model_name=model_name,
                                base_model_ids=current_base_ids, # Pass the relevant ID(s)
                                lora_checkpoint_dir=checkpoint_dir,
                                device=device,
                                target_dtype=target_dtype
                            )
                            # Pass both pipelines to generate_images
                            generate_images(
                                prior_pipeline=prior_pipeline, # Will be None for SDXL
                                decoder_pipeline=pipeline,    # Main pipeline for SDXL, decoder for Kandinsky
                                model_name=model_name,
                                prompts_df=prompts_df,
                                output_base_dir=generation_output_base,
                                checkpoint_label=f"{config_dir_name}_{save_type}",
                                sample_ids=sample_ids,
                                prompt_variations=prompt_variations
                            )
                        except Exception as e:
                            logger.error(f"Failed generation pipeline for {model_name}/{config_dir_name}/{save_type}: {e}\n{traceback.format_exc()}")
                        finally:
                            logger.info(f"Cleaning up after {save_type} checkpoint...")
                            # Delete both potential pipelines
                            del pipeline
                            del prior_pipeline
                            gc.collect()
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                    else:
                        logger.warning(f"No '{save_type}' checkpoint found for {config_dir_name}.")
                logger.info(f"--- Finished processing Config Directory: {config_dir_name} ---")
        logger.info(f"========== Finished processing model: {model_name} ==========")

    logger.info("Image generation script finished.")


if __name__ == "__main__":
    main()
