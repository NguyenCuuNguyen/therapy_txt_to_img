import os
import logging
import yaml
import json
import gc
import torch
from bigger_kandinsky import FinetuneModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# # Configure logging
# logging.basicConfig(
#     filename='/home/iris/Documents/deep_learning/src/logs/run_bigger_kandinsky.log',
#     level=logging.DEBUG,
#     format='%(asctime)s %(message)s',
#     filemode='w',
# )
# logger = logging.getLogger(__name__)

# Import the modified FinetuneModel class
try:
    # Make sure the filename matches where you saved the class
    # Assuming it's named 'finetune_model_accelerate.py' based on previous context
    from bigger_kandinsky import FinetuneModel
except ImportError:
     # Use standard logging here as accelerate logger might not be ready
     logging.error("Could not import FinetuneModel from finetune_model_accelerate.py")
     # Define dummy class
     class FinetuneModel:
         def __init__(self, *args, **kwargs): pass
         def load_model_components(self): pass
         def modify_architecture(self, *args, **kwargs): pass
         def fine_tune(self, *args, **kwargs): pass
         def save_model_state(self, *args, **kwargs): pass

def load_config(config_path):
    """Load configuration from YAML file."""
    # Use standard logger temporarily before accelerate logger is fully set up in main
    temp_logger = logging.getLogger(__name__ + ".config_loader")
    temp_logger.propagate = False # Prevent duplicate logging if root logger gets configured later
    temp_logger.addHandler(logging.StreamHandler()) # Log config loading errors to console
    temp_logger.setLevel(logging.INFO)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            temp_logger.info(f"Loaded config: {config_path}")
            return config
    except Exception as e:
        temp_logger.error(f"Failed load config {config_path}: {e}")
        raise

def run_finetune(config_path):
    """Run finetuning for multiple models, epochs, and hyperparameters."""
    # --- Accelerator Setup ---
    # Load config first to get accumulation steps if defined there
    config_temp = load_config(config_path)
    if config_temp is None: return # Exit if config fails
    gradient_accumulation_steps = config_temp.get("gradient_accumulation_steps", 4)

    # !--- Instantiate Accelerator Correctly ---!
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16', # Or 'bf16'
        log_with=None # Disable default trackers unless configured
    )

    # --- Setup Logging AFTER Accelerator Init ---
    # Use accelerate's logger from now on
    logger = get_logger(__name__, log_level="DEBUG")

    # Log accelerator state using accelerator's print for multi-process safety
    # or log only on main process
    if accelerator.is_main_process:
        logger.info(accelerator.state)
    # Alternatively, print on all processes for debugging:
    # accelerator.print(f"Accelerator state: {accelerator.state}")

        # --- Configure File Handler (Only on Main Process) ---
    if accelerator.is_main_process:
        log_file_path = "/home/iris/Documents/deep_learning/src/logs/run_bigger_kandinsky_accelerate.log" # Define log path
        log_dir = os.path.dirname(log_file_path)
        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print(f"Created log directory: {log_dir}")

            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setLevel("DEBUG")
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            # Add handler to the logger obtained from accelerate
            logger.addHandler(file_handler)
            logger.info(f"File logging configured to: {log_file_path}")
        except Exception as e:
            # Use print as logger might not have console handler yet if file handler failed
            print(f"ERROR: Failed to configure file logging: {e}")

    # set_seed(42) # Optional: Set seed for reproducibility across processes
    try:
        config = load_config(config_path)
        dataset_path = config.get("dataset_path", "/home/iris/Documents/deep_learning/data/finetune_dataset/coco/dataset.json")
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset JSON file not found at: {dataset_path}")
            return
        
        base_output_dir = config.get("base_output_dir", "/home/iris/Documents/deep_learning/experiments/bigger_kandinsky")
        if accelerator.is_main_process: os.makedirs(base_output_dir, exist_ok=True)
        
        # Define models to train
        models_to_train = [
            # ("sdxl", os.path.join(base_output_dir, "sdxl")),
            ("kandinsky", os.path.join(base_output_dir, "kandinsky")),
            # ("karlo", os.path.join(base_output_dir, "karlo"))
        ]
        
        # Hyperparameter configurations
        hyperparam_configs = [
            {
                'epochs': 5,
                'batch_size': 1,
                'learning_rate': 1e-5,
                'val_split': 0.2,
                'apply_lora_to_text_encoder': True,
                'apply_lora_to_unet': True
            },
            {
                'epochs': 5,
                'batch_size': 1,
                'learning_rate': 5e-6,
                'val_split': 0.2,
                'apply_lora_to_text_encoder': True,
                'apply_lora_to_unet': True
            }
        ]
        
        for model_name, output_dir in models_to_train:
            logger.info(f"========== Starting Pipeline for: {model_name} ==========")
            for idx, hyperparams in enumerate(hyperparam_configs):
                logger.info(f"--- Hyperparameter Config {idx+1}/{len(hyperparam_configs)} ---")
                hyperparam_dir = os.path.join(output_dir, f"hyperparam_config_{idx}")
                if accelerator.is_main_process: os.makedirs(hyperparam_dir, exist_ok=True)
                finetuner = None
                try:
                    finetuner = FinetuneModel(model_name, hyperparam_dir, accelerator, logger_instance=logger)
                    finetuner.load_model()
                    finetuner.modify_architecture(
                        apply_lora_to_text_encoder=hyperparams['apply_lora_to_text_encoder'],
                        apply_lora_to_unet=hyperparams['apply_lora_to_unet']
                    )
                    finetuner.fine_tune(
                        dataset_path=dataset_path,
                        epochs=hyperparams['epochs'],
                        batch_size=hyperparams['batch_size'],
                        learning_rate=hyperparams['learning_rate'],
                        val_split=hyperparams['val_split'],
                        gradient_accumulation_steps=gradient_accumulation_steps
                    )
                    logger.info(f"Completed finetuning for {model_name} with hyperparameter config {idx+1}.")
                except Exception as e:
                    logger.error(f"Finetuning failed for {model_name}, config {idx+1}: {e}")
                finally:
                    if finetuner:
                        try: del finetuner.model
                        except AttributeError: pass
                        try: del finetuner.tokenizer
                        except AttributeError: pass
                        try: del finetuner.tokenizer_2
                        except AttributeError: pass
                        try: del finetuner.text_encoder
                        except AttributeError: pass
                        try: del finetuner.text_encoder_2
                        except AttributeError: pass
                        try: del finetuner.unet
                        except AttributeError: pass
                        try: del finetuner.scheduler
                        except AttributeError: pass
                        try: del finetuner.vae
                        except AttributeError: pass
                        del finetuner
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(f"Cleaned up resources for {model_name}, config {idx+1}.")
            logger.info(f"========== Finished Pipeline for: {model_name} ==========\n")
    except Exception as e:
        logger.critical(f"Main execution failed: {e}")
    logger.info("Multi-epoch finetuning script completed.")

if __name__ == "__main__":
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    run_finetune(config_path)