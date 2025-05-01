import os
import logging
import yaml
import json
import gc
import torch
import traceback
import shutil # For copying best model directory
from itertools import product

from bigger_kandinsky import FinetuneModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

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
    logger.info(accelerator.state, main_process_only=True)
    # set_seed(42) # Optional: Set seed for reproducibility across processes

    # --- Initialize Tracking for Best Hyperparameters ---
    overall_best_val_loss = float('inf')
    best_hyperparam_details = None # Will store {'config_idx': idx, 'hyperparams': dict, 'checkpoint_path': path, 'val_loss': float}
    last_run_details = None # Will store {'config_idx': idx, 'hyperparams': dict, 'checkpoint_path': path, 'epoch': int}

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
        # More structured way to define hyperparameters to test
        param_grid = {
            'learning_rate': [1e-5, 5e-6, 2e-6],
            'lora_r': [4, 8, 16],
            'apply_lora_unet': [True], # Assuming always true for this experiment
            'epochs': [5, 10, 15], # Keep epochs fixed for fair comparison, or add to grid
            'batch_size': [1,2,5],
            'val_split': [0.2]
        }
        keys, values = zip(*param_grid.items())
        hyperparam_configs = [dict(zip(keys, v)) for v in product(*values)]
        logger.info(f"Generated {len(hyperparam_configs)} hyperparameter configurations to test.", main_process_only=True)
        # ------------------------------------

        for model_name, base_output_dir in models_to_train:
            output_dir = base_output_dir
            logger.info(f"========== Starting Pipeline for: {model_name} ==========", main_process_only=True)

            # Reset best for this model
            overall_best_val_loss = float('inf')
            best_hyperparam_details = None
            last_run_details = None

            for idx, hyperparams in enumerate(hyperparam_configs):
                config_name = f"hyperparam_config_{idx}"
                logger.info(f"--- Hyperparameter Config {idx+1}/{len(hyperparam_configs)} ---")
                hyperparam_dir = os.path.join(output_dir, f"hyperparam_config_{idx}")
                if accelerator.is_main_process: os.makedirs(hyperparam_dir, exist_ok=True)
                finetuner = None
                try:
                    finetuner = FinetuneModel(model_name, hyperparam_dir, accelerator, logger_instance=logger)
                    finetuner.load_model()
                    finetuner.modify_architecture(
                        # apply_lora_to_text_encoder=hyperparams['apply_lora_text_encoder'],
                        apply_lora_to_unet=hyperparams['apply_lora_unet']
                    )
                    current_run_best_val_loss = finetuner.fine_tune(
                        dataset_path=dataset_path,
                        epochs=hyperparams.get('epochs', 1),
                        batch_size=hyperparams.get('batch_size', 1),
                        learning_rate=hyperparams.get('learning_rate', 1e-5),
                        val_split=hyperparams.get('val_split', 0.2),
                        gradient_accumulation_steps=gradient_accumulation_steps
                    )
                    logger.info(f"Completed finetuning for {model_name} with hyperparameter config {idx+1}.")
                    current_run_best_epoch = finetuner.best_epoch # Get best epoch from this run
                    current_run_last_epoch = finetuner.current_epoch # Get last epoch from this run
                    logger.info(f"Completed run {idx+1}. Best Val Loss for this run: {current_run_best_val_loss:.4f} at epoch {current_run_best_epoch}", main_process_only=True)

                    # --- Track Overall Best ---
                    if current_run_best_val_loss < overall_best_val_loss:
                        overall_best_val_loss = current_run_best_val_loss
                        best_hyperparam_details = {
                            'config_idx': idx,
                            'hyperparams': hyperparams,
                            # Path to the directory saved by fine_tune's internal best checkpointing
                            'checkpoint_path': os.path.join(hyperparam_dir, f"best_epoch_{current_run_best_epoch}"),
                            'val_loss': current_run_best_val_loss,
                            'epoch': current_run_best_epoch
                        }
                        logger.info(f"*** New overall best validation loss found: {overall_best_val_loss:.4f} (Config {idx}) ***", main_process_only=True)

                    # --- Track Last Run Details ---
                    last_run_details = {
                         'config_idx': idx,
                         'hyperparams': hyperparams,
                         'checkpoint_path': os.path.join(hyperparam_dir, f"last_epoch"), # Saved as 'last_epoch' by fine_tune
                         'epoch': current_run_last_epoch
                    }

                except Exception as e:
                    logger.error(f"Run FAILED for {model_name}, {config_name}: {e}\n{traceback.format_exc()}", main_process_only=True)

                finally:
                    logger.info(f"--- Finished cleanup for {model_name}, {config_name} ---", main_process_only=True)
                    del finetuner
                    gc.collect(); torch.cuda.empty_cache()
                    logger.info(f"Cleaned up resources for {model_name}, config {idx+1}.")

            # --- Save Overall Best and Last Models ---
            if accelerator.is_main_process:
                logger.info(f"--- Final Saving for Model: {model_name} ---")

                # Save Last Model
                if last_run_details and os.path.exists(last_run_details['checkpoint_path']):
                    final_last_dir = os.path.join(output_dir, "final_last_run_model")
                    logger.info(f"Copying last run checkpoint from {last_run_details['checkpoint_path']} to {final_last_dir}")
                    try:
                        shutil.copytree(last_run_details['checkpoint_path'], final_last_dir, dirs_exist_ok=True)
                        # Save corresponding hyperparameters
                        last_hyperparam_path = os.path.join(final_last_dir, "hyperparameters_last_run.json")
                        with open(last_hyperparam_path, 'w') as f:
                            json.dump(last_run_details['hyperparams'], f, indent=4)
                        logger.info(f"Saved last run model and config to {final_last_dir}")
                    except Exception as e:
                        logger.error(f"Failed to copy/save last run model: {e}")
                else:
                    logger.warning("Could not find checkpoint for the last hyperparameter run.")

                # Save Best Model
                if best_hyperparam_details and os.path.exists(best_hyperparam_details['checkpoint_path']):
                    final_best_dir = os.path.join(output_dir, "final_overall_best_model")
                    logger.info(f"Copying overall best checkpoint from {best_hyperparam_details['checkpoint_path']} to {final_best_dir}")
                    try:
                        shutil.copytree(best_hyperparam_details['checkpoint_path'], final_best_dir, dirs_exist_ok=True)
                        # Save corresponding hyperparameters and performance
                        best_hyperparam_path = os.path.join(final_best_dir, "hyperparameters_overall_best.json")
                        with open(best_hyperparam_path, 'w') as f:
                            json.dump(best_hyperparam_details, f, indent=4, default=str) # Use default=str for safety
                        logger.info(f"Saved overall best model and config to {final_best_dir}")
                    except Exception as e:
                        logger.error(f"Failed to copy/save overall best model: {e}")
                else:
                    logger.warning("Could not find checkpoint for the overall best hyperparameter run.")

            logger.info(f"========== Finished Pipeline for: {model_name} ==========\n", main_process_only=True)

    except Exception as e:
        logger.critical(f"Main execution failed: {e}\n{traceback.format_exc()}", main_process_only=True)
    finally:
        logger.info("Hyperparameter search script completed.", main_process_only=True)
        # logging.shutdown() # Usually not needed with accelerate logger

if __name__ == "__main__":
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    run_finetune(config_path)