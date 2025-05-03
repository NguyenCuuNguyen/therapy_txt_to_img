import os
import logging
import yaml
import gc
import torch
import traceback
import json
import numpy as np
from itertools import product
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from scripts.bigger_kandinsky_unet_prior import FinetuneModel, load_config
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from src.utils.dataset import load_dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run_finetune(config_path):
    """Run finetuning with k-fold cross-validation to find the best hyperparameters."""
    config = load_config(config_path)
    if config is None:
        return

    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        mixed_precision='fp16',
        log_with="tensorboard",
        project_dir=os.path.join(config.get("base_output_dir", "./output"), "logs")
    )

    logger = get_logger(__name__, log_level="DEBUG")
    if accelerator.is_main_process:
        log_file_path = "/home/iris/Documents/deep_learning/src/logs/run_bigger_kandinsky_accelerate_unet_prior.log"
        log_dir = os.path.dirname(log_file_path)
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setLevel("DEBUG")
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logging.getLogger().addHandler(file_handler)
            logger.info(f"File logging configured to: {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to configure file logging: {e}")
    logger.info(accelerator.state, main_process_only=True)
    set_seed(42)

    try:
        dataset_path = config.get("dataset_path", "/home/iris/Documents/deep_learning/data/finetune_dataset/coco/dataset.json")
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found: {dataset_path}")
            return

        base_output_dir = config.get("base_output_dir", "/home/iris/Documents/deep_learning/experiments/bigger_kandinsky/best_t5_kandinsky_unet_prior")
        if accelerator.is_main_process:
            os.makedirs(base_output_dir, exist_ok=True)

        models_to_train = config.get("models_to_train", ["kandinsky"])
        logger.info(f"Models selected for training: {models_to_train}", main_process_only=True)

        param_grid = {
            'learning_rate': config.get("learning_rate", [5e-8, 1e-7]),
            'lora_r': config.get("lora_r", [4, 8]),
            'apply_lora_unet': config.get("apply_lora_unet", [True]),
            'apply_lora_text_encoder': config.get("apply_lora_text_encoder", [False]),
            'epochs': config.get("epochs", [5]),
            'batch_size': config.get("batch_size", [1]),
            'lora_alpha': config.get("lora_alpha", [16]),
            'lora_dropout': config.get("lora_dropout", [0.1])
        }
        for key in param_grid:
            if not isinstance(param_grid[key], list):
                param_grid[key] = [param_grid[key]]
        keys, values = zip(*param_grid.items())
        hyperparam_configs = [dict(zip(keys, v)) for v in product(*values)]
        logger.info(f"Generated {len(hyperparam_configs)} hyperparameter configurations to test.", main_process_only=True)

        full_dataset = load_dataset(dataset_path)
        if full_dataset is None:
            logger.error("Failed to load dataset")
            return

        k_folds = config.get("k_folds", 5)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        train_val_splits = [(Subset(full_dataset, train_idx), Subset(full_dataset, val_idx)) for train_idx, val_idx in kf.split(range(len(full_dataset)))]

        performance_records = []

        for model_name in models_to_train:
            logger.info(f"========== Starting Pipeline for: {model_name} ==========", main_process_only=True)
            overall_best_val_loss = float('inf')
            best_config = None
            best_finetuner = None

            for idx, hyperparams in enumerate(hyperparam_configs):
                config_name = f"hyperparam_config_{idx}"
                logger.info(f"--- Running Hyperparameter Config {idx+1}/{len(hyperparam_configs)} ({config_name}) ---", main_process_only=True)
                logger.info(f"Hyperparameters: {hyperparams}", main_process_only=True)

                # Log GPU memory status before loading model
                if torch.cuda.is_available() and accelerator.is_main_process:
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GiB
                    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3  # GiB
                    free_memory = (total_memory - allocated_memory)
                    logger.info(f"GPU Memory Status: Total: {total_memory:.2f} GiB, Allocated: {allocated_memory:.2f} GiB, Free: {free_memory:.2f} GiB")

                finetuner = None
                fold_losses = []

                try:
                    finetuner = FinetuneModel(model_name, base_output_dir, accelerator, logger_instance=logger)
                    finetuner.load_model()
                    finetuner.modify_architecture(
                        apply_lora_to_unet=hyperparams['apply_lora_unet'],
                        apply_lora_to_text_encoder=hyperparams['apply_lora_text_encoder'],
                        lora_r=hyperparams['lora_r'],
                        lora_alpha=hyperparams['lora_alpha'],
                        lora_dropout=hyperparams['lora_dropout']
                    )
                    avg_val_loss, model_state, hyperparameters, best_epoch = finetuner.fine_tune(
                        dataset_path=dataset_path,
                        train_val_splits=train_val_splits,
                        epochs=hyperparams['epochs'],
                        batch_size=hyperparams['batch_size'],
                        learning_rate=float(hyperparams['learning_rate']),
                        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
                        lora_r=hyperparams['lora_r'],
                        lora_alpha=hyperparams['lora_alpha'],
                        lora_dropout=hyperparams['lora_dropout']
                    )
                    fold_losses = finetuner.fold_val_losses
                    logger.info(f"Completed run {idx+1}. Average validation loss across folds: {avg_val_loss:.4f}", main_process_only=True)

                    performance_records.append({
                        'config_idx': idx,
                        'hyperparameters': hyperparams,
                        'avg_val_loss': avg_val_loss,
                        'fold_losses': fold_losses,
                        'epoch': best_epoch
                    })

                    if accelerator.is_main_process:
                        logger.debug(f"Checking if new best validation loss: {avg_val_loss:.4f} < {overall_best_val_loss:.4f}")
                        if not np.isnan(avg_val_loss) and avg_val_loss < overall_best_val_loss:
                            overall_best_val_loss = avg_val_loss
                            best_config = {
                                'index': idx,
                                'hyperparameters': hyperparams,
                                'avg_val_loss': avg_val_loss,
                                'epoch': best_epoch,
                                'fold_losses': fold_losses
                            }
                            best_finetuner = finetuner
                            logger.info(f"*** New global best validation loss found: {overall_best_val_loss:.4f} (Config {idx}, Epoch {best_epoch}) ***", main_process_only=True)
                            logger.info(f"Best configuration updated: {best_config}", main_process_only=True)

                            save_dir = os.path.join(base_output_dir, f"config_{idx}_loss_{avg_val_loss:.4f}")
                            logger.debug(f"Attempting to save best model to: {save_dir}")
                            try:
                                os.makedirs(save_dir, exist_ok=True)
                                finetuner.save_model_state(
                                    epoch=best_epoch,
                                    val_loss=avg_val_loss,
                                    hyperparameters=hyperparams,
                                    subdir=save_dir
                                )
                                logger.info(f"Saved best model and hyperparameters for config {idx} with average validation loss {avg_val_loss:.4f} to {save_dir}")
                            except Exception as e:
                                logger.error(f"Failed to save best model to {save_dir}: {e}\n{traceback.format_exc()}")

                except Exception as e:
                    logger.error(f"Run FAILED for {model_name}, {config_name}: {e}\n{traceback.format_exc()}", main_process_only=True)

                finally:
                    if finetuner:
                        del finetuner
                    gc.collect()
                    torch.cuda.empty_cache()

            if best_config and accelerator.is_main_process:
                logger.info(f"Final best configuration for {model_name}: Config {best_config['index']}, Avg Val Loss: {best_config['avg_val_loss']:.4f}, Epoch: {best_config['epoch']}, Hyperparameters: {best_config['hyperparameters']}", main_process_only=True)
                summary_path = os.path.join(base_output_dir, "performance_summary.json")
                try:
                    with open(summary_path, 'w') as f:
                        json.dump(performance_records, f, indent=4)
                    logger.info(f"Saved performance summary to {summary_path}")
                except Exception as e:
                    logger.error(f"Failed to save performance summary to {summary_path}: {e}")
            else:
                logger.warning(f"No valid configuration found for {model_name}", main_process_only=True)
            logger.info(f"========== Finished Pipeline for: {model_name} ==========\n", main_process_only=True)

    except Exception as e:
        logger.critical(f"Main execution failed: {e}\n{traceback.format_exc()}", main_process_only=True)
    finally:
        logger.info("Hyperparameter search script completed.", main_process_only=True)

if __name__ == "__main__":
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    run_finetune(config_path)