import os
import logging
import yaml
import gc
import torch
import traceback
import shutil
from itertools import product
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from scripts.bigger_kandinsky_unet_prior import FinetuneModel
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from src.utils.dataset import load_dataset

def load_config(config_path):
    """Load configuration from YAML file."""
    temp_logger = logging.getLogger(__name__ + ".config_loader")
    temp_logger.propagate = False
    temp_logger.addHandler(logging.StreamHandler())
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
    """Run finetuning with k-fold cross-validation to find the best hyperparameters."""
    config_temp = load_config(config_path)
    if config_temp is None:
        return
    gradient_accumulation_steps = config_temp.get("gradient_accumulation_steps", 4)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16',
        log_with=None
    )

    logger = get_logger(__name__, log_level="DEBUG")
    if accelerator.is_main_process:
        logger.info(accelerator.state)

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
            print(f"ERROR: Failed to configure file logging: {e}")
    logger.info(accelerator.state, main_process_only=True)
    set_seed(42)

    overall_best_val_loss = -1
    best_config = None
    save_dir = "/home/iris/Documents/deep_learning/experiments/bigger_kandinsky/best_t5_kandinsky_unet_prior"

    try:
        config = load_config(config_path)
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
            'learning_rate': config.get("learning_rates", [0.001, 0.0001, 0.01, 0.00001]),
            'lora_r': config.get("lora_ranks", [4, 8, 16]),
            'apply_lora_unet': config.get("apply_lora_unets", [True]),
            'apply_lora_text_encoder': config.get("apply_lora_text_encoders", [True]),
            'epochs': config.get("hyperparam_epochs", [5, 10, 15]),
            'batch_size': config.get("hyperparam_batch_size", [1])
        }
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

        for model_name in models_to_train:
            model_base_output_dir = base_output_dir
            logger.info(f"========== Starting Pipeline for: {model_name} ==========", main_process_only=True)

            overall_best_val_loss = -1
            best_config = None

            for idx, hyperparams in enumerate(hyperparam_configs):
                config_name = f"hyperparam_config_{idx}"
                logger.info(f"--- Running Hyperparameter Config {idx+1}/{len(hyperparam_configs)} ({config_name}) ---", main_process_only=True)
                logger.info(f"Hyperparameters: {hyperparams}", main_process_only=True)

                hyperparam_dir = os.path.join(model_base_output_dir, config_name)
                if accelerator.is_main_process:
                    try:
                        os.makedirs(hyperparam_dir, exist_ok=True)
                        logger.info(f"Created temporary directory: {hyperparam_dir}")
                    except Exception as e:
                        logger.error(f"Failed to create temporary directory {hyperparam_dir}: {e}")

                finetuner = None
                try:
                    finetuner = FinetuneModel(model_name, hyperparam_dir, accelerator, logger_instance=logger)
                    finetuner.load_model()
                    finetuner.modify_architecture(
                        apply_lora_to_unet=hyperparams.get('apply_lora_unet', True),
                        apply_lora_to_text_encoder=hyperparams.get('apply_lora_text_encoder', True)
                    )
                    avg_val_loss, model_state, hyperparameters, best_epoch = finetuner.fine_tune(
                        dataset_path=dataset_path,
                        train_val_splits=train_val_splits,
                        epochs=hyperparams.get('epochs', 5),
                        batch_size=hyperparams.get('batch_size', 1),
                        learning_rate=float(hyperparams.get('learning_rate', 1e-5)),
                        gradient_accumulation_steps=gradient_accumulation_steps
                    )
                    logger.info(f"Completed run {idx+1}. Average validation loss across folds: {avg_val_loss:.4f}", main_process_only=True)

                    if accelerator.is_main_process:
                        if model_state and not np.isnan(avg_val_loss) and (overall_best_val_loss == -1 or avg_val_loss < overall_best_val_loss):
                            overall_best_val_loss = avg_val_loss
                            best_config = {
                                'index': idx,
                                'hyperparameters': hyperparams,
                                'avg_val_loss': avg_val_loss,
                                'epoch': best_epoch
                            }
                            logger.info(f"*** New global best validation loss found: {overall_best_val_loss:.4f} (Config {idx}, Epoch {best_epoch}) ***", main_process_only=True)
                            logger.info(f"Best configuration updated: {best_config}", main_process_only=True)

                            # Remove existing best model directory
                            if os.path.exists(save_dir):
                                try:
                                    shutil.rmtree(save_dir)
                                    logger.info(f"Removed previous best model directory: {save_dir}")
                                except Exception as e:
                                    logger.error(f"Failed to remove previous best model directory {save_dir}: {e}")

                            # Save new best model
                            try:
                                os.makedirs(save_dir, exist_ok=True)
                                logger.info(f"Created save directory: {save_dir}")
                                finetuner.output_dir = save_dir
                                finetuner.save_model_state(
                                    epoch=best_epoch,
                                    val_loss=avg_val_loss,
                                    hyperparameters=hyperparams
                                )
                                logger.info(f"Saved best model and hyperparameters for config {idx} with average validation loss {avg_val_loss:.4f} to {save_dir}")
                            except Exception as e:
                                logger.error(f"Failed to save best model to {save_dir}: {e}")
                        else:
                            reason = []
                            if not model_state:
                                reason.append("model_state is None")
                            if np.isnan(avg_val_loss):
                                reason.append("avg_val_loss is NaN")
                            if overall_best_val_loss != -1 and avg_val_loss >= overall_best_val_loss:
                                reason.append(f"avg_val_loss ({avg_val_loss:.4f}) not better than global best ({overall_best_val_loss:.4f})")
                            logger.info(f"Skipped saving for config {idx}: {', '.join(reason)}", main_process_only=True)

                except Exception as e:
                    logger.error(f"Run FAILED for {model_name}, {config_name}: {e}\n{traceback.format_exc()}", main_process_only=True)

                finally:
                    if os.path.exists(hyperparam_dir) and accelerator.is_main_process:
                        try:
                            shutil.rmtree(hyperparam_dir)
                            logger.info(f"Removed temporary directory: {hyperparam_dir}")
                        except Exception as e:
                            logger.error(f"Failed to remove temporary directory {hyperparam_dir}: {e}")
                    if finetuner:
                        del finetuner
                    gc.collect()
                    torch.cuda.empty_cache()

            if best_config:
                logger.info(f"Final best configuration for {model_name}: Config {best_config['index']}, Avg Val Loss: {best_config['avg_val_loss']:.4f}, Epoch: {best_config['epoch']}, Hyperparameters: {best_config['hyperparameters']}", main_process_only=True)
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