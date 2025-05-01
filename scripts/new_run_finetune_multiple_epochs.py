import os
import logging
import yaml
import json
import gc
import torch
import traceback
from new_finetune_model import FinetuneModel

# Configure logging
logging.basicConfig(
    filename='/home/iris/Documents/deep_learning/src/logs/new_finetune_multiple_epochs.log',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s',
    filemode='w',
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_finetune(config_path):
    """Run finetuning for multiple models, epochs, and hyperparameters."""
    logger.info("Starting multi-epoch finetuning script.")
    try:
        config = load_config(config_path)
        dataset_path = config.get("dataset_path", "/home/iris/Documents/deep_learning/data/finetune_dataset/coco/dataset.json")
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset JSON file not found at: {dataset_path}")
            return
        
        base_output_dir = "/home/iris/Documents/deep_learning/experiments/new_finetune"
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Define models to train
        models_to_train = [
            ("deepfloyd_if", os.path.join(base_output_dir, "deepfloyd_if"))
        ]
        
        # Hyperparameter configurations
        hyperparam_configs = [
            {
                'epochs': 5,
                'batch_size': 1,
                'learning_rate': 1e-5,
                'val_split': 0.2,
                'apply_lora_to_text_encoder': True,
                'apply_lora_to_unet': True,
                'gradient_accumulation_steps': 4
            },
            {
                'epochs': 5,
                'batch_size': 1,
                'learning_rate': 5e-6,
                'val_split': 0.2,
                'apply_lora_to_text_encoder': True,
                'apply_lora_to_unet': True,
                'gradient_accumulation_steps': 4
            }
        ]
        
        for model_name, output_dir in models_to_train:
            os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
            logger.info(f"========== Starting Pipeline for: {model_name} ==========")
            for idx, hyperparams in enumerate(hyperparam_configs):
                logger.info(f"--- Hyperparameter Config {idx+1}/{len(hyperparam_configs)} ---")
                hyperparam_dir = os.path.join(output_dir, f"hyperparam_config_{idx}")
                os.makedirs(hyperparam_dir, exist_ok=True)
                finetuner = None
                try:
                    finetuner = FinetuneModel(model_name, hyperparam_dir, logger_instance=logger)
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
                        gradient_accumulation_steps=hyperparams['gradient_accumulation_steps']
                    )
                    logger.info(f"Completed finetuning for {model_name} with hyperparameter config {idx+1}.")
                except Exception as e:
                    logger.error(f"Finetuning failed for {model_name}, config {idx+1}: {e}\n{traceback.format_exc()}")
                finally:
                    if finetuner:
                        try: del finetuner.pipeline
                        except AttributeError: pass
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
                        try: del finetuner.prior
                        except AttributeError: pass
                        try: del finetuner.decoder
                        except AttributeError: pass
                        try: del finetuner.image_processor
                        except AttributeError: pass
                        del finetuner
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info(f"Cleaned up resources for {model_name}, config {idx+1}.")
            logger.info(f"========== Finished Pipeline for: {model_name} ==========\n")
    except Exception as e:
        logger.critical(f"Main execution failed: {e}\n{traceback.format_exc()}")
    logger.info("Multi-epoch finetuning script completed.")

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
    run_finetune(config_path)

# import os
# import logging
# import yaml
# import json
# import gc
# import torch
# import traceback
# from new_finetune_model import FinetuneModel

# # Configure logging
# logging.basicConfig(
#     filename='/home/iris/Documents/deep_learning/src/logs/new_finetune_multiple_epochs.log',
#     level=logging.DEBUG,
#     format='%(asctime)s %(message)s',
#     filemode='w',
# )
# logger = logging.getLogger(__name__)

# def load_config(config_path):
#     """Load configuration from YAML file."""
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)

# def run_finetune(config_path):
#     """Run finetuning for multiple models, epochs, and hyperparameters."""
#     logger.info("Starting multi-epoch finetuning script.")
#     try:
#         config = load_config(config_path)
#         dataset_path = config.get("dataset_path", "/home/iris/Documents/deep_learning/data/finetune_dataset/coco/dataset.json")
#         if not os.path.exists(dataset_path):
#             logger.error(f"Dataset JSON file not found at: {dataset_path}")
#             return
        
#         base_output_dir = "/home/iris/Documents/deep_learning/experiments/new_finetune"
#         os.makedirs(base_output_dir, exist_ok=True)
        
#         # Define models to train
#         models_to_train = [
#             # ("sdxl", os.path.join(base_output_dir, "sdxl")),
#             # ("kandinsky", os.path.join(base_output_dir, "kandinsky")),
#             # ("karlo", os.path.join(base_output_dir, "karlo")),
#             ("deepfloyd_if", os.path.join(base_output_dir, "deepfloyd_if"))
#         ]
        
#         # Hyperparameter configurations
#         hyperparam_configs = [
#             {
#                 'epochs': 5,
#                 'batch_size': 1,
#                 'learning_rate': 1e-5,
#                 'val_split': 0.2,
#                 'apply_lora_to_text_encoder': True,
#                 'apply_lora_to_unet': True
#             },
#             {
#                 'epochs': 5,
#                 'batch_size': 1,
#                 'learning_rate': 5e-6,
#                 'val_split': 0.2,
#                 'apply_lora_to_text_encoder': True,
#                 'apply_lora_to_unet': True
#             }
#         ]
        
#         for model_name, output_dir in models_to_train:
#             os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
#             logger.info(f"========== Starting Pipeline for: {model_name} ==========")
#             for idx, hyperparams in enumerate(hyperparam_configs):
#                 logger.info(f"--- Hyperparameter Config {idx+1}/{len(hyperparam_configs)} ---")
#                 hyperparam_dir = os.path.join(output_dir, f"hyperparam_config_{idx}")
#                 os.makedirs(hyperparam_dir, exist_ok=True)
#                 finetuner = None
#                 try:
#                     finetuner = FinetuneModel(model_name, hyperparam_dir, logger_instance=logger)
#                     finetuner.load_model()
#                     finetuner.modify_architecture(
#                         apply_lora_to_text_encoder=hyperparams['apply_lora_to_text_encoder'],
#                         apply_lora_to_unet=hyperparams['apply_lora_to_unet']
#                     )
#                     finetuner.fine_tune(
#                         dataset_path=dataset_path,
#                         epochs=hyperparams['epochs'],
#                         batch_size=hyperparams['batch_size'],
#                         learning_rate=hyperparams['learning_rate'],
#                         val_split=hyperparams['val_split']
#                     )
#                     logger.info(f"Completed finetuning for {model_name} with hyperparameter config {idx+1}.")
#                 except Exception as e:
#                     logger.error(f"Finetuning failed for {model_name}, config {idx+1}: {e}\n{traceback.format_exc()}")
#                 finally:
#                     if finetuner:
#                         try: del finetuner.pipeline
#                         except AttributeError: pass
#                         try: del finetuner.model
#                         except AttributeError: pass
#                         try: del finetuner.tokenizer
#                         except AttributeError: pass
#                         try: del finetuner.tokenizer_2
#                         except AttributeError: pass
#                         try: del finetuner.text_encoder
#                         except AttributeError: pass
#                         try: del finetuner.text_encoder_2
#                         except AttributeError: pass
#                         try: del finetuner.unet
#                         except AttributeError: pass
#                         try: del finetuner.scheduler
#                         except AttributeError: pass
#                         try: del finetuner.vae
#                         except AttributeError: pass
#                         try: del finetuner.prior
#                         except AttributeError: pass
#                         try: del finetuner.decoder
#                         except AttributeError: pass
#                         try: del finetuner.image_processor
#                         except AttributeError: pass
#                         del finetuner
#                     gc.collect()
#                     if torch.cuda.is_available():
#                         torch.cuda.empty_cache()
#                     logger.info(f"Cleaned up resources for {model_name}, config {idx+1}.")
#             logger.info(f"========== Finished Pipeline for: {model_name} ==========\n")
#     except Exception as e:
#         logger.critical(f"Main execution failed: {e}\n{traceback.format_exc()}")
#     logger.info("Multi-epoch finetuning script completed.")

# if __name__ == "__main__":
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#     config_path = "/home/iris/Documents/deep_learning/config/config.yaml"
#     run_finetune(config_path)