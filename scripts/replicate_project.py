import os
import subprocess
import logging
import sys
import shutil
import urllib.request
import zipfile
from pathlib import Path
import yaml
import json

# Configure logging
log_dir = "/home/iris/Documents/deep_learning/src/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "replicate_project.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
logger = logging.getLogger(__name__)

# Console handler for logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Project root directory
PROJECT_ROOT = "/home/iris/Documents/deep_learning"
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

# List of scripts to run in order
SCRIPTS = [
    "collect_coco_data.py",
    "finetune_model.py",
    "run_finetune_multiple_epochs.py",
    "generate_images_from_finetuned.py",
    "train_sdxl_base_refiner.py",
    "generate_img_sdxl_t5_no_refiner.py",
    "sdxl_fullscaled_chunk.py"
]

# Dependencies for requirements.txt
DEPENDENCIES = [
    "torch==2.1.1+cu121",
    "torchvision==0.16.0+cu121",
    "diffusers==0.29.2",
    "transformers==4.41.0",
    "xformers==0.0.23",
    "pandas",
    "numpy",
    "pillow",
    "psutil",
    "pyyaml",
    "peft",
    "accelerate",
    "pycocotools",
    "bitsandbytes",
    "openai"
]

def check_and_create_dirs():
    """Create necessary directories if they don't exist."""
    dirs = [PROJECT_ROOT, SCRIPTS_DIR, DATA_DIR, CONFIG_DIR, log_dir]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Ensured directory exists: {d}")

def setup_environment():
    """Set up Conda environment and install dependencies."""
    env_name = "dl2"
    logger.info("Setting up Conda environment...")
    
    # Check if environment exists
    result = subprocess.run(f"conda env list | grep {env_name}", shell=True, capture_output=True, text=True)
    if env_name not in result.stdout:
        logger.info(f"Creating Conda environment: {env_name}")
        subprocess.run(f"conda create -n {env_name} python=3.10 -y", shell=True, check=True)

    # Generate requirements.txt
    requirements_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    with open(requirements_path, "w") as f:
        for dep in DEPENDENCIES:
            f.write(f"{dep}\n")
    logger.info(f"Created requirements.txt at {requirements_path}")

    # Install dependencies
    # logger.info("Installing dependencies...")
    # activate_cmd = f"conda run -n {env_name} pip install -r {requirements_path}"
    # try:
    #     subprocess.run(activate_cmd, shell=True, check=True)
    #     logger.info("Dependencies installed successfully.")
    # except subprocess.CalledProcessError as e:
    #     logger.error(f"Failed to install dependencies: {e}")
    #     sys.exit(1)

def download_coco_dataset():
    """Download and extract COCO 2017 dataset if not already present."""
    coco_dir = os.path.join(DATA_DIR, "finetune_dataset", "coco")
    os.makedirs(coco_dir, exist_ok=True)

    # URLs for COCO 2017 dataset
    coco_urls = {
        "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }

    for file_name, url in coco_urls.items():
        file_path = os.path.join(coco_dir, file_name)
        if not os.path.exists(file_path):
            logger.info(f"Downloading {file_name}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                logger.info(f"Downloaded {file_name}")
            except Exception as e:
                logger.error(f"Failed to download {file_name}: {e}")
                sys.exit(1)

        # Extract zip file
        extract_dir = coco_dir
        if not os.path.exists(os.path.join(extract_dir, file_name.replace(".zip", ""))):
            logger.info(f"Extracting {file_name}...")
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                logger.info(f"Extracted {file_name}")
            except Exception as e:
                logger.error(f"Failed to extract {file_name}: {e}")
                sys.exit(1)

def check_prerequisites():
    """Check for required files and directories."""
    required_files = [
        os.path.join(CONFIG_DIR, "config.yaml"),
        os.path.join(CONFIG_DIR, "prompt_config.yaml"),
        os.path.join(DATA_DIR, "sample_list.txt"),
        os.path.join(DATA_DIR, "input_csv", "FILE_SUPERTOPIC_DESCRIPTION.csv")
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.warning(f"Required file not found: {file_path}. Some scripts may fail.")
    
    # Create dummy config.yaml if not present
    config_path = os.path.join(CONFIG_DIR, "config.yaml")
    if not os.path.exists(config_path):
        logger.info(f"Creating default config.yaml at {config_path}")
        config = {
            "openai": {"api_key": "your_openai_api_key_here"},
            "base_output_dir": "/home/iris/Documents/deep_learning/experiments/custom_finetuned",
            "dataset_path": "/home/iris/Documents/deep_learning/data/finetune_dataset/coco/dataset.json",
            "base_output_dir_sdxl": "/home/iris/Documents/deep_learning/experiments/trained_sdxl_t5_refiner",
            "models_to_generate": ["sdxl", "kandinsky"]
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        logger.warning("Please update config.yaml with your OpenAI API key.")

    # Create dummy hyperparameter_performance_summary_copy.json if not present
    hyperparam_path = os.path.join(PROJECT_ROOT, "experiments", "sdxl_t5_refiner", "hyperparameter_performance_summary_copy.json")
    os.makedirs(os.path.dirname(hyperparam_path), exist_ok=True)
    if not os.path.exists(hyperparam_path):
        logger.info(f"Creating default hyperparameter_performance_summary_copy.json at {hyperparam_path}")
        hyperparam_config = [{
            "config_idx": 0,
            "config_name": "hyperparam_config_0",
            "hyperparameters": {
                "learning_rate": 1e-7,
                "lora_r": 8,
                "apply_lora_unet": True,
                "apply_lora_refiner": True,
                "apply_lora_text_encoder": True,
                "epochs": 20,
                "batch_size": 1,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "validation_split": 0.2,
                "early_stopping_patience": 5,
                "generation_frequency": 1,
                "perceptual_loss_weight": 0.1
            },
            "avg_kfold_val_loss": 0.0
        }]
        with open(hyperparam_path, "w") as f:
            json.dump(hyperparam_config, f, indent=4)

def run_script(script_name):
    """Run a single script and handle errors."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        return False

    logger.info(f"Running script: {script_name}")
    env_name = "dl2"
    cmd = f"conda run -n {env_name} python {script_path}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Successfully ran {script_name}")
            logger.debug(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"Failed to run {script_name}. Return code: {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Exception while running {script_name}: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    """Main function to orchestrate script execution."""
    logger.info("Starting project replication...")
    
    # Create necessary directories
    check_and_create_dirs()

    # Set up environment and install dependencies
    setup_environment()

    # Download COCO dataset for collect_coco_data.py
    download_coco_dataset()

    # Check for required files
    check_prerequisites()

    # Run scripts in sequence
    for script in SCRIPTS:
        success = run_script(script)
        if not success:
            logger.error(f"Stopping execution due to failure in {script}")
            sys.exit(1)

    logger.info("Project replication completed successfully.")

if __name__ == "__main__":
    main()