import os
import yaml
import logging
import torch
import gc
from multiprocessing import Pool
from src.utils.file_utils import FileUtils
from src.models.sdxl import SDXLModel
from src.models.kandinsky import KandinskyModel
from src.models.karlo import KarloModel
from src.models.deepfloyd_if import DeepFloydIFModel
from src.prompts.theory_prompts import get_description_prompt

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_prompt(args):
    """Process a single prompt for a given model."""
    model, prompt, prompt_name, filename, image_dir, prompt_config_path, prompt_getter = args
    try:
        image_path = os.path.join(image_dir, filename, f"{prompt_name}.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = model.generate_image(prompt, prompt_name, prompt_config_path, prompt_getter)
        image.save(image_path)
        logging.info(f"Saved image to {image_path}")
        del image
        gc.collect()
        torch.cuda.empty_cache()
        return image_path
    except Exception as e:
        logging.error(f"Failed to process prompt {prompt_name} for {filename}: {e}")
        return None

def run_model(model, filenames, descriptions, image_dir, prompt_config_path, prompt_getter, max_workers=2):
    """Run a model on all prompts in parallel."""
    config = load_config(prompt_config_path)
    experiment = os.path.basename(image_dir)
    prompt_list = [list(prompt.keys())[0] for prompt in config.get(experiment, {}).get('prompts', [])]
    
    tasks = []
    for filename, desc in zip(filenames, descriptions):
        for prompt_name in prompt_list:
            prompt = str(desc)  # Convert description to string
            tasks.append((model, prompt, prompt_name, filename, image_dir, prompt_config_path, prompt_getter))
    
    with Pool(processes=max_workers) as pool:
        results = pool.map(process_prompt, tasks)
    
    return results

if __name__ == "__main__":
    config_path = "config/config.yaml"
    prompt_config_path = "config/prompt_config.yaml"
    config = load_config(config_path)
    
    logging.basicConfig(
        filename=os.path.join(config['project']['base_path'], config['project']['log_dir'], 'experiment.log'),
        level=logging.INFO,
        format='%(asctime)s %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    file_utils = FileUtils(logger=logger)
    csv_path = os.path.join(config['project']['base_path'], config['project']['input_csv_dir'], 'FILE_SUPERTOPIC_DESCRIPTION.csv')
    
    # Load filenames and descriptions
    filenames, descriptions = file_utils.load_description_csv(csv_path)
    
    # Define models
    models = [
        ("sdxl", SDXLModel, get_description_prompt),
        ("kandinsky", KandinskyModel, get_description_prompt),
        ("karlo", KarloModel, get_description_prompt),
        ("deepfloyd_if", DeepFloydIFModel, get_description_prompt)
    ]
    
    # Run each model sequentially
    for model_name, model_class, prompt_getter in models:
        logger.info(f"Running model: {model_name}")
        image_dir = os.path.join(config['project']['base_path'], config['project']['image_dir'], model_name)
        try:
            model = model_class(logger=logger)
            run_model(model, filenames, descriptions, image_dir, prompt_config_path, prompt_getter, max_workers=2)
        except Exception as e:
            logger.error(f"Failed to run model {model_name}: {e}")
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


