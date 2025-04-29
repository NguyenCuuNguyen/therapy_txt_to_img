import yaml
import os

def load_prompt_config(config_path):
    """Load prompt configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_description_prompt(txt_prompt, prompt_name, config_path):
    """Return the description experiment prompt."""
    config = load_prompt_config(config_path)
    prompts = config['description_exp']['prompts']
    for prompt in prompts:
        if prompt_name in prompt:
            return prompt[prompt_name].format(txt_prompt=txt_prompt)
    raise ValueError(f"Prompt {prompt_name} not found in description_exp prompts")