import torch
from diffusers import AutoPipelineForText2Image
import gc
import logging

class SDXLModel:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.model = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        self.model.enable_xformers_memory_efficient_attention()
        self.generator = torch.Generator(device="cuda").manual_seed(31)

    def generate_image(self, prompt, prompt_name, prompt_config_path, prompt_getter):
        full_prompt = prompt_getter(prompt, prompt_name, prompt_config_path)
        image = self.model(full_prompt, generator=self.generator).images[0]
        return image

    def __del__(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()