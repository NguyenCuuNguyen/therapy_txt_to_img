import torch
from diffusers import DiffusionPipeline
from PIL import Image

class KandinskyModel:
    def __init__(self, logger=None):
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder",
                torch_dtype=self.dtype,
                use_safetensors=True
            ).to(self.device)
            self.logger.info("Loaded Kandinsky 2.2 pipeline.")
        except Exception as e:
            self.logger.error(f"Failed to load Kandinsky model: {e}")
            raise

    def generate_image(self, prompt, prompt_name, prompt_config_path, prompt_getter):
        """Generate an image from a prompt."""
        try:
            # Use prompt_getter to process the prompt
            processed_prompt = prompt_getter(prompt, prompt_name, prompt_config_path)
            # Generate image
            image = self.pipe(
                prompt=processed_prompt,
                height=512,
                width=512,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            return image
        except Exception as e:
            self.logger.error(f"Failed to generate image for prompt {prompt_name}: {e}")
            raise