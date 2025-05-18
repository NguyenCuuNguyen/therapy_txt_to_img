import torch
from diffusers import DiffusionPipeline
from PIL import Image

class DeepFloydIFModel:
    def __init__(self, logger=None):
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-XL-v1.0",
                variant="fp16",
                torch_dtype=self.dtype,
                use_safetensors=True
            ).to(self.device)
            self.logger.info("Loaded DeepFloyd IF pipeline.")
        except Exception as e:
            self.logger.error(f"Failed to load DeepFloyd IF model: {e}")
            raise

    def generate_image(self, prompt, prompt_name, prompt_config_path, prompt_getter):
        """Generate an image from a prompt."""
        try:
            # Use prompt_getter to process the prompt
            processed_prompt = prompt_getter(prompt, prompt_name, prompt_config_path)
            # Generate image
            image = self.pipe(
                prompt=processed_prompt,
                height=64,
                width=64,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            # Resize to 64x64 if necessary
            image = image.resize((64, 64))
            return image
        except Exception as e:
            self.logger.error(f"Failed to generate image for prompt {prompt_name}: {e}")
            raise