import os
import torch
from diffusers import StableDiffusionPipeline

class AIEngine:
    """
    Handles loading the local Stable Diffusion model and running inference.
    """
    def __init__(self):
        self.model_path = "./models/v1-5-pruned-emaonly.safetensors"
        self.pipeline = None
        self.is_loaded = False

    def load_model(self):
        """
        Loads the Stable Diffusion pipeline from the local model file.
        """
        if not os.path.exists(self.model_path):
            print(f"AI Model not found at {self.model_path}. Please follow the setup instructions.")
            return False
        if self.is_loaded:
            return True
        try:
            print("Loading AI model... This may take a moment.")
            self.pipeline = StableDiffusionPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            self.pipeline.to("cuda")
            self.is_loaded = True
            print("AI Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load AI model: {e}")
            # This can happen if CUDA is not available or if the model file is corrupt.
            return False

    def generate_images(self, prompt, num_images=4):
        """
        Generates images using the loaded Stable Diffusion model.
        Returns a list of PIL Images.
        """
        if not self.is_loaded:
            print("Cannot generate images, AI model is not loaded.")
            return []
        print(f"Generating {num_images} images with prompt: {prompt}")
        images = self.pipeline(prompt, num_images_per_prompt=num_images).images
        print("Image generation complete.")
        return images