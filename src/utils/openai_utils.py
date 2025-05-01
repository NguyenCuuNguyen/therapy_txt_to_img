import os
import pickle
import ast
import requests
from openai import OpenAI
import pandas as pd
import numpy as np
import logging
import yaml
import re
import traceback
from transformers import CLIPTokenizer

# Configure basic logging if run standalone or imported early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
class OpenAIUtils:
    """Utility class for OpenAI API interactions, including summarization."""

    def __init__(self, api_key, logger=None):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=api_key)
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("OpenAIUtils initialized.")

    def load_superset_topics(self):
        """Load 34 superset topics from pickle file."""
        try:
            with open(self.superset_file, "rb") as file:
                set_topic = pickle.load(file)
            return set_topic.keys()
        except Exception as e:
            self.logger.error(f"Failed to load superset topics: {e}")
            raise

    def clean_response(self, response_text):
        """Clean the API response to ensure it's a valid Python dictionary string."""
        # Remove code block markers (e.g., ```python, ```)
        response_text = re.sub(r'```(?:python)?\s*', '', response_text)
        # Remove leading/trailing whitespace
        response_text = response_text.strip()
        # Ensure the response starts with { and ends with }
        if not response_text.startswith('{') or not response_text.endswith('}'):
            self.logger.error(f"Invalid response format: {response_text}")
            raise ValueError("Response is not a valid dictionary string")
        return response_text

    def generate_gpt_description(self, transcript, superset_topics):
        """Generate a dictionary of topic descriptions using GPT-4o-mini."""
        prompt = (
            "You are a helpful assistant that reads a therapy transcript and returns a Python dictionary. "
            "Return ONLY the dictionary as a string, with no additional text, explanations, or formatting (e.g., no code blocks). "
            "The dictionary keys are the topics provided, and values are brief descriptions of relevant topics or 'null' if not relevant. "
            "Surround all string values with double quotes to ensure compatibility with ast.literal_eval. "
            "Example: {\"topic1\": \"description\", \"topic2\": \"null\"}"
        )
        query = (
            f"Given the therapy transcript: {transcript} and the set of topics: {superset_topics}, "
            f"return a Python dictionary with keys as the topics and values as brief descriptions or 'null' if not relevant. "
            f"Output only the dictionary string, compatible with ast.literal_eval, with no extra text or formatting."
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            response_text = response.choices[0].message.content
            cleaned_response = self.clean_response(response_text)
            return ast.literal_eval(cleaned_response)
        except Exception as e:
            self.logger.error(f"Failed to generate or parse GPT description: {e}\nResponse: {response_text}")
            raise

    def create_dalle_image(self, full_prompt, output_path, size="1024x1024", quality="standard"):
        """Generate and save a DALL-E image, ensuring directory exists."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=full_prompt,
                size=size,
                quality=quality,
                style="vivid",
                n=1,
            )
            img_data = requests.get(response.data[0].url).content
            with open(output_path, 'wb') as handler:
                handler.write(img_data)
            self.logger.info(f"Image saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to create image at {output_path}: {e}")
            raise

    # !--- New Method for Summarization ---!
    def summarize_text_with_openai(self, text_to_summarize: str, target_max_tokens: int, model: str = "gpt-4o-mini"):
        """
        Summarizes the input text using the specified OpenAI model to be
        around the target token length, focusing on key themes and details.

        Args:
            text_to_summarize (str): The long text prompt to summarize.
            target_max_tokens (int): The desired maximum token count for the summary (e.g., 75 for CLIP).
            model (str): The OpenAI model to use for summarization.

        Returns:
            str: The summarized text, or the original text if summarization fails or is not needed.
        """
        self.logger.info(f"Requesting summarization for text (first 100 chars): {text_to_summarize[:100]}...")
        self.logger.debug(f"Target max tokens for summary: {target_max_tokens}")

        # Estimate word count for summary length hint (very approximate)
        # Aim for slightly fewer tokens than the max to be safe
        estimated_word_count = int(target_max_tokens * 0.6) # Rough estimate

        system_prompt = (
            "You are an expert text summarizer. Your goal is to condense the provided text while preserving "
            "the core subject, key descriptive details, style, and any specific instructions for image generation. "
            "Avoid adding any conversational text or explanations. Output only the summarized text."
        )
        user_prompt = (
            f"Summarize the following text to be concise, ideally under {target_max_tokens} tokens (around {estimated_word_count} words), "
            f"while retaining the main subject, essential details, style, and instructions. Text:\n\n"
            f"\"{text_to_summarize}\""
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2, # Lower temperature for more focused summary
                max_tokens=target_max_tokens + 50 # Give OpenAI some buffer, we check length later
            )
            summary = response.choices[0].message.content.strip()
            self.logger.info(f"OpenAI Summary received (first 100 chars): {summary[:100]}...")
            return summary
        except Exception as e:
            self.logger.error(f"OpenAI summarization API call failed: {e}\n{traceback.format_exc()}")
            # Fallback to returning original text if API fails
            return text_to_summarize

# Example usage (optional, for testing the class directly)
if __name__ == '__main__':
    # Example: Load API key from environment variable or a config file
    # Make sure to replace this with your actual key loading mechanism
    try:
        # Attempt to load from environment variable first
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
             # Fallback: attempt to load from a config file
             print("Loading API key from config file...")
             config = load_config("/home/iris/Documents/deep_learning/config/config.yaml") # Adjust path if needed
             api_key = config.get("openai_api_key")

        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables or config file.")

        openai_util = OpenAIUtils(api_key=api_key)

        long_text = "Generate an image depicting a vast, serene mountain landscape at dawn. The peaks should be snow-capped, reflecting the soft, warm light of the rising sun. A winding river should flow through the valley below, its surface like glass. Include a small, solitary cabin nestled amongst pine trees near the riverbank. The overall mood should be peaceful, awe-inspiring, and slightly mystical. Ensure high detail, sharp focus, and photorealistic quality, 8k resolution, masterpiece."
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14") # Example tokenizer
        clip_limit = 77

        summarized = openai_util.summarize_text_with_openai(long_text, clip_limit - 2, clip_tokenizer) # Pass tokenizer

        print("\nOriginal Text:")
        print(long_text)
        print(f"\nToken count (CLIP): {len(clip_tokenizer(long_text)['input_ids'])}")

        print("\nSummarized Text:")
        print(summarized)
        print(f"\nToken count (CLIP): {len(clip_tokenizer(summarized)['input_ids'])}")

    except Exception as main_e:
        print(f"An error occurred: {main_e}")

