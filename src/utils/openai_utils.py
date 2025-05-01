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
from transformers import T5Tokenizer  # Use T5 tokenizer for consistency

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

    def summarize_text_with_openai(self, text_to_summarize: str, target_max_tokens: int, tokenizer: T5Tokenizer, model: str = "gpt-4o-mini"):
        """
        Summarizes the input text using the specified OpenAI model to fit precisely within the target token count,
        preserving key themes and details for image generation.

        Args:
            text_to_summarize (str): The long text prompt to summarize.
            target_max_tokens (int): The desired maximum token count for the summary (e.g., 510 for T5-base).
            tokenizer (T5Tokenizer): The tokenizer to count tokens (T5-base tokenizer).
            model (str): The OpenAI model to use for summarization (default: "gpt-4o-mini").

        Returns:
            str: The summarized text, guaranteed to be within target_max_tokens.
        """
        self.logger.info(f"Summarizing text (first 100 chars): {text_to_summarize[:100]}...")
        self.logger.debug(f"Target max tokens: {target_max_tokens}")

        # Count initial tokens
        initial_token_ids = tokenizer(text_to_summarize, max_length=target_max_tokens + 100, truncation=False)["input_ids"]
        initial_token_count = len(initial_token_ids)
        self.logger.debug(f"Initial token count: {initial_token_count}")

        if initial_token_count <= target_max_tokens:
            self.logger.info("Text is already within token limit. No summarization needed.")
            return text_to_summarize

        # Estimate word count for initial summary (approximate, refined later)
        estimated_word_count = int(target_max_tokens * 0.6)  # Rough estimate, assuming ~1.67 words per token

        system_prompt = (
            "You are an expert text summarizer. Your goal is to condense the provided text while preserving "
            "the core subject, key descriptive details, style, and any specific instructions for image generation. "
            "Output only the summarized text, with no additional explanations or formatting."
        )
        user_prompt = (
            f"Summarize the following text to be concise, targeting around {estimated_word_count} words to fit within "
            f"{target_max_tokens} tokens, retaining the main subject, essential details, style, and instructions. Prioritize the distinctive topics dictionary given in the text to capture its key important descriptive details."
            f"Text:\n\n\"{text_to_summarize}\""
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        max_attempts = 3
        current_max_tokens = target_max_tokens + 20  # Initial buffer
        summary = None

        for attempt in range(max_attempts):
            try:
                self.logger.debug(f"Summarization attempt {attempt + 1}/{max_attempts} with max_tokens={current_max_tokens}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,  # Low temperature for focused summary
                    max_tokens=current_max_tokens
                )
                summary = response.choices[0].message.content.strip()
                summary_token_ids = tokenizer(summary, max_length=target_max_tokens + 100, truncation=False)["input_ids"]
                summary_token_count = len(summary_token_ids)
                self.logger.debug(f"Summary token count: {summary_token_count}")

                if summary_token_count <= target_max_tokens:
                    self.logger.info(f"Summary fits within {target_max_tokens} tokens ({summary_token_count} tokens).")
                    return summary

                # If summary is too long, reduce max_tokens and tighten prompt
                self.logger.warning(f"Summary exceeds {target_max_tokens} tokens ({summary_token_count} tokens). Retrying...")
                current_max_tokens = max(target_max_tokens - 10, target_max_tokens // 2)
                estimated_word_count = int(current_max_tokens * 0.5)
                user_prompt = (
                    f"Summarize the following text to be very concise, targeting around {estimated_word_count} words to fit strictly within "
                    f"{target_max_tokens} tokens, retaining only the most essential subject, details, style, and instructions. "
                    f"Text:\n\n\"{text_to_summarize}\""
                )
                messages[1]["content"] = user_prompt

            except Exception as e:
                self.logger.error(f"OpenAI summarization attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    # On final attempt, fallback to truncation
                    self.logger.error("Max summarization attempts reached. Falling back to truncation.")
                    truncated_summary = tokenizer.decode(
                        initial_token_ids[:target_max_tokens],
                        skip_special_tokens=True
                    )
                    self.logger.info(f"Truncated summary: '{truncated_summary[:100]}...' ({len(tokenizer(truncated_summary)['input_ids'])} tokens)")
                    return truncated_summary
                continue

        # If all attempts fail, truncate the original text
        self.logger.error("All summarization attempts failed. Falling back to truncation.")
        truncated_summary = tokenizer.decode(
            initial_token_ids[:target_max_tokens],
            skip_special_tokens=True
        )
        self.logger.info(f"Truncated summary: '{truncated_summary[:100]}...' ({len(tokenizer(truncated_summary)['input_ids'])} tokens)")
        return truncated_summary

# Example usage (optional, for testing the class directly)
if __name__ == '__main__':
    try:
        # Load API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            config = load_config("/home/iris/Documents/deep_learning/config/config.yaml")
            api_key = config.get("openai", {}).get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables or config file.")

        openai_util = OpenAIUtils(api_key=api_key)
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

        long_text = "Generate an image depicting a vast, serene mountain landscape at dawn. The peaks should be snow-capped, reflecting the soft, warm light of the rising sun. A winding river should flow through the valley below, its surface like glass. Include a small, solitary cabin nestled amongst pine trees near the riverbank. The overall mood should be peaceful, awe-inspiring, and slightly mystical. Ensure high detail, sharp focus, and photorealistic quality, 8k resolution, masterpiece."
        target_max_tokens = 510

        summarized = openai_util.summarize_text_with_openai(
            text_to_summarize=long_text,
            target_max_tokens=target_max_tokens,
            tokenizer=tokenizer
        )

        print("\nOriginal Text:")
        print(long_text)
        print(f"\nToken count (T5): {len(tokenizer(long_text)['input_ids'])}")

        print("\nSummarized Text:")
        print(summarized)
        print(f"\nToken count (T5): {len(tokenizer(summarized)['input_ids'])}")

    except Exception as main_e:
        print(f"An error occurred: {main_e}")