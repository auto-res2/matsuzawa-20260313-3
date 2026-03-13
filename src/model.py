"""LLM API wrapper for inference experiments."""

import os
import time
from typing import Dict, List, Optional
import openai
from openai import OpenAI


class LLMModel:
    """
    Wrapper for LLM API calls with support for OpenAI and Anthropic.
    """

    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs,
    ):
        """
        Initialize LLM model wrapper.

        Args:
            model_name: Name of the model (e.g., 'gpt-4o', 'claude-3-opus')
            provider: API provider ('openai' or 'anthropic')
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key_env: Environment variable name for API key
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # Get API key from environment
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {api_key_env}"
            )

        # Initialize client
        if provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: Input prompt
            temperature: Override temperature (optional)
            max_tokens: Override max_tokens (optional)

        Returns:
            Generated text response
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "openai":
            return self._generate_openai(prompt, temp, max_tok)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, temp, max_tok)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_openai(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **self.kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise

    def _generate_anthropic(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Generate using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **self.kwargs,
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            raise

    def batch_generate(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        delay: float = 0.1,
    ) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            temperature: Override temperature (optional)
            max_tokens: Override max_tokens (optional)
            delay: Delay between API calls in seconds

        Returns:
            List of generated responses
        """
        responses = []
        for i, prompt in enumerate(prompts):
            print(f"Generating response {i + 1}/{len(prompts)}...", end="\r")
            response = self.generate(prompt, temperature, max_tokens)
            responses.append(response)

            # Add delay to avoid rate limiting
            if i < len(prompts) - 1:
                time.sleep(delay)

        print()  # New line after progress
        return responses


def create_model_from_config(config: Dict) -> LLMModel:
    """
    Create LLM model from configuration dictionary.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        Initialized LLMModel instance
    """
    return LLMModel(
        model_name=config.get("name", "gpt-4o"),
        provider=config.get("provider", "openai"),
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 2048),
        api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
    )


if __name__ == "__main__":
    # Test model creation
    model = LLMModel(
        model_name="gpt-4o", provider="openai", temperature=0.0, max_tokens=100
    )

    response = model.generate("What is 2+2? Answer with just the number.")
    print(f"Response: {response}")
