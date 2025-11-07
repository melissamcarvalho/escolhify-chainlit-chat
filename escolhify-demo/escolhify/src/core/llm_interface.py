# -*- coding: utf-8 -*-
"""LLM Interface for connecting to various language models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    metadata: Dict[str, Any] = None


class LLMInterface(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters specific to the LLM
            
        Returns:
            LLMResponse with the generated content
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider."""
        pass


class OpenAILLM(LLMInterface):
    """OpenAI GPT implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI LLM.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            
            client = AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            content = response.choices[0].message.content
            metadata = {
                "model": self.model,
                "usage": response.usage.dict() if response.usage else None,
                "finish_reason": response.choices[0].finish_reason
            }
            
            return LLMResponse(content=content, metadata=metadata)
            
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"OpenAI ({self.model})"

class GeminiLLM(LLMInterface):
    """Google Gemini implementation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        """Initialize Gemini LLM.
        
        Args:
            api_key: Google API key (if None, will try to get from environment)
            model: Model name to use (gemini-2.5-flash, gemini-2.5-pro, etc.)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Google API key not provided and not found in environment variables (GOOGLE_API_KEY or GEMINI_API_KEY)")

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Gemini API."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            # Configure generation parameters
            generation_config = {
                "max_output_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 64)
            }
            
            # Generate response
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            
            # Handle safety blocks and empty responses
            if not response.candidates or len(response.candidates) == 0:
                raise Exception("No candidates returned from Gemini API")
            
            candidate = response.candidates[0]
            
            # Check if content was blocked for safety reasons
            if candidate.finish_reason.name in ["SAFETY", "RECITATION"]:
                safety_info = []
                if candidate.safety_ratings:
                    safety_info = [
                        f"{rating.category.name}: {rating.probability.name}"
                        for rating in candidate.safety_ratings
                        if rating.probability.name in ["HIGH", "MEDIUM"]
                    ]
                raise Exception(f"Content blocked by Gemini safety filters. Reason: {candidate.finish_reason.name}. Safety concerns: {', '.join(safety_info) if safety_info else 'General safety violation'}")
            
            # Get content safely
            if not candidate.content or not candidate.content.parts:
                raise Exception(f"Empty response from Gemini. Finish reason: {candidate.finish_reason.name}")
            
            content = ""
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    content += part.text
            
            if not content.strip():
                raise Exception("Gemini returned empty content")
            
            metadata = {
                "model": self.model,
                "finish_reason": candidate.finish_reason.name,
                "safety_ratings": [
                    {"category": rating.category.name, "probability": rating.probability.name}
                    for rating in candidate.safety_ratings
                ] if candidate.safety_ratings else []
            }
            
            return LLMResponse(content=content, metadata=metadata)
            
        except ImportError:
            raise ImportError("Google GenerativeAI package not installed. Install with: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

    def get_provider_name(self) -> str:
        """Get provider name."""
        return f"Google Gemini ({self.model})"


class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def create_llm(provider: str, **kwargs) -> LLMInterface:
        """Create an LLM instance based on provider name.
        
        Args:
            provider: Provider name ('openai', 'anthropic', 'gemini', 'ollama')
            **kwargs: Provider-specific configuration
            
        Returns:
            LLM instance
        """
        providers = {
            "openai": OpenAILLM,
            "gemini": GeminiLLM,
        }
        
        if provider.lower() not in providers:
            available = ", ".join(providers.keys())
            raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
        
        return providers[provider.lower()](**kwargs)
