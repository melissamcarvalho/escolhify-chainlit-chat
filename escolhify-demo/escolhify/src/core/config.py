# -*- coding: utf-8 -*-
"""Configuration management for Escolhify."""

import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "openai"  # Default to OpenAI
    model: str = "gpt-5"
    api_key: Optional[str] = None
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create LLM config from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        # Default configurations for different providers
        configs = {
            "gemini": {
                "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                "api_key": os.getenv("GEMINI_API_KEY"),
            },
            "openai": {
                "model": os.getenv("OPENAI_MODEL", "gpt-5"),
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
        }
        
        config_data = configs.get(provider, configs["openai"])
        
        return cls(
            provider=provider,
            model=config_data.get("model"),
            api_key=config_data.get("api_key"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0"))
        )


@dataclass 
class EscolhifyConfig:
    """Main Escolhify configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    
    # MongoDB settings for collaborative filtering
    mongodb_uri: Optional[str] = None
    mongodb_database: str = "amazon_db"
    mongodb_collection: str = "product_reviews"
    
    # Logging settings
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    
    @classmethod
    def load(cls) -> 'EscolhifyConfig':
        """Load configuration from environment."""
        return cls(
            llm=LLMConfig.from_env(),
            mongodb_uri=os.getenv("MONGODB_URI"),
            mongodb_database=os.getenv("MONGODB_DATABASE", "amazon_db"),
            mongodb_collection=os.getenv("MONGODB_COLLECTION", "product_reviews"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO").upper()
        )


@dataclass
class AppConfig:
    """Application configuration - alias for backwards compatibility."""
    
    @classmethod
    def load(cls) -> EscolhifyConfig:
        """Load configuration from environment."""
        return EscolhifyConfig.load()
