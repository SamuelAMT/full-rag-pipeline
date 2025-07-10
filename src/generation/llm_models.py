"""
LLM models for text generation in the RAG pipeline.
Uses llama-cpp-python for efficient GGUF model inference.
"""

import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
from functools import lru_cache
import os
from pathlib import Path

from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_chat_format import Llama3ChatFormatter

from src.core.config import get_settings
from src.monitoring.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class LLMModel:
    """Base class for LLM models."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        raise NotImplementedError

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate text from prompt with streaming."""
        raise NotImplementedError


class LlamaCppModel(LLMModel):
    """LlamaCpp model implementation for GGUF models."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int = None,
        verbose: bool = False
    ):
        super().__init__(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads or os.cpu_count()
        self.verbose = verbose
        self.chat_formatter = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize LlamaCpp model."""
        try:
            logger.info(f"Loading LlamaCpp model: {self.model_path}")

            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose,
                chat_format="llama-3"
            )

            self.chat_formatter = Llama3ChatFormatter()

            logger.info("LlamaCpp model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LlamaCpp model: {e}")
            raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Llama 3 chat format."""
        try:
            formatted = self.chat_formatter.format_messages(messages)
            return formatted
        except Exception as e:
            logger.error(f"Error formatting messages: {e}")
            return self._simple_format_messages(messages)

    def _simple_format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Simple message formatting as fallback."""
        formatted_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                formatted_parts.append(f"<|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>")
            elif role == "user":
                formatted_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>")
            elif role == "assistant":
                formatted_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>")

        # Add assistant start token for generation
        formatted_parts.append("<|start_header_id|>assistant<|end_header_id|>\n")

        return "".join(formatted_parts)

    async def generate(
        self,
        prompt: str,
        system_message: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        if not self.model:
            raise ValueError("Model not initialized")

        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            formatted_prompt = self._format_messages(messages)

            # Generate in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _generate():
                return self.model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    echo=False,
                    **kwargs
                )

            result = await loop.run_in_executor(None, _generate)

            generated_text = result["choices"][0]["text"].strip()

            metrics = get_metrics_collector()
            metrics.record_tokens_generated(result["usage"]["completion_tokens"])

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_message: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text from prompt with streaming."""
        if not self.model:
            raise ValueError("Model not initialized")

        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            formatted_prompt = self._format_messages(messages)

            loop = asyncio.get_event_loop()

            def _generate_stream():
                return self.model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    echo=False,
                    stream=True,
                    **kwargs
                )

            stream = await loop.run_in_executor(None, _generate_stream)

            total_tokens = 0
            for chunk in stream:
                if chunk["choices"][0]["finish_reason"] is None:
                    token = chunk["choices"][0]["text"]
                    total_tokens += 1
                    yield token

            metrics = get_metrics_collector()
            metrics.record_tokens_generated(total_tokens)

        except Exception as e:
            logger.error(f"Error streaming text: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.model:
            return {"status": "not_initialized"}

        return {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "vocab_size": self.model.n_vocab(),
            "context_size": self.model.n_ctx(),
        }


class LLMManager:
    """Manager for LLM operations."""

    def __init__(self, model_path: str = None):
        self.settings = get_settings()
        self.model_path = model_path or os.getenv("LLM_MODEL_PATH", self.settings.LLM_MODEL_PATH)
        self.llm_model = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM based on configuration."""
        try:
            if not self.model_path:
                self.model_path = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

            self.llm_model = LlamaCppModel(
                model_path=self.model_path,
                n_ctx=self.settings.LLM_MAX_CONTEXT,
                n_gpu_layers=self.settings.LLM_GPU_LAYERS,
                n_threads=self.settings.LLM_THREADS,
                verbose=self.settings.LLM_VERBOSE
            )

            logger.info(f"LLM initialized: {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    async def generate_response(
        self,
        prompt: str,
        system_message: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate response from prompt."""
        try:
            return await self.llm_model.generate(
                prompt,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    async def generate_response_stream(
        self,
        prompt: str,
        system_message: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate response from prompt with streaming."""
        try:
            async for token in self.llm_model.generate_stream(
                prompt,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            ):
                yield token

        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.llm_model:
            return {"status": "not_initialized"}

        return {
            **self.llm_model.get_model_info(),
            "manager_settings": {
                "model_path": self.model_path,
                "max_context": self.settings.LLM_MAX_CONTEXT,
                "gpu_layers": self.settings.LLM_GPU_LAYERS,
                "threads": self.settings.LLM_THREADS
            }
        }


@lru_cache()
def get_llm_manager() -> LLMManager:
    """Get cached LLM manager instance."""
    return LLMManager()


async def generate_response(
    prompt: str,
    system_message: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """Generate response from prompt."""
    manager = get_llm_manager()
    return await manager.generate_response(
        prompt,
        system_message=system_message,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )


async def generate_response_stream(
    prompt: str,
    system_message: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    **kwargs
) -> AsyncGenerator[str, None]:
    """Generate response from prompt with streaming."""
    manager = get_llm_manager()
    async for token in manager.generate_response_stream(
        prompt,
        system_message=system_message,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    ):
        yield token


def get_model_info() -> Dict[str, Any]:
    """Get model information."""
    manager = get_llm_manager()
    return manager.get_model_info()