"""
LLM Providers: HuggingFace and vLLM Integration

Supports:
- HuggingFace Inference API
- HuggingFace Local Models
- vLLM Server (OpenAI-compatible)
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class LLMProvider(ABC):
    """Abstract base for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion with message history."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass


class HuggingFaceLLM(LLMProvider):
    """HuggingFace Inference API provider."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        from huggingface_hub import InferenceClient
        
        self._model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        api_key = api_key or os.environ.get("HF_TOKEN")
        if not api_key:
            raise ValueError("HF_TOKEN environment variable or api_key required")
        
        self.client = InferenceClient(model_name, token=api_key)
        print(f"[HuggingFace] Initialized: {model_name}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate using chat completion format."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        
        try:
            response = self.client.chat_completion(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM Error] {type(e).__name__}: {e}"


class HuggingFaceLocalLLM(LLMProvider):
    """Local HuggingFace model loading."""
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = "auto",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        self._model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        print(f"[LocalLLM] Loading: {model_name}...")
        
        # Quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        print(f"[LocalLLM] Loaded on {self.model.device}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Format messages and generate."""
        # Simple chat template
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n\n".join(prompt_parts)
        
        return self.generate(prompt, **kwargs)


class VLLMProvider(LLMProvider):
    """
    vLLM Server Provider (OpenAI-compatible API).
    
    Optimized for local GPU inference (NVIDIA L4, etc.)
    Supports both chat and raw completion endpoints.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "default",
        api_key: str = "EMPTY",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        use_chat_endpoint: bool = True,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai package: pip install openai")
        
        self._model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_chat_endpoint = use_chat_endpoint
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        print(f"[vLLM] Connected to {base_url}")
        print(f"[vLLM] Model: {model_name}, Chat endpoint: {use_chat_endpoint}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text - uses appropriate endpoint based on config."""
        if self.use_chat_endpoint:
            messages = [{"role": "user", "content": prompt}]
            return self.chat(messages, **kwargs)
        else:
            return self._raw_completion(prompt, **kwargs)
    
    def _raw_completion(self, prompt: str, **kwargs) -> str:
        """Raw completion endpoint (for base models)."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        
        try:
            response = self.client.completions.create(
                model=self._model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f"[vLLM Error] {type(e).__name__}: {e}"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion endpoint (for instruct models)."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[vLLM Error] {type(e).__name__}: {e}"


# ============================================================================
# Factory Function
# ============================================================================

def create_llm(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    provider: str = "huggingface",
    **kwargs
) -> LLMProvider:
    """
    Factory function to create LLM provider.
    
    Args:
        model_name: Model identifier
        provider: "huggingface", "local", or "vllm"
        **kwargs: Provider-specific arguments
    
    Returns:
        LLMProvider instance
    """
    providers = {
        "huggingface": HuggingFaceLLM,
        "hf": HuggingFaceLLM,
        "local": HuggingFaceLocalLLM,
        "vllm": VLLMProvider,
    }
    
    provider_class = providers.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider}. Options: {list(providers.keys())}")
    
    return provider_class(model_name=model_name, **kwargs)
