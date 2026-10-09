"""LLM abstraction layer for BEAM toolkit."""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass
import asyncio
from openai import AsyncOpenAI
import async_timeout
from tenacity import retry, wait_random_exponential, stop_after_attempt, wait_fixed


@dataclass
class Message:
    """Chat message format."""
    role: str
    content: str


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_NUM_COMPLETIONS = 1

    @abstractmethod
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Generate response asynchronously."""
        pass

    @abstractmethod
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Generate response synchronously."""
        pass


class LLMRegistry:
    """Registry for LLM implementations."""
    
    _registry: Dict[str, type] = {}
    _instances: Dict[str, BaseLLM] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an LLM class."""
        def decorator(llm_class):
            cls._registry[name] = llm_class
            return llm_class
        return decorator

    @classmethod
    def get(
        cls,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Get an LLM instance by model name."""
        if not model_name:
            model_name = "gpt-4o"
        
        # Determine which implementation to use
        if 'llama' in model_name.lower() or 'qwen' in model_name.lower():
            impl_name = 'local'
        elif 'deepseek' in model_name.lower():
            impl_name = 'deepseek'
        elif 'claude' in model_name.lower():
            impl_name = 'anthropic'
        else:
            impl_name = 'openai'
        
        # Create instance if not cached
        cache_key = f"{impl_name}:{model_name}"
        if cache_key not in cls._instances:
            if impl_name in cls._registry:
                cls._instances[cache_key] = cls._registry[impl_name](
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs
                )
            else:
                # Default to OpenAI-compatible
                cls._instances[cache_key] = OpenAILLM(
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs
                )
        
        return cls._instances[cache_key]

    @classmethod
    def keys(cls) -> List[str]:
        """Get registered LLM names."""
        return list(cls._registry.keys())


@LLMRegistry.register('openai')
class OpenAILLM(BaseLLM):
    """OpenAI-compatible LLM implementation."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def _get_client(self) -> AsyncOpenAI:
        """Get or create async client."""
        if self._client is None:
            kwargs = {}
            if self.api_key:
                kwargs['api_key'] = self.api_key
            if self.base_url:
                kwargs['base_url'] = self.base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        temperature = temperature or self.DEFAULT_TEMPERATURE
        
        # Convert to dict format if needed
        if messages and isinstance(messages[0], Message):
            msg_list = [{"role": m.role, "content": m.content} for m in messages]
        else:
            msg_list = messages
        
        client = self._get_client()
        async with async_timeout.timeout(600):
            completion = await client.chat.completions.create(
                model=self.model_name,
                messages=msg_list,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        
        return completion.choices[0].message.content

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Synchronous generation using asyncio."""
        return asyncio.get_event_loop().run_until_complete(
            self.agen(messages, max_tokens, temperature, num_comps)
        )


@LLMRegistry.register('deepseek')
class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM implementation."""

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.api_key = api_key or ""
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self._client = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client

    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        temperature = temperature or self.DEFAULT_TEMPERATURE
        
        if messages and isinstance(messages[0], Message):
            msg_list = [{"role": m.role, "content": m.content} for m in messages]
        else:
            msg_list = messages
        
        client = self._get_client()
        async with async_timeout.timeout(600):
            completion = await client.chat.completions.create(
                model=self.model_name,
                messages=msg_list,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        
        return completion.choices[0].message.content

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        return asyncio.get_event_loop().run_until_complete(
            self.agen(messages, max_tokens, temperature, num_comps)
        )


@LLMRegistry.register('local')
class LocalLLM(BaseLLM):
    """Local LLM implementation (vLLM, Ollama, etc.)."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.api_key = api_key or "API-KEY"
        self.base_url = base_url or "http://localhost:8000/v1"
        self._client = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        temperature = temperature or self.DEFAULT_TEMPERATURE
        
        if messages and isinstance(messages[0], Message):
            msg_list = [{"role": m.role, "content": m.content} for m in messages]
        else:
            msg_list = messages
        
        client = self._get_client()
        async with async_timeout.timeout(600):
            completion = await client.chat.completions.create(
                model=self.model_name,
                messages=msg_list,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        
        return completion.choices[0].message.content

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        return asyncio.get_event_loop().run_until_complete(
            self.agen(messages, max_tokens, temperature, num_comps)
        )


class LangChainLLMWrapper(BaseLLM):
    """Wrapper to use LangChain LLMs with BEAM."""

    def __init__(self, langchain_llm):
        """
        Args:
            langchain_llm: A LangChain BaseChatModel or LLM instance
        """
        self.llm = langchain_llm

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        lc_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                if msg.role == "system":
                    lc_messages.append(SystemMessage(content=msg.content))
                elif msg.role == "user":
                    lc_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    lc_messages.append(AIMessage(content=msg.content))
            elif isinstance(msg, dict):
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))
        
        response = await self.llm.ainvoke(lc_messages)
        return response.content

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        lc_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                if msg.role == "system":
                    lc_messages.append(SystemMessage(content=msg.content))
                elif msg.role == "user":
                    lc_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    lc_messages.append(AIMessage(content=msg.content))
            elif isinstance(msg, dict):
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))
        
        response = self.llm.invoke(lc_messages)
        return response.content
