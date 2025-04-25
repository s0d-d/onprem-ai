import logging
import json
import asyncio
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

try:
    import aiohttp
except ImportError:
    logger.error("aiohttp is not installed. Install it with: pip install aiohttp")
    aiohttp = None


class OllamaLLM:
    """
    A LLM implementation using Ollama API.

    This implementation allows using local Ollama models with LiveKit agents.
    """

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: Optional[int] = None,
        context_window: Optional[int] = None,
        system_prompt: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize the Ollama LLM implementation.

        Args:
            model: The Ollama model to use (e.g., "llama3", "mistral", "mixtral")
            host: The Ollama API host URL
            temperature: Temperature for sampling (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            max_tokens: Maximum number of tokens to generate
            context_window: Context window size
            system_prompt: System prompt to use for all conversations
            timeout: Timeout for API requests in seconds
        """
        if aiohttp is None:
            raise ImportError("aiohttp is required for OllamaLLM. Install it with: pip install aiohttp")

        self.model = model
        self.host = host.rstrip("/")  # Remove trailing slash if present
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.system_prompt = system_prompt
        self.timeout = timeout

        # Check that the model exists
        logger.info(f"Initializing Ollama LLM with model: {model}")

    async def _verify_model_exists(self):
        """Verify that the specified model exists in Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.host}/api/tags",
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get models from Ollama: {response.status}")
                        return False

                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]

                    if self.model not in models:
                        logger.warning(f"Model {self.model} not found in Ollama. Available models: {models}")
                        return False

                    return True
        except Exception as e:
            logger.error(f"Error checking Ollama model: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the Ollama model.

        Args:
            prompt: The prompt to send to the model (used if messages is None)
            messages: List of message dictionaries with 'role' and 'content' keys
                     (OpenAI-style format, takes precedence over prompt if provided)
            stream: Whether to stream the response
            tools: JSON Schema function descriptions for tool use

        Returns:
            Dictionary with model response
        """
        if stream:
            # For streaming responses
            return await self._generate_stream(prompt, messages, tools)

        # Prepare API call parameters
        if messages:
            # Convert messages format for Ollama
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                # Map assistant role to Ollama's expected format
                if role == "assistant":
                    role = "assistant"
                formatted_messages.append({
                    "role": role,
                    "content": msg.get("content", "")
                })

            request_data = {
                "model": self.model,
                "messages": formatted_messages,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            }

            if self.system_prompt:
                request_data["system"] = self.system_prompt

            if self.max_tokens:
                request_data["options"]["num_predict"] = self.max_tokens

            endpoint = f"{self.host}/api/chat"
        else:
            # Simple prompt-based completion
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            }

            if self.system_prompt:
                request_data["system"] = self.system_prompt

            if self.max_tokens:
                request_data["options"]["num_predict"] = self.max_tokens

            endpoint = f"{self.host}/api/generate"

        # Add function calling if tools are provided
        if tools and len(tools) > 0:
            # Check if model supports function calling
            if not await self._check_function_calling_support():
                logger.warning(f"Model {self.model} may not support function calling")

            # Convert tools to Ollama's expected format
            # Note: Ollama's function calling format might differ from OpenAI
            # This is a best-effort implementation
            request_data["tools"] = tools

        # Make API call
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        return {
                            "error": f"Ollama API error: {response.status}",
                            "details": error_text
                        }

                    result = await response.json()

                    # Format response to be compatible with expected interface
                    if messages:  # Chat completion
                        formatted_response = {
                            "content": result.get("message", {}).get("content", ""),
                            "model": self.model,
                            "usage": {
                                "prompt_tokens": result.get("prompt_eval_count", 0),
                                "completion_tokens": result.get("eval_count", 0),
                                "total_tokens": (
                                    result.get("prompt_eval_count", 0) +
                                    result.get("eval_count", 0)
                                )
                            }
                        }

                        # Handle function calling response
                        if "tool_calls" in result.get("message", {}):
                            formatted_response["tool_calls"] = result["message"]["tool_calls"]

                    else:  # Text completion
                        formatted_response = {
                            "content": result.get("response", ""),
                            "model": self.model,
                            "usage": {
                                "prompt_tokens": result.get("prompt_eval_count", 0),
                                "completion_tokens": result.get("eval_count", 0),
                                "total_tokens": (
                                    result.get("prompt_eval_count", 0) +
                                    result.get("eval_count", 0)
                                )
                            }
                        }

                    return formatted_response

        except asyncio.TimeoutError:
            logger.error(f"Timeout while calling Ollama API (timeout={self.timeout}s)")
            return {"error": "Timeout while calling Ollama API"}
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return {"error": f"Error calling Ollama API: {str(e)}"}

    async def _generate_stream(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a streaming response from the Ollama model.

        Args:
            prompt: The prompt to send to the model (used if messages is None)
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: JSON Schema function descriptions for tool use

        Returns:
            Dictionary with complete model response (after collecting all chunks)
        """
        # Prepare API call parameters
        if messages:
            # Convert messages format for Ollama
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                # Map assistant role to Ollama's expected format
                if role == "assistant":
                    role = "assistant"
                formatted_messages.append({
                    "role": role,
                    "content": msg.get("content", "")
                })

            request_data = {
                "model": self.model,
                "messages": formatted_messages,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                },
                "stream": True
            }

            if self.system_prompt:
                request_data["system"] = self.system_prompt

            if self.max_tokens:
                request_data["options"]["num_predict"] = self.max_tokens

            endpoint = f"{self.host}/api/chat"
        else:
            # Simple prompt-based completion
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                },
                "stream": True
            }

            if self.system_prompt:
                request_data["system"] = self.system_prompt

            if self.max_tokens:
                request_data["options"]["num_predict"] = self.max_tokens

            endpoint = f"{self.host}/api/generate"

        # Add function calling if tools are provided
        if tools and len(tools) > 0:
            request_data["tools"] = tools

        # Variables to accumulate streaming response
        full_response = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        tool_calls = None

        # Make API call
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama streaming API error: {response.status} - {error_text}")
                        return {
                            "error": f"Ollama API error: {response.status}",
                            "details": error_text
                        }

                    # Process streaming response
                    async for line in response.content:
                        if not line.strip():
                            continue

                        try:
                            chunk = json.loads(line)

                            if messages:  # Chat completion
                                if "message" in chunk:
                                    chunk_text = chunk["message"].get("content", "")
                                    full_response += chunk_text

                                    # Check for tool calls
                                    if "tool_calls" in chunk["message"]:
                                        tool_calls = chunk["message"]["tool_calls"]
                            else:  # Text completion
                                chunk_text = chunk.get("response", "")
                                full_response += chunk_text

                            # Track token usage
                            if "prompt_eval_count" in chunk:
                                total_prompt_tokens = chunk["prompt_eval_count"]
                            if "eval_count" in chunk:
                                total_completion_tokens = chunk["eval_count"]

                            # You can yield chunks here if needed

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON from chunk: {line}")

                    # Format final response
                    if messages:  # Chat completion
                        formatted_response = {
                            "content": full_response,
                            "model": self.model,
                            "usage": {
                                "prompt_tokens": total_prompt_tokens,
                                "completion_tokens": total_completion_tokens,
                                "total_tokens": total_prompt_tokens + total_completion_tokens
                            }
                        }

                        if tool_calls:
                            formatted_response["tool_calls"] = tool_calls

                    else:  # Text completion
                        formatted_response = {
                            "content": full_response,
                            "model": self.model,
                            "usage": {
                                "prompt_tokens": total_prompt_tokens,
                                "completion_tokens": total_completion_tokens,
                                "total_tokens": total_prompt_tokens + total_completion_tokens
                            }
                        }

                    return formatted_response

        except asyncio.TimeoutError:
            logger.error(f"Timeout while calling Ollama streaming API (timeout={self.timeout}s)")
            return {"error": "Timeout while calling Ollama API"}
        except Exception as e:
            logger.error(f"Error calling Ollama streaming API: {e}")
            return {"error": f"Error calling Ollama API: {str(e)}"}

    async def _check_function_calling_support(self) -> bool:
        """Check if the selected model supports function calling."""
        # This is a best-effort check and may not be accurate
        # Ollama may not expose model capabilities via API
        # Models like llama3, claude-3, gpt-4 typically support function calling
        function_capable_models = [
            "llama3", "llama-3", "llama3-",
            "mistral-", "mixtral-",
            "claude-", "claude3",
            "gpt-4", "gpt4", "openai"
        ]

        for prefix in function_capable_models:
            if self.model.lower().startswith(prefix.lower()):
                return True

        return False