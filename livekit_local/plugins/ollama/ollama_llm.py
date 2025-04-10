import os
import json
import asyncio
import httpx
from typing import List, Dict, Any, Optional, AsyncIterator, Union
from livekit.agents import Plugin
from livekit.agents.llm.llm import (
    LLM,
    LLMCapabilities,
    ChatContext,
    LLMTools,
    LLMMessage,
    ChatResponse,
    ChatResponseChunk,
    Role
)

class OllamaLLM(LLM):
    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        super().__init__(
            capabilities=LLMCapabilities(
                streaming=True,
                tools=True,
                image_input=False
            )
        )
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    def _prepare_messages(self, context: ChatContext) -> List[Dict[str, Any]]:
        """Convert LiveKit chat context to Ollama-compatible messages."""
        formatted_messages = []

        for message in context.messages:
            role = "user" if message.role == Role.USER else "assistant"
            content = message.content

            formatted_messages.append({
                "role": role,
                "content": content
            })

        return formatted_messages

    def _prepare_tools(self, tools: Optional[LLMTools]) -> Optional[str]:
        """Format tools for Ollama if supported by the model."""
        if not tools or not tools.available_tools:
            return None

        # Format tools as a system prompt instruction
        # This is a simplified approach since Ollama doesn't have native tool calling
        tools_description = "You have access to the following tools:\n\n"

        for tool in tools.available_tools:
            tools_description += f"- {tool.name}: {tool.description}\n"
            if tool.parameters:
                tools_description += "  Parameters:\n"
                for param_name, param in tool.parameters.items():
                    param_type = param.get("type", "string")
                    param_desc = param.get("description", "")
                    tools_description += f"    - {param_name} ({param_type}): {param_desc}\n"

        tools_description += "\nTo use a tool, respond with JSON in the following format:\n"
        tools_description += '{"tool": "tool_name", "parameters": {"param1": "value1", ...}}\n'

        return tools_description

    async def chat(
        self,
        context: ChatContext,
        tools: Optional[LLMTools] = None
    ) -> ChatResponse:
        """Send a chat request to Ollama."""
        messages = self._prepare_messages(context)
        tools_instruction = self._prepare_tools(tools)

        # Add system message with tools instruction if provided
        if tools_instruction:
            messages.insert(0, {
                "role": "system",
                "content": tools_instruction
            })

        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            }
        }

        if self.max_tokens:
            payload["options"]["num_predict"] = self.max_tokens
        if self.top_p:
            payload["options"]["top_p"] = self.top_p
        if self.top_k:
            payload["options"]["top_k"] = self.top_k

        # Send request to Ollama
        response = await self.client.post(
            f"{self.host}/api/chat",
            json=payload
        )
        response.raise_for_status()

        result = response.json()
        message = result.get("message", {})
        content = message.get("content", "")

        # Check if response is a tool call (in JSON format)
        tool_name = None
        tool_params = None

        try:
            if content.strip().startswith("{") and content.strip().endswith("}"):
                tool_data = json.loads(content)
                if "tool" in tool_data and "parameters" in tool_data:
                    tool_name = tool_data["tool"]
                    tool_params = tool_data["parameters"]
        except (json.JSONDecodeError, KeyError):
            pass

        # Create and return the chat response
        return ChatResponse(
            message=LLMMessage(
                role=Role.ASSISTANT,
                content=content
            ),
            tool_calls=[{"name": tool_name, "parameters": tool_params}] if tool_name else None
        )

    async def chat_stream(
        self,
        context: ChatContext,
        tools: Optional[LLMTools] = None
    ) -> AsyncIterator[ChatResponseChunk]:
        """Stream a chat response from Ollama."""
        messages = self._prepare_messages(context)
        tools_instruction = self._prepare_tools(tools)

        # Add system message with tools instruction if provided
        if tools_instruction:
            messages.insert(0, {
                "role": "system",
                "content": tools_instruction
            })

        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
            }
        }

        if self.max_tokens:
            payload["options"]["num_predict"] = self.max_tokens
        if self.top_p:
            payload["options"]["top_p"] = self.top_p
        if self.top_k:
            payload["options"]["top_k"] = self.top_k

        # Send streaming request to Ollama
        async with self.client.stream(
            "POST",
            f"{self.host}/api/chat",
            json=payload,
            timeout=300.0
        ) as response:
            response.raise_for_status()

            accumulated_text = ""

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                try:
                    chunk_data = json.loads(line)
                    chunk_content = chunk_data.get("message", {}).get("content", "")

                    if chunk_content:
                        accumulated_text += chunk_content

                        yield ChatResponseChunk(
                            delta=chunk_content,
                            message=LLMMessage(
                                role=Role.ASSISTANT,
                                content=accumulated_text
                            )
                        )
                except json.JSONDecodeError:
                    continue

            # Check if accumulated_text might be a tool call
            tool_name = None
            tool_params = None

            try:
                if accumulated_text.strip().startswith("{") and accumulated_text.strip().endswith("}"):
                    tool_data = json.loads(accumulated_text)
                    if "tool" in tool_data and "parameters" in tool_data:
                        tool_name = tool_data["tool"]
                        tool_params = tool_data["parameters"]

                        # Yield a final chunk with the tool call
                        yield ChatResponseChunk(
                            delta="",
                            message=LLMMessage(
                                role=Role.ASSISTANT,
                                content=accumulated_text
                            ),
                            tool_calls=[{"name": tool_name, "parameters": tool_params}]
                        )
            except (json.JSONDecodeError, KeyError):
                pass


# Register the plugin
class OllamaLLMPlugin(Plugin):
    @classmethod
    def load(cls):
        return OllamaLLM

Plugin.register_plugin("ollama", OllamaLLMPlugin)