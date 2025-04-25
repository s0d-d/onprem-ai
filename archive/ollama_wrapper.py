import aiohttp

class LLaMA3LLM:
    def __init__(self, endpoint="http://localhost:8080/completion"):
        self.endpoint = endpoint

    async def generate_reply(self, prompt: str, **kwargs) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json={"prompt": prompt}) as resp:
                result = await resp.json()
                return result.get("content", "")
